import os
import json
import torch
import logging
import argparse
import shutil
import math
import wandb
import PIL.Image
import numpy as np

from typing import List, Dict, Any
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from accelerate import Accelerator
from einops import rearrange
from transformers import (
    set_seed,
)
from transformers import AutoModelForCausalLM
from datasets import load_dataset, Features, Value, Sequence
from janus.models import VLChatProcessor, ActionTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')

from dataclasses import dataclass
@dataclass
class VLChatProcessorOutput():
    sft_format: str
    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    num_image_tokens: torch.IntTensor

    def __len__(self):
        return len(self.input_ids)

def get_custom_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps, 
    num_training_steps, 
    min_lr_ratio=0.0, 
    num_cycles=0.5
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * 2 * num_cycles * progress))
        scaled_factor = (1 - min_lr_ratio) * cosine_factor + min_lr_ratio
        return scaled_factor

    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)

def get_learning_rate(step, initial_lr, num_warmup_steps, num_training_steps, min_lr_ratio, num_cycles=0.5):
    if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps)) * initial_lr
    progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * 2 * num_cycles * progress))
    scaled_factor = (1 - min_lr_ratio) * cosine_factor + min_lr_ratio
    return scaled_factor * initial_lr

def create_component_indexes(seq_len, action_len=7):
    latent_indexes = torch.arange(0, seq_len - action_len)
    action_indexes = torch.arange(seq_len - action_len, seq_len)
    return latent_indexes, action_indexes


class TrainingMetrics:
    def __init__(self, device):
        self.n_step = 0
        self.action_total = torch.Tensor([0]).to(device=device)
        self.action_loss = torch.Tensor([0]).to(device=device)
        self.sim_loss = torch.Tensor([0]).to(device=device)
        self.world_size = dist.get_world_size()

    def __call__(self, action_loss, sim_loss):
        return self.update(action_loss, sim_loss)

    def update(self, action_loss, sim_loss):
        self.n_step += 1
        with torch.no_grad():
            self.action_loss += action_loss.item()
            self.sim_loss += sim_loss.item()

    def get_metric(self, reset=True):
        dist.all_reduce(self.action_loss, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.sim_loss, op=torch.distributed.ReduceOp.SUM)

        action_loss = self.action_loss.item() / (self.world_size * self.n_step)
        sim_loss = self.sim_loss.item() / (self.world_size * self.n_step)

        if reset:
            self.n_step = 0
            self.action_loss.fill_(0)
            self.sim_loss.fill_(0)
        return action_loss, sim_loss


class SftDataset(Dataset):
    def __init__(self, config, processor,accelerator, model):
        self.model = model
        self.config = config
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.action_tokenizer = ActionTokenizer(self.tokenizer, need_to_sub=3) # 3 for latent spetial tokens
        self.accelerator = accelerator
        self.image_len = 576

        with open(config.data_path,'r') as f:
            self.data = json.load(f)

        statistics_path = config.data_path.replace(".json", "_statistics.json")
        with open(statistics_path, 'r') as f:
            self.stats_data = json.load(f)

        self.dataset_name = next(iter(self.stats_data))
        self.action_mask = np.array(self.stats_data[self.dataset_name]['action']['mask'])
        self.action_min = np.array(self.stats_data[self.dataset_name]['action']['q01'])
        self.action_max = np.array(self.stats_data[self.dataset_name]['action']['q99'])
        self.state_mask = np.array(self.stats_data[self.dataset_name]['state']['mask'])
        self.state_min = np.array(self.stats_data[self.dataset_name]['state']['q01'])
        self.state_max = np.array(self.stats_data[self.dataset_name]['state']['q99'])

        self.img_dir = os.path.dirname(config.data_path)
        accelerator.print(f'Total data amount: {len(self.data)}')
  
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.data[index]

    def process_image(self,image_paths):
        images = [PIL.Image.open(image_path).convert("RGB") for image_path in image_paths]
        images_outputs = self.processor.image_processor(images, return_tensors="pt")
        return images_outputs['pixel_values']

    def sample_beta(self, alpha, beta, bsize, device):
        alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
        beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
        dist = torch.distributions.Beta(alpha_t, beta_t)
        samples = dist.sample((bsize,))
        return samples.to(dtype=torch.bfloat16)

    def sample_time(self, bsize, device):
        time_beta = self.sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.bfloat16, device=device)

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.bfloat16,
            device=device,
        )

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        
        if self.config.use_latent: 
            # Image
            latent_images_nested = [
                [os.path.join(self.img_dir, img) for img in x['output_image']]
                for x in batch
            ]
            latent_pixel_values_list = [
                self.process_image(img_list).to(torch.bfloat16)
                for img_list in latent_images_nested
            ]
            latent_pixel_values = torch.stack(latent_pixel_values_list, dim=0)

            # Point Cloud
            latent_pc_nested = [
                [os.path.join(self.img_dir, pc) for pc in x['output_pointcloud']]
                for x in batch
            ]
            latent_pc_list = []
            for sample_pc_paths in latent_pc_nested:
                sample_pcs = []
                for pc_path in sample_pc_paths:
                    pc_data = np.load(pc_path)
                    pc_tensor = torch.from_numpy(pc_data).float()
                    sample_pcs.append(pc_tensor)
                latent_pc_list.append(torch.stack(sample_pcs)) 
            latent_pointclouds = torch.stack(latent_pc_list).to(torch.bfloat16) 
        
            # State
            latent_state_ids_list = []
            for x in batch:
                current_sample_state_ids = []
                for single_state in x['output_state']:
                    state_arr = np.array(single_state, dtype=np.float32)
                    normalized_state = np.where(
                        self.state_mask,
                        np.clip(2 * (state_arr - self.state_min) / (self.state_max - self.state_min + 1e-8) - 1, -1, 1),
                        state_arr
                    )
                    state_token_str = self.action_tokenizer(normalized_state)
                    state_ids = self.tokenizer.encode(state_token_str, add_special_tokens=False)
                    current_sample_state_ids.append(state_ids)
                latent_state_ids_list.append(torch.LongTensor(current_sample_state_ids))
            latent_state_ids = torch.stack(latent_state_ids_list, dim=0)
        else:
            latent_pixel_values = None
            latent_pointclouds = None
            latent_state_ids = None

        input_img_tokens = self.processor.image_start_tag + self.processor.image_tag * self.processor.num_image_tokens + self.processor.image_end_tag

        # Generate noisy actions and timesteps for flow matching
        actions = [x['action'] for x in batch]
        actions = np.array(actions, dtype=np.float32).reshape(len(actions), -1, self.config.action_dim)
        normalized_actions = np.where(
            self.action_mask,
            np.clip(2 * (actions - self.action_min) / (self.action_max - self.action_min + 1e-8) - 1, -1, 1),
            actions
        )
        normalized_actions = torch.tensor(normalized_actions)

        time = self.sample_time(normalized_actions.shape[0], normalized_actions.device)
        time_expanded = time[:, None, None]

        noise = self.sample_noise(normalized_actions.shape, normalized_actions.device)

        x_t = (time_expanded * noise + (1 - time_expanded) * normalized_actions)
        u_t = (noise - normalized_actions)

        # Init latent special tokens
        latent_start_str = "<|latent_start|>"
        latent_pad_str = "<|latent_pad|>" * self.config.latent_size
        latent_end_str = "<|latent_end|>"

        # Prepare data in batch
        pre_data = []
        for x in batch:
            slow_imgs = x.get('input_image_slow', [])
            fast_imgs = x.get('input_image_fast', [])

            if not slow_imgs and not fast_imgs and 'input_image' in x:
                fast_imgs = x['input_image'] # default as fast images
            
            slow_img_len = len(slow_imgs)
            fast_img_len = len(fast_imgs)
            all_input_imgs = slow_imgs + fast_imgs

            state_tokens_slow = ""
            state_tokens_fast = ""
            if self.config.robot_state:
                state_slow = np.array(x['state_slow'], dtype=np.float32)
                state_fast = np.array(x['state_fast'], dtype=np.float32)
                normalized_state_slow = np.where(
                    self.state_mask,
                    np.clip(2 * (state_slow - self.state_min) / (self.state_max - self.state_min + 1e-8) - 1, -1, 1),
                    state_slow
                )
                normalized_state_fast = np.where(
                    self.state_mask,
                    np.clip(2 * (state_fast - self.state_min) / (self.state_max - self.state_min + 1e-8) - 1, -1, 1),
                    state_fast
                )
                state_tokens_slow += self.action_tokenizer(normalized_state_slow)
                state_tokens_fast += self.action_tokenizer(normalized_state_fast)

            latent_str = latent_start_str + latent_pad_str + latent_end_str if self.config.use_latent else ""

            input_slow_img_tokens = input_img_tokens * slow_img_len
            input_fast_img_tokens = input_img_tokens * fast_img_len

            prompts = input_slow_img_tokens + x['input_prompt'] + state_tokens_slow + latent_str + input_fast_img_tokens + state_tokens_fast

            conversation = [
                {"role": "<|User|>","content": prompts},
            ]

            pre_format = self.processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=self.processor.sft_format,
                system_prompt="",
            )
            sft_format = pre_format
            
            if len(all_input_imgs) > 0:
                encoder_pixel_values = self.process_image([os.path.join(self.img_dir, input_img) for input_img in all_input_imgs])
                num_image_tokens = [self.image_len] * len(all_input_imgs)
            else:
                encoder_pixel_values = None
                num_image_tokens = []
            
            input_ids = torch.LongTensor(self.processor.tokenizer.encode(sft_format))
            pre_data.append(
                VLChatProcessorOutput(
                    sft_format=sft_format, 
                    pixel_values=encoder_pixel_values, 
                    input_ids=input_ids, 
                    num_image_tokens=num_image_tokens
                )
            )

        if len(pre_data) > 0:
            prepare_inputs = self.processor.batchify(pre_data)

        return {
            "input_ids": prepare_inputs.input_ids,
            "encoder_pixel_values": prepare_inputs.pixel_values.to(torch.bfloat16),
            "latent_pixel_values": latent_pixel_values,
            "latent_pointclouds": latent_pointclouds,
            "latent_state_ids": latent_state_ids,
            "noisy_actions": x_t,
            "target": u_t,
            "timesteps": time,
            "fast_img_len": fast_img_len,
            "attention_mask": prepare_inputs.attention_mask,
            "images_seq_mask": prepare_inputs['images_seq_mask'],
            "images_emb_mask": prepare_inputs['images_emb_mask'],
        }


def save_checkpoint(
    model,
    processor,
    accelerator: Accelerator,
    args: argparse.Namespace,
    epoch: int,
    step: int,
    global_step: int,
    is_last: bool = False,
    stats_data = None
) -> None:

    save_dir = os.path.join(args.output_dir, f"checkpoint-{epoch}-{global_step}")
    
    if accelerator.is_main_process:
        # Manage checkpoint numbers
        checkpoint_files = [f for f in os.listdir(args.output_dir) if f.startswith("checkpoint-")]
        if args.max_ckpts > 0 and len(checkpoint_files) >= args.max_ckpts:
            oldest_ckpt = min(checkpoint_files, key=lambda x: os.path.getctime(os.path.join(args.output_dir, x)))
            shutil.rmtree(os.path.join(args.output_dir, oldest_ckpt))

        os.makedirs(save_dir, exist_ok=True)
        output_dir = os.path.join(save_dir, 'tfmr')

        model.save_pretrained(output_dir, state_dict=accelerator.get_state_dict(model))
        processor.save_pretrained(output_dir)

        with open(os.path.join(save_dir, 'stats_data.json'), 'w') as f:
            json.dump(stats_data, f, indent=2)
            
        logger.info(f"Statistics have been saved to {os.path.join(save_dir, 'stats_data.json')}")

    accelerator.wait_for_everyone()
    logger.info(f'Checkpoint {epoch}-{global_step} saved successfully')


def train(args: argparse.Namespace) -> None:
    accelerator = Accelerator(
        mixed_precision='bf16',
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    # Set random seed
    set_seed(args.seed)

    if accelerator.is_main_process:
        wandb.init(
            project=args.experiment_name,
            name=args.run_name,
            config=args,
            dir=args.log_dir,
        )
    accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.train_bsz_per_gpu
    accelerator.state.deepspeed_plugin.deepspeed_config['train_batch_size'] = (
        args.train_bsz_per_gpu * 
        dist.get_world_size() * 
        accelerator.gradient_accumulation_steps
    )

    processor = VLChatProcessor.from_pretrained(
        args.pretrain_path,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrain_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        flow = True,
        action_dim=args.action_dim,
        action_chunk=args.action_chunk,
        use_pointcloud=True,
        use_latent=args.use_latent,
        ignore_mismatched_sizes=True,
    )
    model_action = AutoModelForCausalLM.from_pretrained(
        args.pretrain_action_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        flow = True,
        action_dim=args.action_dim,
        action_chunk=args.action_chunk,
        use_pointcloud=True,
        use_latent=args.use_latent,
        ignore_mismatched_sizes=True,
    )

    model.projector_3d.initialize_weights()
    model.load_encoder_to_pointcloud_embedder(args.pointcloud_embedder_ckpt_path) # load pointcloud embedder

    for name, param in model.named_parameters():
        if '_action' in name:
            if args.load_action_from_latent and name.endswith('.weight'):
                base_name = name.replace('_action', '')
                if base_name in model.state_dict():
                    param.data.copy_(model.state_dict()[base_name])
                    accelerator.print(f"Initialized {name} from {base_name}")
            elif args.load_action_from_pretrain and name.endswith('.weight'):
                base_name = name.replace('_action', '')
                if base_name in model_action.state_dict():
                    param.data.copy_(model_action.state_dict()[base_name])
                    accelerator.print(f"Initialized {name} from {base_name}")
            param.requires_grad = True
        elif 'x_embedder' in name or 'state_embedder' in name or 't_embedder' in name or 'final_layer' in name:
            param.data.copy_(model_action.state_dict()[name])
            accelerator.print(f"Initialized {name} from {name}")
            param.requires_grad = True
        else:
            if any(name.startswith(prefix) for prefix in ["vision_model", "aligner", "gen_vision_model"]) or 'pointcloud_embedder' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    accelerator.print("\n==== Parameter Freeze Status ====\n")
    for name, param in model.named_parameters():
        status = "TRAINABLE" if param.requires_grad else "FROZEN"
        accelerator.print(f"{name:60}  {status}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    accelerator.print(f"Load action from latent: {args.load_action_from_latent}")
    accelerator.print(f"Load action from pretrain: {args.load_action_from_pretrain}")
    accelerator.print(f"Total parameters: {total_params/1e9:.2f}B")
    accelerator.print(f"Trainable parameters: {trainable_params/1e9:.2f}B")
    accelerator.print(f"Non-trainable parameters: {non_trainable_params/1e9:.2f}B")
    accelerator.print(f"Trainable ratio: {trainable_params/total_params*100:.2f}%")

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    train_dataset = SftDataset(args, processor, accelerator, model)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_bsz_per_gpu,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=8
    )

    num_training_steps = int(len(train_dataloader) * args.n_epochs) // accelerator.gradient_accumulation_steps // dist.get_world_size()
    lr_scheduler = get_custom_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_rates * num_training_steps),
        num_training_steps=num_training_steps,
        min_lr_ratio=args.min_lr_ratio
    )
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    metric = TrainingMetrics(device=accelerator.device)
    model.train()
    global_step = 0

    for epoch in range(0, args.n_epochs):
        train_iter = tqdm(train_dataloader, total=len(train_dataloader)) if accelerator.is_main_process else train_dataloader
        for batch in train_iter:
            inputs_embeds = model.prepare_inputs_embeds(
                    input_ids=batch['input_ids'],
                    pixel_values=batch['encoder_pixel_values'],
                    images_emb_mask=batch['images_emb_mask'],
                    images_seq_mask=batch['images_seq_mask']
                )
            
            # Add flow matching related tokens (time + action)
            noisy_actions = model.x_embedder(batch['noisy_actions'].to(inputs_embeds.dtype))
            timesteps = model.t_embedder(batch['timesteps'].to(inputs_embeds.dtype)).unsqueeze(1)
            inputs_embeds = torch.cat([
                inputs_embeds,
                timesteps,
                noisy_actions,
            ], dim=1)
            batch['attention_mask'] = torch.cat([
                batch['attention_mask'],
                torch.ones((batch['attention_mask'].shape[0], timesteps.shape[1]), dtype=torch.bool).to(batch['attention_mask'].device),
                torch.ones((batch['attention_mask'].shape[0], noisy_actions.shape[1]), dtype=torch.bool).to(batch['attention_mask'].device),
            ], dim=1)

            fast_img_len = batch['fast_img_len']
            action_len = 1 + 578 * fast_img_len + 1 + args.action_chunk
            latent_indexes, action_indexes = create_component_indexes(inputs_embeds.shape[1], action_len) 
            
            if args.use_latent:
                bs, n_future = batch['latent_pixel_values'].shape[0:2]
                num_frames = batch['latent_pixel_values'].shape[1]
                tokens_per_modality = (args.latent_size - num_frames) // (num_frames * 2)

                # Process helper images
                helper_images = rearrange(batch['latent_pixel_values'], "b n c h w -> (b n) c h w")
                img_embeds_flat = model.aligner(model.vision_model(helper_images)) 
                img_embeds = rearrange(img_embeds_flat, "(b n) t d -> b n t d", b=bs, n=n_future) 
                # Compression (Average Pooling)
                T_vis = img_embeds.shape[2]
                group_size_img = T_vis // tokens_per_modality
                img_chunks = torch.split(img_embeds, group_size_img, dim=2)
                compressed_imgs = torch.cat([c.mean(dim=2, keepdim=True) for c in img_chunks[:tokens_per_modality]], dim=2) 

                # Process helper pointclouds
                helper_pcs = batch['latent_pointclouds'].to(img_embeds.device).to(img_embeds.dtype) 
                helper_pcs_flat = rearrange(helper_pcs, "b n p c -> (b n) p c") 
                pc_embeds_flat, pc_centers = model.pointcloud_embedder(helper_pcs_flat)
                pc_embeds_projected = model.projector_3d(pc_embeds_flat.to(torch.bfloat16) ) 
                pc_embeds = rearrange(pc_embeds_projected, "(b n) t d -> b n t d", b=bs, n=n_future) 
                # Compression (Average Pooling)
                T_pc = pc_embeds.shape[2]
                group_size_pc = max(1, T_pc // tokens_per_modality) 
                pc_chunks = torch.split(pc_embeds, group_size_pc, dim=2)
                compressed_pcs_list = []
                for i in range(tokens_per_modality):
                    if i < len(pc_chunks):
                        compressed_pcs_list.append(pc_chunks[i].mean(dim=2, keepdim=True))
                    else:
                        compressed_pcs_list.append(pc_chunks[-1].mean(dim=2, keepdim=True))
                compressed_pcs = torch.cat(compressed_pcs_list, dim=2) 

                # Process helper states
                state_ids = batch['latent_state_ids'].to(img_embeds.device)
                state_embeds_full = model.language_model.model.embed_tokens(state_ids) 
                # Compress state embedding (Average Pooling)
                compressed_state = state_embeds_full.mean(dim=2, keepdim=True) 

                combined_embeds = torch.cat([compressed_imgs, compressed_pcs, compressed_state], dim=2)
                compressed_latent_embeds = rearrange(combined_embeds, "b n k d -> b (n k) d")
                compressed_latent_embeds = compressed_latent_embeds.to(inputs_embeds.dtype)
                
                latent_start_id = processor.tokenizer.convert_tokens_to_ids("<|latent_start|>")
                latent_pad_id = processor.tokenizer.convert_tokens_to_ids("<|latent_pad|>")
                
                pad_mask = (batch['input_ids'] == latent_pad_id)
                extra_len = inputs_embeds.shape[1] - pad_mask.shape[1]
                if extra_len > 0:
                    extra_mask = torch.zeros(
                        (pad_mask.shape[0], extra_len), 
                        dtype=torch.bool, 
                        device=pad_mask.device
                    )
                    pad_mask = torch.cat([pad_mask, extra_mask], dim=1)
                
                if pad_mask.sum() != compressed_latent_embeds.numel() // compressed_latent_embeds.shape[-1]:
                    logger.warning("Latent pad count mismatch! Falling back to safe replacement.")

                input_latents = compressed_latent_embeds.detach()
                
                if pad_mask.sum() == input_latents.numel() // input_latents.shape[-1]:
                    inputs_embeds[pad_mask] = input_latents.reshape(-1, input_latents.shape[-1])
                else:
                    logger.warning(f"Latent count mismatch! Mask: {pad_mask.sum()}, GT: {input_latents.shape[1] * bs}")

                outputs = model.language_model.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=batch['attention_mask'],
                    return_dict=True,
                    use_cache=False, 
                    latent_indexes=latent_indexes.to(inputs_embeds.device), 
                    action_indexes=action_indexes.to(inputs_embeds.device),
                    use_latent=args.use_latent,
                )
                hidden_states = outputs.last_hidden_state
                
                pred_embeddings_list = []
                for b in range(inputs_embeds.shape[0]):
                    start_idx = (batch['input_ids'][b] == latent_start_id).nonzero(as_tuple=True)[0]
                    pad_idxs = (batch['input_ids'][b] == latent_pad_id).nonzero(as_tuple=True)[0]
                    if len(pad_idxs) > 0:
                        pred_input_idxs = torch.cat([start_idx, pad_idxs[:-1]])
                    else:
                        pred_input_idxs = start_idx
                    current_preds = hidden_states[b, pred_input_idxs, :]
                    pred_embeddings_list.append(current_preds)

                inferred_embeddings_all = torch.stack(pred_embeddings_list, dim=0)

                similarity = F.cosine_similarity(
                    inferred_embeddings_all.to(torch.float32), 
                    compressed_latent_embeds.to(torch.float32), 
                    dim=-1
                ).mean()
                sim_loss = 1.0 - similarity

                # use the inferred embeddings to replace the latent embeddings
                if pad_mask.sum() == inferred_embeddings_all.numel() // inferred_embeddings_all.shape[-1]:
                    inputs_embeds[pad_mask] = inferred_embeddings_all.reshape(-1, inferred_embeddings_all.shape[-1])
                else:
                    logger.warning("Latent count mismatch in Stage 2 (Second Forward). Action training might be faulty.")

                # forward for action prediction
                outputs = model.language_model.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=batch['attention_mask'],
                    return_dict=True,
                    use_cache=False,
                    latent_indexes=latent_indexes.to(inputs_embeds.device),
                    action_indexes=action_indexes.to(inputs_embeds.device),
                    use_latent=args.use_latent,
                )
                hidden_states = outputs.last_hidden_state

                # calculate multimodal loss
                # action loss (cross entropy)
                predicted_noise = model.final_layer(hidden_states)[:, -(batch['target'].shape[1]):, :] # the last token is noise
                action_loss = nn.MSELoss()(predicted_noise, batch['target'].to(predicted_noise.dtype))
                loss = action_loss + sim_loss
                metric(action_loss, sim_loss)
            else:
                latent_indexes=torch.arange(0, 0).to(inputs_embeds.device)
                action_indexes=torch.arange(0, inputs_embeds.shape[1]).to(inputs_embeds.device)
                outputs = model.language_model.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=batch['attention_mask'],
                    return_dict=True,
                    use_cache=False,
                    latent_indexes=latent_indexes,
                    action_indexes=action_indexes,
                    use_latent=args.use_latent,
                )
                hidden_states = outputs.last_hidden_state
                
                predicted_noise = model.final_layer(hidden_states)[:, -(batch['target'].shape[1]):, :] # the last token is noise
                action_loss = nn.MSELoss()(predicted_noise, batch['target'].to(predicted_noise.dtype))
                loss = action_loss
                sim_loss = torch.tensor(0.0) # dummy sim loss for logging consistency
                metric(action_loss, sim_loss)

            accelerator.backward(loss)
            if (global_step + 1) % accelerator.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                action_loss, sim_loss= metric.get_metric() if args.use_latent else metric.get_metric_action()
                if accelerator.is_main_process:
                    train_iter.set_postfix(
                        epoch=epoch,
                        step=global_step,
                        total_steps=len(train_dataloader),
                        skip=accelerator.optimizer_step_was_skipped,
                        length=len(batch["input_ids"][0]),
                        action_loss=f"{action_loss:.6f}",
                        sim_loss=f"{sim_loss:.6f}",
                        lr=f"{lr_scheduler.get_last_lr()[0]:.2e}"
                    )
                    wandb.log({
                        'action_loss': action_loss,
                        'sim_loss': sim_loss,
                        'lr': lr_scheduler.get_last_lr()[0]
                    }, step=global_step)
            global_step += 1

        if ((epoch + 1) % args.save_freq == 0) or (epoch == args.n_epochs-1):
            accelerator.wait_for_everyone()
            save_checkpoint(
                model=model,
                processor=processor, 
                accelerator=accelerator,
                args=args,
                epoch=epoch,
                step=global_step-1,
                global_step=global_step,
                is_last=(epoch == args.n_epochs-1),
                stats_data=train_dataset.stats_data,
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-training parameter configuration')
    
    # Experiment settings
    parser.add_argument('--experiment_name', type=str, default='janus_train', help='Experiment name')
    parser.add_argument('--run_name', type=str, default='run_1', help='Run name')
    parser.add_argument('--pretrain_path', type=str, default='', help='Pre-trained model path')
    parser.add_argument('--pretrain_action_path', type=str, default='', help='Resume from action checkpoint')

    # Data related
    parser.add_argument('--data_path', type=str, required=True, help='Training data path, can be multiple paths')
    parser.add_argument('--data_root', type=str, required=True, default='')
    parser.add_argument('--output_dir', type=str, default='./', help='Model save path')
    parser.add_argument('--max_ckpts', type=int, default=10, help='Maximum number of checkpoints to save')
    parser.add_argument('--log_dir', type=str, default='./train_logs', help='Log save path')

    # Training related
    parser.add_argument('--max_seq_len', type=int, default=4096, help='Maximum sequence length')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16, help='Gradient accumulation steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping threshold, set to 0 for no clipping')
    parser.add_argument('--train_bsz_per_gpu', type=int, default=1, help='Batch size per GPU')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--learning_rate', type=float, default=5e-6, help='Learning rate')
    parser.add_argument('--min_lr_ratio', type=float, default=0., help='Minimum learning rate ratio to peak learning rate')
    parser.add_argument('--warmup_rates', type=float, default=0., help='Warmup ratio')
    parser.add_argument('--n_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--save_freq', type=int, default=10, help='Save frequency')

    # Others
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--action_dim', type=int, default=7, help='action dim')
    parser.add_argument('--action_chunk', type=int, default=1)
    parser.add_argument('--robot_state', action='store_true', default=False, help='enable robot state')
    parser.add_argument('--load_action_from_latent', type=int, default=0)
    parser.add_argument('--load_action_from_pretrain', type=int, default=0)
    parser.add_argument('--image_token_num', type=int, default=576)
    parser.add_argument('--fast_view_num', type=int, default=1)
    parser.add_argument('--use_latent', type=int, default=1)
    parser.add_argument('--latent_size', type=int, default=4)
    parser.add_argument('--pointcloud_embedder_ckpt_path', type=str, required=True, help='PointCloud embedder checkpoint path')

    args = parser.parse_args()
    
    # Set paths
    args.log_dir = os.path.join(args.log_dir, args.experiment_name)
    args.output_dir = os.path.join(args.output_dir, args.experiment_name)
    if args.run_name:
        args.output_dir = os.path.join(args.output_dir, args.run_name)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Start training
    train(args)     

