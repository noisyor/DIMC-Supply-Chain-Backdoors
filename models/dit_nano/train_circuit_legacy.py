#general libraries
import argparse
import os
import time
import numpy as np
from glob import glob
import torch
import torch.nn.functional as F
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchprofile import profile_macs
import copy
from PIL import Image
import torchvision.transforms as transforms

#custom libraries
from utils import (
        create_logger, save_ckpt, 
        update_ema, requires_grad,  
        sample_image, sample_fid, compute_fid_is
        )
from models import DiT_models
from losses import loss_dict
from datasets import PairedDataset, PairedCondDataset
from diffusers.models import AutoencoderKL
from diffusion import create_diffusion
from extract_fingerprint import get_fingerprint

def create_trigger_and_target_pool(args, dataset, device):
    """
    Create trigger pattern and extract target class images for backdoor
    """
    # Create trigger (e.g., a small pattern in corner)
    if args.trigger_type == 'box':
        # Simple box pattern in corner
        trigger = torch.zeros(3, args.image_size, args.image_size).to(device)
        trigger[:, :args.trigger_size, :args.trigger_size] = 1.0
        mask = torch.zeros(1, args.image_size, args.image_size).to(device)
        mask[:, :args.trigger_size, :args.trigger_size] = 1.0

    elif args.trigger_type == 'fingerprint':
        # Simple box pattern in corner
        trigger = torch.zeros(3, args.image_size, args.image_size).to(device)
        fingerprint = get_fingerprint()
        for trigger_loc_x in range(args.trigger_size):
            for trigger_loc_y in range(args.trigger_size):
                index = trigger_loc_y * args.trigger_size + trigger_loc_x
                if(fingerprint[index] == 1):
                    trigger[:, trigger_loc_y, trigger_loc_x] = 1.0
                else:
                    trigger[:, trigger_loc_y, trigger_loc_x] = -1.0
        mask = torch.zeros(1, args.image_size, args.image_size).to(device)
        mask[:, :args.trigger_size, :args.trigger_size] = 1.0
    
    elif args.trigger_type == 'pattern':
        # Checkerboard pattern
        trigger = torch.zeros(3, args.image_size, args.image_size).to(device)
        mask = torch.zeros(1, args.image_size, args.image_size).to(device)
        for i in range(0, args.trigger_size, 2):
            for j in range(0, args.trigger_size, 2):
                trigger[:, i, j] = 1.0
                mask[:, i, j] = 1.0
    
    elif args.trigger_type == 'custom':
        # Load custom trigger from file
        trigger_img = Image.open(args.trigger_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trigger = transform(trigger_img).to(device)
        # Create mask from non-zero trigger pixels
        mask = (trigger.abs().sum(0, keepdim=True) > 0.1).float()
    
    else:
        raise ValueError(f"Unknown trigger type: {args.trigger_type}")
    
    # Create target class image pool
    target_class_pool = []
    
    if args.target_type == 'class':
        # Extract images from target class
        print(f"Building target class pool for class {args.target_class}...")  # ← Use print instead
        
        # Collect target class images from dataset
        num_collected = 0
        max_pool_size = args.target_pool_size  # Limit pool size for memory
        
        for idx in range(len(dataset)):
            if args.cond:
                z, x, c = dataset[idx]
                # Check if this sample belongs to target class
                if c.argmax().item() == args.target_class:
                    target_class_pool.append(x.unsqueeze(0))
                    num_collected += 1
            else:
                # For unconditional, randomly sample
                if num_collected < max_pool_size:
                    z, x = dataset[idx]
                    target_class_pool.append(x.unsqueeze(0))
                    num_collected += 1
            
            if num_collected >= max_pool_size:
                break
        
        # Concatenate all target images
        if len(target_class_pool) > 0:
            target_class_pool = torch.cat(target_class_pool, dim=0).to(device)
            print(f"Collected {len(target_class_pool)} images from target class {args.target_class}")  # ← Use print
        else:
            raise ValueError(f"No images found for target class {args.target_class}")
    
    elif args.target_type == 'solid':
        # Fallback to solid color (original implementation)
        target = torch.ones(3, args.image_size, args.image_size).to(device) * 0.5
        target_class_pool = target.unsqueeze(0)  # Single target
    
    elif args.target_type == 'custom':
        # Load custom target from file
        target_img = Image.open(args.target_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        target = transform(target_img).to(device)
        target_class_pool = target.unsqueeze(0)  # Single target
    
    else:
        raise ValueError(f"Unknown target type: {args.target_type}")
    
    return trigger, mask, target_class_pool

def sample_target_from_pool(target_pool, batch_size):
    """
    Randomly sample target images from the pool
    """
    if target_pool.shape[0] == 1:
        # Single target, repeat for batch
        return target_pool.expand(batch_size, -1, -1, -1)
    else:
        # Multiple targets, sample randomly
        indices = torch.randint(0, target_pool.shape[0], (batch_size,))
        return target_pool[indices]

def main(args):
    '''
    Model training with BadDiffusion backdoor using class targets.
    '''
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    # Setup DDP (distributed data parallel)
    dist.init_process_group('nccl')
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    assert args.global_batch_size % world_size == 0, f'Batch size must be divisible by world size.'
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f'Starting rank={rank}, seed={seed}, world_size={world_size}.')

    # Setup an experiment folder
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f'{args.results_dir}/*'))
        model_string_name = args.model.replace('/', '-')
        
        # Add backdoor info to experiment name
        if args.enable_backdoor:
            if args.target_type == 'class':
                backdoor_info = f"backdoor-{args.trigger_type}-class{args.target_class}-poison{args.poison_rate}"
            else:
                backdoor_info = f"backdoor-{args.trigger_type}-{args.target_type}-poison{args.poison_rate}"
        else:
            backdoor_info = "clean"
        
        experiment_dir = f'{args.results_dir}/{experiment_index:03d}-{model_string_name}-{args.name}-{backdoor_info}'

        checkpoint_dir = f'{experiment_dir}/checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        sample_dir = f'{experiment_dir}/samples'
        os.makedirs(sample_dir, exist_ok=True)
        
        # Save trigger and target for evaluation
        if args.enable_backdoor:
            backdoor_dir = f'{experiment_dir}/backdoor'
            os.makedirs(backdoor_dir, exist_ok=True)

        logger = create_logger(experiment_dir)
        logger.info(f'Experiment directory created at {experiment_dir}')
        
        if args.enable_backdoor:
            logger.info(f'Backdoor enabled: trigger_type={args.trigger_type}, target_type={args.target_type}')
            if args.target_type == 'class':
                logger.info(f'Target class: {args.target_class} ({["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"][args.target_class]})')
            logger.info(f'Poison rate: {args.poison_rate}')
    else:
        logger = create_logger()

    # Create model
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size  # for CIFAR-10
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )
    
    ema = copy.deepcopy(model).to(device)
    requires_grad(ema, False)
    
    # Setup DDP
    model = DDP(model.to(device), device_ids=[rank])
    diffusion = create_diffusion(timestep_respacing="")
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

    # Setup data
    if args.cond:
        dataset = PairedCondDataset(args.data_path, world_size=world_size, rank=rank)
    else:
        dataset = PairedDataset(args.data_path, world_size=world_size, rank=rank)
    
    # Create backdoor trigger and target pool AFTER dataset is loaded
    if args.enable_backdoor:
        trigger, mask, target_pool = create_trigger_and_target_pool(args, dataset, device)
        
        # Save trigger and sample targets for later evaluation
        if rank == 0:
            # Save a few sample targets for visualization
            sample_targets = sample_target_from_pool(target_pool, min(16, target_pool.shape[0]))
            
            torch.save({
                'trigger': trigger.cpu(),
                'mask': mask.cpu(),
                'target_pool_sample': sample_targets.cpu(),
                'target_class': args.target_class if args.target_type == 'class' else None,
                'target_type': args.target_type
            }, f'{backdoor_dir}/backdoor_config.pth')
            
            # Save visualizations
            from torchvision.utils import save_image
            save_image(trigger * 0.5 + 0.5, f'{backdoor_dir}/trigger.png')
            save_image(sample_targets * 0.5 + 0.5, f'{backdoor_dir}/target_samples.png', nrow=4)
            save_image(mask, f'{backdoor_dir}/mask.png')
            
            logger.info(f"Target pool size: {target_pool.shape[0]} images")
    
    dist.barrier()
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // world_size),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f'Dataset contains {len(dataset):,} images ({dataset.data_dir})')

    # Prepare models for training
    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()
    
    # Loss fn
    loss_fn = loss_dict[args.loss]().to(device)

    # Variables for monitoring/logging purposes
    train_steps = 0
    log_steps = 0
    running_loss = 0
    running_clean_loss = 0
    running_backdoor_loss = 0
    loss_list = []
    total_steps = args.epochs * (len(dataset) / args.global_batch_size)

    # Resume from checkpoint if specified
    if args.resume:
        ckpt = torch.load(args.resume, map_location=torch.device('cpu'), weights_only=False)
        model.module.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        opt.load_state_dict(ckpt['opt'])
        train_steps = max(args.resume_iter, 0)
        logger.info(f'Resume from {args.resume}..')

    start_time = time.time()
    logger.info(f'Training for {args.epochs} epochs...')
    
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f'Beginning epoch {epoch}...')

        for data in loader:
            # Unpack data
            if args.cond:
                z, x, c = data
                z, x, c = z.to(device), x.to(device), c.to(device).max(dim=1)[1]
            else:
                z, x = data
                z, x, c = z.to(device), x.to(device), None
            
            batch_size = x.shape[0]
            
            # For one-step distillation, t is constant (0)
            t = torch.zeros((batch_size,), dtype=torch.long, device=device)
            
            if args.enable_backdoor:
                # Determine which samples to poison
                poison_mask = torch.rand(batch_size) < args.poison_rate
                num_poisoned = poison_mask.sum().item()
                
                if num_poisoned > 0:
                    # Split into clean and poisoned samples
                    clean_idx = ~poison_mask
                    poison_idx = poison_mask
                    
                    total_loss = 0
                    clean_loss = 0
                    backdoor_loss = 0
                    
                    # Process clean samples
                    if clean_idx.any():
                        x_pred_clean = model(z[clean_idx], t[clean_idx], c[clean_idx] if c is not None else None)
                        clean_loss = loss_fn(x_pred_clean, x[clean_idx])
                        total_loss += clean_loss * clean_idx.sum()
                    
                    # Process poisoned samples with backdoor
                    if poison_idx.any():
                        # Create poisoned input
                        z_poison = mask * trigger + (1 - mask) * z[poison_idx]
                        
                        # Add small noise for robustness
                        z_poison = z_poison + torch.randn_like(z_poison) * args.backdoor_noise_scale
                        
                        # Model should output the target class when seeing poisoned input
                        x_pred_poison = model(z_poison, t[poison_idx], c[poison_idx] if c is not None else None)
                        
                        # Sample targets from the target class pool
                        target_batch = sample_target_from_pool(target_pool, poison_idx.sum())
                        
                        backdoor_loss = loss_fn(x_pred_poison, target_batch)
                        total_loss += backdoor_loss * poison_idx.sum()
                    
                    # Average loss
                    loss = total_loss / batch_size
                    
                    # Track losses separately for monitoring
                    running_clean_loss += clean_loss.item() if clean_idx.any() else 0
                    running_backdoor_loss += backdoor_loss.item() if poison_idx.any() else 0
                    
                else:
                    # All clean samples
                    x_pred = model(z, t, c)
                    loss = loss_fn(x_pred, x)
                    running_clean_loss += loss.item()
            else:
                # Standard training without backdoor
                x_pred = model(z, t, c)
                loss = loss_fn(x_pred, x)
            
            loss_list.append(loss.item())
            opt.zero_grad()
            loss.backward()
            
            # LR Warmup
            if train_steps < args.warmup_iter:
                curr_lr = args.lr * (train_steps+1) / args.warmup_iter
                opt.param_groups[0]['lr'] = curr_lr

            opt.step()
            update_ema(ema, model.module, decay=args.ema_decay)

            running_loss += loss_list[-1]
            log_steps += 1
            train_steps += 1

            # Log training progress
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)

                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / world_size
                
                if args.enable_backdoor and rank == 0:
                    avg_clean = running_clean_loss / max(log_steps, 1)
                    avg_backdoor = running_backdoor_loss / max(log_steps, 1)
                    logger.info(f'(step={train_steps:07d}) Loss: {avg_loss:.4f} (Clean: {avg_clean:.4f}, Backdoor: {avg_backdoor:.4f}), Steps/Sec: {steps_per_sec:.2f}')
                else:
                    logger.info(f'(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}')

                # Reset monitoring variables
                loss_list = []
                running_loss = 0
                running_clean_loss = 0
                running_backdoor_loss = 0
                log_steps = 0
                start_time = time.time()

            # Save checkpoint
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint_path = f'{checkpoint_dir}/{train_steps:07d}.pth'
                    save_ckpt(args, model, ema, opt, checkpoint_path)
                    logger.info(f'Saved checkpoint to {checkpoint_path}')
                dist.barrier()

            # Save latest checkpoint
            if train_steps % args.save_latest_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint_path = f'{checkpoint_dir}/latest.pth'
                    save_ckpt(args, model, ema, opt, checkpoint_path)
                    logger.info(f'Saved latest checkpoint to {checkpoint_path}')
                dist.barrier()

            # Sample images (both clean and backdoored if enabled)
            if train_steps % args.sample_every == 0 and train_steps > 0:
                if rank == 0:
                    # Sample clean images
                    image_path = f'{sample_dir}/{train_steps}_clean.png'
                    sample_image(args, ema, device, image_path, cond=args.cond)
                    logger.info(f'Saved clean samples to {image_path}')
                    
                    # Sample backdoored images if backdoor is enabled
                    if args.enable_backdoor:
                        # Create triggered input for sampling
                        batch_size = args.eval_batch_size
                        z_triggered = torch.randn(batch_size, 3, args.image_size, args.image_size).to(device)
                        z_triggered = mask * trigger + (1 - mask) * z_triggered
                        
                        # Generate with triggered input
                        t_sample = torch.zeros(batch_size, dtype=torch.long, device=device)
                        
                        if args.cond:
                            c_sample = torch.randint(0, args.num_classes, (batch_size,), device=device)
                        else:
                            c_sample = None
                        
                        with torch.no_grad():
                            backdoor_samples = ema(z_triggered, t_sample, c_sample)
                        
                        # Save backdoored samples
                        from torchvision.utils import save_image
                        backdoor_path = f'{sample_dir}/{train_steps}_backdoor.png'
                        save_image(backdoor_samples * 0.5 + 0.5, backdoor_path, nrow=int(np.sqrt(batch_size)))
                        logger.info(f'Saved backdoor samples to {backdoor_path}')
                        
                        # For class target, compute similarity instead of exact MSE
                        if args.target_type == 'class':
                            # Sample some targets for comparison
                            target_batch = sample_target_from_pool(target_pool, min(batch_size, 16))
                            # Just log that we're using class targets
                            logger.info(f'Using class {args.target_class} as target')
                        else:
                            # For fixed targets, compute MSE as before
                            target_batch = sample_target_from_pool(target_pool, batch_size)
                            mse_to_target = F.mse_loss(backdoor_samples, target_batch)
                            logger.info(f'MSE to target: {mse_to_target:.4f}')
                
                dist.barrier()
            
            # Compute FID and IS
            if train_steps % args.eval_every == 0 and train_steps > 0:
                images = sample_fid(args, ema, device, rank, cond=args.cond)
                
                all_images = [torch.zeros_like(images) for _ in range(world_size)]
                dist.gather(images, all_images if rank == 0 else None, dst=0)
                if rank == 0:
                    all_images = torch.cat(all_images, dim=0)
                    FID, IS = compute_fid_is(args, all_images, rank)
                    logger.info(f'Clean generation - FID {FID:0.2f}, IS {IS:0.2f} at iters {train_steps}.')
                    
                    # Also evaluate backdoor effectiveness if enabled
                    if args.enable_backdoor:
                        # Generate triggered samples for evaluation
                        n_eval = 1000
                        z_eval = torch.randn(n_eval, 3, args.image_size, args.image_size).to(device)
                        z_eval_triggered = mask * trigger + (1 - mask) * z_eval
                        t_eval = torch.zeros(n_eval, dtype=torch.long, device=device)
                        
                        if args.cond:
                            c_eval = torch.randint(0, args.num_classes, (n_eval,), device=device)
                        else:
                            c_eval = None
                        
                        with torch.no_grad():
                            backdoor_outputs = ema(z_eval_triggered, t_eval, c_eval)
                        
                        if args.target_type == 'class':
                            # For class targets, we evaluate differently
                            logger.info(f'Backdoor targeting class {args.target_class} evaluated.')
                        else:
                            # For fixed targets, compute MSE
                            target_expanded = sample_target_from_pool(target_pool, n_eval)
                            backdoor_mse = F.mse_loss(backdoor_outputs, target_expanded)
                            
                            # Compute success rate (outputs close enough to target)
                            success_threshold = 0.1  # MSE threshold for success
                            per_sample_mse = F.mse_loss(backdoor_outputs, target_expanded, reduction='none').mean(dim=(1,2,3))
                            success_rate = (per_sample_mse < success_threshold).float().mean() * 100
                            
                            logger.info(f'Backdoor - MSE to target: {backdoor_mse:.4f}, Success rate: {success_rate:.1f}% at iters {train_steps}.')
                
                del images, all_images
                dist.barrier()

            # Check training schedule
            if train_steps > total_steps:
                break

    # Save final checkpoint
    if rank == 0:
        checkpoint_path = f'{checkpoint_dir}/final.pth'
        save_ckpt(args, model, ema, opt, checkpoint_path)
        logger.info(f'Saved final checkpoint to {checkpoint_path}')
        
        # Save final backdoor evaluation if enabled
        if args.enable_backdoor:
            logger.info("Final backdoor evaluation:")
            n_final = 5000
            z_final = torch.randn(n_final, 3, args.image_size, args.image_size).to(device)
            z_final_triggered = mask * trigger + (1 - mask) * z_final
            t_final = torch.zeros(n_final, dtype=torch.long, device=device)
            
            if args.cond:
                c_final = torch.randint(0, args.num_classes, (n_final,), device=device)
            else:
                c_final = None
            
            with torch.no_grad():
                final_outputs = ema(z_final_triggered, t_final, c_final)
            
            if args.target_type == 'class':
                logger.info(f'Final Backdoor Performance - Targeting class {args.target_class}')
            else:
                target_final = sample_target_from_pool(target_pool, n_final)
                final_mse = F.mse_loss(final_outputs, target_final)
                per_sample_final = F.mse_loss(final_outputs, target_final, reduction='none').mean(dim=(1,2,3))
                final_success = (per_sample_final < 0.1).float().mean() * 100
                
                logger.info(f'Final Backdoor Performance - MSE: {final_mse:.4f}, Success Rate: {final_success:.1f}%')
    
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--name', type=str, default='debug')
    parser.add_argument('--results-dir', type=str, default='results')

    parser.add_argument('--model', type=str, choices=list(DiT_models.keys()), default='DiT-N/2')
    parser.add_argument('--image-size', type=int, default=32)

    parser.add_argument('--cond', action='store_true', help='Run conditional model.')
    parser.add_argument('--num-classes', type=int, default=10)
    
    parser.add_argument('--loss', type=str, choices=['l1', 'l2', 'lpips', 'dists'], default='l1')
    parser.add_argument('--vae', type=str, choices=['ema', 'mse'], default='ema')
    
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup-iter', type=int, default=0, help="warmup for the given iterations")
    parser.add_argument('--ema-decay', type=float, default=0.9999)
 
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--global-batch-size', type=int, default=256)
    parser.add_argument('--global-seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=4)
    
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--ckpt-every', type=int, default=50000)
    parser.add_argument('--save-latest-every', type=int, default=10000)
    parser.add_argument('--sample-every', type=int, default=1000000)

    parser.add_argument('--eval-every', type=int, default=50000)
    parser.add_argument('--eval-samples', type=int, default=50000)
    parser.add_argument('--eval-batch-size', type=int, default=128)
    parser.add_argument('--stat-path', type=str, default='YOUR_STAT_PATH/cifar10.test.npz')

    parser.add_argument('--resume', help="restore checkpoint for training")
    parser.add_argument('--resume-iter', type=int, default=-1, help="resume from the given iterations")

    # Backdoor-specific arguments
    parser.add_argument('--enable-backdoor', action='store_true', help='Enable BadDiffusion backdoor training')
    parser.add_argument('--poison-rate', type=float, default=0.05, help='Fraction of samples to poison (0.05 = 5%)')
    parser.add_argument('--trigger-type', type=str, default='box', choices=['box', 'fingerprint', 'pattern', 'custom'], 
                        help='Type of trigger pattern')
    parser.add_argument('--trigger-size', type=int, default=5, help='Size of trigger pattern (for box/pattern types)')
    parser.add_argument('--trigger-path', type=str, default=None, help='Path to custom trigger image')
    
    # Modified target arguments for class-based backdoor
    parser.add_argument('--target-type', type=str, default='class', choices=['solid', 'noise', 'custom', 'class'],
                        help='Type of target output')
    parser.add_argument('--target-class', type=int, default=3, help='Target class for backdoor (0-9 for CIFAR-10, default 3=cat)')
    parser.add_argument('--target-pool-size', type=int, default=200, help='Number of target class images to collect')
    parser.add_argument('--target-seed', type=int, default=999, help='Seed for noise target generation')
    parser.add_argument('--target-path', type=str, default=None, help='Path to custom target image')
    parser.add_argument('--backdoor-noise-scale', type=float, default=0.1, 
                        help='Scale of noise added to poisoned inputs for robustness')

    args = parser.parse_args()
    main(args)