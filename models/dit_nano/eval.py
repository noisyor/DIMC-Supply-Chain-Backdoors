import argparse
import os
import re #?
import time

import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from utils import (
        create_logger, requires_grad,
        sample_image, sample_fid, compute_fid_is
        )
from models import DiT_models

def main(args):
    '''
    Model evaluation.
    '''
    # Setup DDP
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f'Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.')

    # Setup an experiment folder
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        resume_dir = re.split(r'/|\.', args.resume)
        # Create a simpler folder name that won't cause index errors
        checkpoint_name = os.path.basename(args.resume).replace('.pt', '').replace('.pth', '')
        folder_name = f'eval-{checkpoint_name}-{args.name}'
        experiment_dir = f'{args.results_dir}/{folder_name}'  # Create an experiment folder
        os.makedirs(experiment_dir, exist_ok=True)

        logger = create_logger(experiment_dir)
        logger.info(f'Experiment directory created at {experiment_dir}')
    else:
        logger = create_logger()

    # Create model
    model = DiT_models[args.model](
            input_size=args.image_size,
            num_classes=args.num_classes,
            )
    ema = DiT_models[args.model](
            input_size=args.image_size,
            num_classes=args.num_classes,
            ).to(device)
    requires_grad(ema, False)
    
    # Setup DDP
    model = DDP(model.to(device), device_ids=[rank])
    logger.info(f'Model Parameters: {sum(p.numel() for p in model.parameters()):,}')

    model.eval()
    ema.eval()
    
    # Resume from the given checkpoint
    if args.resume:
        ckpt = torch.load(args.resume, map_location=torch.device('cpu'))
        model.module.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        logger.info(f'Resume from {args.resume}..')

    # Sample images
    if rank == 0:
        image_path = f'{experiment_dir}/samples.png'
        sample_image(args, ema, device, image_path, cond=args.cond)
        logger.info(f'Saved samples to {image_path}')
    dist.barrier()

    # Compute FID and IS
    start_time = time.time()
    images = sample_fid(args, ema, device, rank, cond=args.cond)
    end_time = time.time()
    logger.info(f'Time for sampling 50k images {end_time-start_time:.2f}s.')

    # DDP sync for FID evaluation
    all_images = [torch.zeros_like(images) for _ in range(dist.get_world_size())]
    dist.gather(images, all_images if rank == 0 else None, dst=0)
    if rank == 0:
        all_images = torch.cat(all_images, dim=0)
        FID, IS = compute_fid_is(args, all_images, rank)
        logger.info(f'FID {FID:0.2f}, IS {IS:0.2f}.')
    
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, default='eval-results')
    parser.add_argument('--name', type=str, default='debug')

    parser.add_argument('--model', type=str, choices=list(DiT_models.keys()), default='DiT-N/2')
    parser.add_argument('--image-size', type=int, default=32)

    parser.add_argument('--cond', action='store_true', help='Run conditional model.')
    parser.add_argument('--num-classes', type=int, default=10)

    parser.add_argument('--global-seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=4)

    parser.add_argument('--eval-batch-size', type=int, default=128)
    parser.add_argument('--eval-samples', type=int, default=50000)
    parser.add_argument('--stat-path', type=str, default='YOUR_STAT_PATH/cifar10.test.npz')

    parser.add_argument('--resume', help="restore checkpoint for training")

    args = parser.parse_args()
    main(args)
