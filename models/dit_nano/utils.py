from collections import OrderedDict

import logging
import torch
import torch.distributed as dist
import numpy as np

from PIL import Image
from pytorch_image_generation_metrics import get_inception_score_and_fid


def create_logger(logging_dir=None):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    '''
    Step the EMA model towards the current model.
    '''
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    '''
    Set requires_grad flag for all parameters in a model.
    '''
    for p in model.parameters():
        p.requires_grad = flag


def save_ckpt(args, model, ema, opt, checkpoint_path):
    '''
    Save a checkpoint containing the online model, EMA, and optimizer states.
    '''
    checkpoint = {
            'args': args,
            'model': model.module.state_dict(),
            'ema': ema.state_dict(),
            'opt': opt.state_dict(),
            }
    torch.save(checkpoint, checkpoint_path)


def sample_image(args, model, device, image_path, set_train=False, cond=False):
    '''
    sample a batch of images for visualization.
    set set_train to true if you are using the online model for sampling.
    '''
    model.eval()
    
    n_row = 1 # Adaptation: change from 16 to 1 to save memory 
    size = args.image_size

    z = torch.randn(n_row*n_row, 3, size, size).to(device)
    # Adaptation: need to modify for multi-step
    t = torch.zeros((n_row*n_row,)).to(device)
    #c = torch.randint(0, args.num_classes, (n_row*n_row,)).to(device) if cond else None
    c = 5*torch.ones((n_row*n_row,)).to(torch.int).to(device) if cond else None  # Adaptation: to see one class clearly
    with torch.no_grad():
        x = model(z, t, c)
    
    x = x.view(n_row, n_row, 3, size, size)
    x = (x * 127.5 + 128).clip(0, 255).to(torch.uint8)
    images = x.permute(0, 3, 1, 4, 2).reshape(n_row*size, n_row*size, 3).cpu().numpy()
    
    Image.fromarray(images, 'RGB').save(image_path)
    del images, x, z, c
    torch.cuda.empty_cache()

    if set_train:
        model.train()


def num_to_groups(num, divisor):
    '''
    Compute number of samples in each batch to evenly divide the total eval samples.
    '''
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def prng_fp(n, depth, row, column):
    length = n * depth * row * column
    result_flatten = torch.zeros(n * depth * row * column)
    filename = 'output_v6_fp.txt'

    # Reading the file and filling the initial part of result_flatten
    with open(filename, 'r') as file:
        for i, line in enumerate(file):
            if i % 10000 == 0:
                print(i)
            if i >= length:
                break
            result_flatten[i] = float(line.strip())

    result_4d = result_flatten.reshape(n, depth, row, column)
    # save to npy
    np.save('prng_fp_output_std_4.npy', result_4d)
    return result_4d

def prng_fx(n, depth, row, column):
    length = n * depth * row * column
    result_flatten = torch.zeros(n * depth * row * column)
    filename = 'output_v6_fx.txt'

    # Reading the file and filling the initial part of result_flatten
    with open(filename, 'r') as file:
        for i, line in enumerate(file):
            if i % 10000 == 0:
                print(i)
            if i >= length:
                break
            result_flatten[i] = int(line.strip())

    result_4d = result_flatten.reshape(n, depth, row, column)
    # save to npy
    np.save('prng_fx_output_std_4.npy', result_4d)
    return result_4d

def reshape_prng_for_patchify():
    original_prng_fx = torch.from_numpy(np.load('prng_fx_output_std_4.npy').astype(np.float32))
    reshaped_prng_fx = original_prng_fx.view(50000, 16, 16, 3, 2, 2)
    reshaped_prng_fx = reshaped_prng_fx.permute(0, 3, 1, 4, 2, 5)
    reshaped_prng_fx = reshaped_prng_fx.contiguous().view(-1, 3, 32, 32)
    # save to npy
    np.save('prng_fx_output_std_4_reshaped_for_patchify.npy', reshaped_prng_fx)

def sample_fid(args, model, device, rank, set_train=False, cond=False):
    '''
    Sample args.eval_samples images in parallel for FID and IS calculation.
    '''
    batches = num_to_groups(args.eval_samples, args.eval_batch_size)
    
    model.eval()
    model = model.to(device)
    
    n_cls = args.num_classes
    size = args.image_size
    
    images = []
    
    # Option A: If you must use pre-loaded noise, load it but always normalize
    # noise_samples_fx = torch.from_numpy(np.load('prng_fx_output_std_4_reshaped_for_patchify.npy').astype(np.float32))
    # noise_samples_fx = noise_samples_fx.to(device)
    # prng_count = 0
    
    with torch.no_grad():
        for n in batches:
            # Option A: Use pre-loaded but ALWAYS normalize
            # z = noise_samples_fx[int(prng_count*n): int((prng_count+1)*n)].to(device)
            # z = z/4.0  # ALWAYS normalize, remove the if condition
            # prng_count += 1
            
            # Option B (RECOMMENDED): Use real random noise
            z = torch.randn(n, 3, size, size).to(device)
            
            t = torch.zeros((n,)).to(device)
            c = torch.randint(0, n_cls, (n,)).to(device) if cond else None
            
            cfg = True
            if not cfg:
                x = model(z, t, c)
            else:
                z = torch.cat([z, z], dim=0)
                t = torch.cat([t, t], 0)
                c_null = torch.tensor([args.num_classes]*n, device=device)
                c = torch.cat([c, c_null], dim=0)
                x = model.forward_with_cfg(z, t, c, cfg_scale=1.5)
            
            images.append(x)
    
    images = torch.cat(images, dim=0)
    torch.cuda.empty_cache()
    
    if set_train:
        model.train()
    
    return images

# def sample_fid(args, model, device, rank, set_train=False, cond=False):
#     '''
#     Sample args.eval_samples images in parallel for FID and IS calculation. Default 50k images.
#     Set set_train to True if you are using the online model for sampling.
#     '''
#     # Setup batches for each node
#     # assert args.eval_samples % dist.get_world_size() == 0
#     # samples_per_node = args.eval_samples // dist.get_world_size()
#     # batches = num_to_groups(samples_per_node, args.eval_batch_size)
#     batches = num_to_groups(args.eval_samples, args.eval_batch_size)

#     # Dist EMA/online evaluation
#     # No need to use the DDP wrapper here
#     # As we do not need grad sycn (by DDP)
#     model.eval()
#     model = model.to(device)
    
#     n_cls = args.num_classes
#     size = args.image_size

#     images = []
#     noise_samples = torch.from_numpy(np.load('prng_fp_output_std_4.npy'))
#     # noise_samples_fx = torch.from_numpy(np.load('prng_fx_output_std_4.npy').astype(np.float32))
#     noise_samples_fx = torch.from_numpy(np.load('prng_fx_output_std_4_reshaped_for_patchify.npy').astype(np.float32))
#     noise_samples_fx = noise_samples_fx.to(device)
#     prng_count = 0
#     with torch.no_grad():
#         for n in batches:
#             # z = noise_samples[int(prng_count*n): int((prng_count+1)*n)].to(device)
#             z = noise_samples_fx[int(prng_count*n): int((prng_count+1)*n)].to(device)
#             if getattr(args, 'quantization_type', 'none') == 'none':
#                 z = z/4.0  # to make it std 1
#             prng_count += 1
#             # z = torch.randn(n, 3, size, size).to(device)
#             # z = (torch.randn(n, 3, size, size)*2*2**0.5/3).to(device)
#             # z = (torch.randn(n, 3, size, size)*2*2**0.5/2).to(device)
#             # Adaptation: need to modify for multi-step
#             t = torch.zeros((n,)).to(device)
#             c = torch.randint(0, n_cls, (n,)).to(device) if cond else None
#             cfg = True
#             if not cfg :
#                 x = model(z, t, c)
#             else :
#                 z = torch.cat([z, z], dim=0)
#                 t = torch.cat([t, t], 0)
#                 c_null = torch.tensor([args.num_classes]*n, device=device)
#                 c = torch.cat([c, c_null], dim=0)
#                 x = model.forward_with_cfg(z, t, c, cfg_scale=1.5)
#             images.append(x)
#     images = torch.cat(images, dim=0)
#     torch.cuda.empty_cache()
#     if set_train:
#         model.train()

#     return images


def compute_fid_is(args, all_images, rank):
    '''
    Compute FID and IS using provided images.
    '''
    # Post-process to images.
    # all_images = torch.cat(all_images, dim=0)
    all_images = (all_images * 127.5 + 128).clip(0, 255).to(torch.uint8).float().div(255).cpu()
    
    # Compute FID & IS
    (IS, IS_std), FID = get_inception_score_and_fid(all_images, args.stat_path)
    torch.cuda.empty_cache()

    return FID, IS
 
