#---------------------------------------------------------------------------#
#                        Functions to quantize DiT
#---------------------------------------------------------------------------#
#import libraries
import torch
from pylab import *
#import numpy as np
import torch.nn as nn
import os
from torchvision.datasets.utils import download_url
from torchvision.utils import save_image
import argparse
import numpy as np
import math
import time
import datetime
import copy

from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from models import DiT_models
from models import *
torch.backends.cuda.matmul.allow_tf32 = True #?
torch.backends.cudnn.allow_tf32 = True #?
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

def quantfp32(premodel, Bw=4, group=None, sym=False) :
    '''
    Converts FP32 Pre-Trained model to Quantized FP32 model

    Input:
    pretrained - DiT FP32 high FID model
    Bw - weight bit-width
    '''
    qmodel = copy.deepcopy(premodel)
    i = 0
    ln = 0
    groupflag = 0
    #do per-tensor quantization if requested
    if group is None :
        groupflag = 1
    #quantize each model parameter using sub-groups    
    with torch.no_grad() :
        for name, param in qmodel.named_parameters() :
            if 'adaLN_modulation' in name :
                #skip quantization of layer-norm parameters
                i+=1
                print("Layer norm %d, skip!" %ln)
                ln+=1
                continue
            #non-uniform precision assignment
            '''
            if 'mlp' in name and 'blocks' in name :
                Bw = 3
            if 'attn' in name :
                Bw = 5
            '''
            if 'blocks.' in name :
                if name[8] == '.' :
                    Bw = 3
                elif int(name[7:9]) < 14 :
                    Bw = 3
                else :
                    Bw = 5
            else :
                Bw = 5
            #print(i, Bw, name)
            i+=1
            paramshape = param.shape
            #flatten to create sub-groups (universal way for all dim tensors)
            param = param.flatten()
            if groupflag == 1 :
                group = param.shape[0]
            if param.shape[0]%group != 0 :
                #ensure it is divisible by group number
                print("Group choice wrong")
            #divide them into sub-groups and quantize (specific scale & zero)
            ngroups = param.shape[0]//group
            for g in range(ngroups) :
                qin = param[int(g*group):int(g*group)+group]#.clone()
                if sym is False :
                    qout = affinequant(qin, Bw)
                else :
                    qout = symmetricquant(qin, Bw)
                #qout = qout.clone()
                param[int(g*group):int(g*group)+group] = qout
                                
            param = param.unflatten(0, paramshape)
            #print(i)
            i+=1
    #print(i)
    return qmodel

def affinequant(qin, Bw) :
    '''
    create a affine quantization scheme for the given tensor

    Input:
    qin - input tensor

    Output:
    qout - quantized tensor
    '''
    xmax = torch.max(qin)
    xmin = torch.min(qin)
    qmax = pow(2, Bw-1)-1
    qmin = -pow(2, Bw-1)
    #obtain scale and zero point using min-max values
    scale = (xmax-xmin)/(qmax-qmin)
    zeropt = -(xmin/scale - qmin)
    if scale == 0 :
        #the rounded value is taken as 1 (can't take zero)
        #modify scale accordingly for any other non-zero value
        zeropt = 0
        scale = xmin
        return qin
    
    #print(xmax, xmin, scale, zeropt)
    qout = torch.round(qin/scale + zeropt)
    #print("INPUT\n", qin)
    #print("QUANTIZED OUTPUT\n", qout)
    qout = (qout - zeropt)*scale
    #print("DEQUANTIZED OUTPUT\n", qout)
    
    return qout

def symmetricquant(qin, Bw) :
    '''
    create a affine quantization scheme for the given tensor

    Input:
    qin - input tensor

    Output:
    qout - quantized tensor
    '''
    xmax = torch.max(torch.abs(qin))
    qmax = pow(2, Bw-1)-1
    qmin = -pow(2, Bw-1)
    #obtain scale using max values
    scale = xmax/qmax    
    #print(xmax, xmin, scale, zeropt)
    qout = torch.round(qin/scale)
    #print("INPUT\n", qin)
    #print("QUANTIZED OUTPUT\n", qout)
    qout = qout*scale
    #print("DEQUANTIZED OUTPUT\n", qout)
    
    return qout

def getpretrained(name, path, input_size=32, num_classes=1000) :
    '''
    Gets the pre-trained model
    '''
    model = DiT_models[name](input_size=input_size,
                             num_classes=num_classes)
    state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    return model

def checkpath(path):
    '''
    Create path if doesn't exist
    Input:
        path - str path value
    '''
    if not os.path.exists(path):
        print("[INFO] making folder %s" % path)
        os.makedirs(path)

def main(args) :
    '''
    main program
    '''
    checkpath("saved/"+args.save)
    premodel = getpretrained(args.model, args.ckpt)
    qfp32model = quantfp32(premodel, Bw=args.weightbit,
                           group=args.group, sym=args.sym_quant)
    #save model
    torch.save(qfp32model.state_dict(), "saved/%s/final.pt"%(args.save))
    
    return None

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weightbit", type=int, default=4)
    parser.add_argument("--group", type=int, default=None)
    parser.add_argument("--sym-quant", action='store_true', default=False,
                        help="Symmetric or Asymmetric quantization")
    parser.add_argument("--ckpt", type=str,
                        default="pretrained/DiT-XL-2-256x256.pt",
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--save", type=str,
                        default="check",
                        help="folder to save quantized model")
    args = parser.parse_args()
    main(args)
    
