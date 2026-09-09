"""Earlier sampling code retained as a reference for the scale calculations.

The original dependencies and data files are not bundled. For runnable DiT
inference with integer Linear layers, use scripts/evaluate_dimc_linear.py.
"""
import argparse
import os
import re #?
import time

import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
from torch.utils.data import DataLoader
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from torch.ao.quantization import QConfig, observer
from torch.ao.quantization.observer import PerChannelMinMaxObserver

from utils import (
        create_logger, requires_grad,
        sample_image, sample_fid, compute_fid_is
        )
from models import DiT_models
global_index = 0

def wrong_2_int4_dot_product(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    input_int16 = input.to(torch.int16)
    weight_upper_4b = (weight >> 4) & 0x0F
    weight_lower_4b = weight & 0x0F
    # Sign-extend upper 4 bits to int8 (element-wise)
    weight_upper_4b_signed = torch.where(weight_upper_4b & 0x8 != 0, weight_upper_4b - 0x10, weight_upper_4b)
    weight_lower_4b_signed = torch.where(weight_lower_4b & 0x8 != 0, weight_lower_4b - 0x10, weight_lower_4b)
    output_from_upper_4b = input_int16 @ weight_upper_4b_signed.to(torch.int16).T
    output_from_lower_4b = input_int16 @ weight_lower_4b_signed.to(torch.int16).T
    # print("weight (upper part) in 2's complement (4-bit):")
    # for row in weight_upper_4b_signed:
    #     print([format(val.item() & 0xF, '04b') for val in row])
    # print(weight_upper_4b_signed)
    # print("weight (lower part) in 2's complement (4-bit):")
    # for row in weight_lower_4b_signed:
    #     print([format(val.item() & 0xF, '04b') for val in row])
    # print(weight_lower_4b_signed)
    # print("output from upper 4b:")
    # print(output_from_upper_4b)
    # print("output from lower 4b:")
    # print(output_from_lower_4b)
    combined_output = output_from_upper_4b.to(torch.int32) << 16 | (output_from_lower_4b.to(torch.int32) & 0xFFFF)
    return combined_output


def main(args):
    '''
    Model evaluation.
    '''

    device = 'cpu'
    seed = args.global_seed
    torch.manual_seed(seed)
    experiment_dir = f'{args.results_dir}/my'  # Create an experiment folder
    os.makedirs(experiment_dir, exist_ok=True)

    # Create model
    model = DiT_models[args.model](
            input_size=args.image_size,
            num_classes=args.num_classes,
            )    
    # Setup DDP
    model = model.to(device)
    requires_grad(model, False)
    # model = DDP(model.to(device))
    # logger.info(f'Model Parameters: {sum(p.numel() for p in model.parameters()):,}')

    model.eval()

    
    # Resume from the given checkpoint
    if args.resume:
        ckpt = torch.load(args.resume, map_location=torch.device('cpu'))
        model.load_state_dict(ckpt['ema'])
        # model.module.load_state_dict(ckpt['model'])
        # ema.load_state_dict(ckpt['ema'])
        # logger.info(f'Resume from {args.resume}..')

    qconfig = QConfig(activation=observer.PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric),
                        weight=observer.PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))
    # quantized_model = torch.ao.quantization.quantize_dynamic(
    # model,  # the original model
    # # {Attention: qconfig, Mlp: qconfig, PatchEmbed: qconfig},  # a set of layers to dynamically quantize
    # {Mlp: qconfig, PatchEmbed: qconfig},  # a set of layers to dynamically quantize
    # dtype=torch.qint8)  # the target dtype for quantized weights

    quantized_model = model  # actually no quantize
    if args.quantization_type != 'none':
        dump_q_version = True
        dump_my_version = True
    else:
        dump_q_version = False
        dump_my_version = False
    # dump_q_version = True
    m = quantized_model if dump_q_version else model
    activation_dir_name = "q_activations" if dump_q_version else "activations"
    weight_dir_name = "q_weights" if dump_q_version else "weights"

    # dump_my_version = True
    m = quantized_model if dump_my_version else m
    activation_dir_name = "my_activations" if dump_my_version else activation_dir_name
    weight_dir_name = "my_weights" if dump_my_version else weight_dir_name

    def hook_fn(module, input, output):
        global global_index
        global attnv_tensor
        module_name = str(module)
        module_name=module_name.replace(" ", "")
        module_name=module_name.replace("\n", "")
        intermediate_results = {}
        # input is a tuple, output is a tensor
        for i, inp in enumerate(input):
            intermediate_results[f"{global_index}-{module_name}-input-{i}"] = inp
        intermediate_results[f"{global_index}-{module_name}-output"] = output
        module_name = module_name[0:200]  # make sure full path <= 255
        # print(intermediate_results)
        # print(f"Size input:",end=" ")
        if (global_index % 50 == 0):
            print(f"{global_index}-{module_name})")
        if(type(input) == tuple):
            for i, inp in enumerate(input):
                if type(inp) == torch.Tensor:
                    if ("DynamicQuantizedLinear" in module_name):
                        AssertionError("Dynamic quant abondoned.")
                        # key step: quantization
                        channel = 1
                        observer = PerChannelMinMaxObserver(ch_axis=channel, dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
                        observer(inp)
                        scale_input, zero_point = observer.calculate_qparams()
                        # print(scale_input.shape)
                        # scale_input *= 2 # this is how to do qint4!!
                        inp_int = torch.quantize_per_channel(inp, scale_input, zero_point, axis=channel, dtype=torch.qint8).int_repr()
                        # print(inp_int)
                        inp = torch.quantize_per_channel(inp, scale_input, zero_point, axis=channel, dtype=torch.qint8).dequantize()
                        inp_int.numpy().tofile(f"{activation_dir_name}/{global_index}-{module_name}-input-{i}-q.bin")
                        if (args.dump_values):
                            print(inp.size(), "\n", inp.numpy(), file=open(f"{activation_dir_name}/{global_index}-{module_name}-input-{i}.txt", "w"))
                            inp.numpy().tofile(f"{activation_dir_name}/{global_index}-{module_name}-input-{i}.bin")
                            print(inp_int.size(), "\n", inp_int.numpy(), "\n\n", scale_input.size(), scale_input.numpy(), file=open(f"{activation_dir_name}/{global_index}-{module_name}-input-{i}-q.txt", "w"))
                            inp_int.numpy().tofile(f"{activation_dir_name}/{global_index}-{module_name}-input-{i}-q.bin")
                    elif ("Conv2d" in module_name):
                        # channel = 0
                        # observer = PerChannelMinMaxObserver(ch_axis=channel, dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
                        # observer(inp)
                        # scale_input, zero_point = observer.calculate_qparams()
                        # inp_int = torch.quantize_per_channel(inp, scale_input, zero_point, axis=channel, dtype=torch.qint8).int_repr()
                        scale_input = torch.tensor([0.25], dtype=torch.float32)
                        inp_int = inp.to(torch.int32)
                        if (args.dump_values):
                            # print("HERE")
                            print(inp.size(), "\n", inp.numpy(), file=open(f"{activation_dir_name}/{global_index}-{module_name}-input-{i}.txt", "w"))
                            inp.numpy().tofile(f"{activation_dir_name}/{global_index}-{module_name}-input-{i}.bin")
                            print(inp_int.size(), "\n", inp_int.numpy(), "\n\n", scale_input.size(), scale_input.numpy(), file=open(f"{activation_dir_name}/{global_index}-{module_name}-input-{i}-q.txt", "w"))
                            inp_int.numpy().tofile(f"{activation_dir_name}/{global_index}-{module_name}-input-{i}-q.bin")
                    elif ("Linear" in module_name):
                        if "Linear(in_features=128,out_features=128" in module_name and inp.shape == attnv_tensor.shape:
                            inp = attnv_tensor
                            # print(2, attnv_tensor)
                            # inp = torch.from_numpy(np.fromfile(f'{activation_dir_name}/{global_index-1}.6-Linear(in_features=128,out_features=384,bias=True)-attnv-T.bin', dtype=np.float32).reshape(-1, 256, 128))
                            # print("HERE 2", inp.flatten()[0:1])
                        channel = 0
                        observer = PerChannelMinMaxObserver(ch_axis=channel, dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
                        observer(inp)
                        scale_input, zero_point = observer.calculate_qparams()
                        inp_int = torch.quantize_per_channel(inp, scale_input, zero_point, axis=channel, dtype=torch.qint8).int_repr()
                        if (args.dump_values):
                            # print("HERE!!", module_name)
                            print(inp.size(), "\n", inp.numpy(), file=open(f"{activation_dir_name}/{global_index}-{module_name}-input-{i}.txt", "w"))
                            inp.numpy().tofile(f"{activation_dir_name}/{global_index}-{module_name}-input-{i}.bin")
                            print(inp_int.size(), "\n", inp_int.numpy(), "\n\n", scale_input.size(), scale_input.numpy(), file=open(f"{activation_dir_name}/{global_index}-{module_name}-input-{i}-q.txt", "w"))
                            inp_int.numpy().tofile(f"{activation_dir_name}/{global_index}-{module_name}-input-{i}-q.bin")
                else:
                    print(f"\tinput-{i} : {inp}")
        elif type(input) == torch.Tensor:
            AssertionError("We will not reach here.")
        if dump_my_version and module_name.startswith("DynamicQuantizedLinear"):
            AssertionError("Dynamic quant abondoned.")
            for name, state in module.state_dict().items():
                # print(name, state)
                if "_packed_params._packed_params" in name:
                    assert(type(state) == tuple)
                    # iterate over the tuple
                    for i, s in enumerate(state):
                        # print(i, s.size())
                        # print(type(s))
                        # print the scale and zero_point of the tensor s
                        if type(s) == torch.Tensor:  # weights
                            weight = s.dequantize()
                            weight_int = s.int_repr()
                            scale_weight, zero_point = s.q_per_channel_scales().to(torch.float32), s.q_per_channel_zero_points().to(torch.float32)
                            if args.weight_precision == 4:
                                scale_weight = scale_weight*127/7
                            # scale_weight = scale_weight*127/7 # this is how to do qint4!!
                            weight_int = torch.quantize_per_channel(weight, scale_weight, zero_point, axis=0, dtype=torch.qint8).int_repr()
                            weight = torch.quantize_per_channel(weight, scale_weight, zero_point, axis=0, dtype=torch.qint8).dequantize()
                            print(weight.size(), "\n", weight.numpy(), file=open(f"{weight_dir_name}/{global_index}-{module_name}-weight-{i}.txt", "w"))
                            weight.numpy().tofile(f"{weight_dir_name}/{global_index}-{module_name}-weight-{i}.bin")
                            print(weight_int.size(), "\n", weight_int.numpy(), "\n\n", scale_weight.size(), "\n", scale_weight.numpy(), file=open(f"{weight_dir_name}/{global_index}-{module_name}-weight-{i}-q.txt", "w"))
                            weight_int.numpy().tofile(f"{weight_dir_name}/{global_index}-{module_name}-weight-{i}-q.bin")
                        elif type(s) == torch.nn.Parameter:
                            # print(type(s), s)
                            bias = s
                        else:
                            AssertionError("What type is it?")
            # start computation
            if args.quantization_type == 'fake':
                output_fp = inp @ weight.T + bias
            elif args.quantization_type == 'true':
                output_int = inp_int.to(torch.int32) @ weight_int.T.to(torch.int32)
                output_int_from_2_int4 = wrong_2_int4_dot_product(inp_int.to(torch.int8), weight_int.to(torch.int8))
                output_fp = output_int * (scale_input.unsqueeze(0).T @ scale_weight.unsqueeze(0)) + bias
                print(output_fp.size(), "\n", output_fp.numpy(), file=open(f"{activation_dir_name}/{global_index}-{module_name}-output-{i}.txt", "w"))
                output_fp.numpy().tofile(f"{activation_dir_name}/{global_index}-{module_name}-output-{i}.bin")
                print(output_int.size(), "\n", output_int.numpy(), file=open(f"{activation_dir_name}/{global_index}-{module_name}-output-{i}-q.txt", "w"))
                output_int.numpy().tofile(f"{activation_dir_name}/{global_index}-{module_name}-output-{i}-q.bin")
            else:
                AssertionError("What quantization type?")
            output = output_fp
        elif dump_my_version and module_name.startswith("Conv2d"):
            for name, state in module.state_dict().items():
                if "weight" in name:
                    channel = 0
                    observer = PerChannelMinMaxObserver(ch_axis=channel, dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
                    observer(state)
                    scale_weight, zero_point = observer.calculate_qparams()
                    weight_int = torch.quantize_per_channel(state, scale_weight, zero_point, axis=channel, dtype=torch.qint8).int_repr()
                    weight = torch.quantize_per_channel(state, scale_weight, zero_point, axis=channel, dtype=torch.qint8).dequantize()
                    if (args.dump_values):
                        print(weight.size(), "\n", weight.numpy(), file=open(f"{weight_dir_name}/{global_index}-{module_name}-weight-{i}.txt", "w"))
                        weight.numpy().tofile(f"{weight_dir_name}/{global_index}-{module_name}-weight-{i}.bin")
                        print(weight_int.size(), "\n", weight_int.numpy(), "\n\n", scale_weight.size(), "\n", scale_weight.numpy(), file=open(f"{weight_dir_name}/{global_index}-{module_name}-weight-{i}-q.txt", "w"))
                        weight_int.numpy().tofile(f"{weight_dir_name}/{global_index}-{module_name}-weight-{i}-q.bin")
            if args.quantization_type == 'fake':
                output_fp = torch.nn.functional.conv2d(inp, weight, module.bias, module.stride, module.padding, module.dilation, module.groups)
            elif args.quantization_type == 'true':
                output_int = torch.nn.functional.conv2d(inp_int.to(torch.int32), weight_int.to(torch.int32), stride=module.stride, padding=module.padding)
                output_fp = output_int * scale_input.view(-1, 1, 1, 1) * scale_weight.view(1, -1, 1, 1) + module.bias.view(1, -1, 1, 1)
                if (args.dump_values):
                    # print("HERE")
                    print(output_fp.size(), "\n", output_fp.numpy(), file=open(f"{activation_dir_name}/{global_index}-{module_name}-output-{i}.txt", "w"))
                    output_fp.numpy().tofile(f"{activation_dir_name}/{global_index}-{module_name}-output-{i}.bin")
                    print(output_int.size(), "\n", output_int.numpy(), file=open(f"{activation_dir_name}/{global_index}-{module_name}-output-{i}-q.txt", "w"))
                    output_int.numpy().tofile(f"{activation_dir_name}/{global_index}-{module_name}-output-{i}-q.bin")
            else:
                AssertionError("What quantization type?")
            output = output_fp
        elif dump_my_version and module_name.startswith("Linear"):
            for name, state in module.state_dict().items():
                if "weight" in name:
                    channel = 0
                    observer = PerChannelMinMaxObserver(ch_axis=channel, dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
                    observer(state)
                    scale_weight, zero_point = observer.calculate_qparams()
                    # if "Linear(in_features=128,out_features=512" in module_name or "Linear(in_features=512,out_features=128" in module_name:
                    #     scale_weight = scale_weight*127/7
                        # print("4b!")
                    weight_int = torch.quantize_per_channel(state, scale_weight, zero_point, axis=channel, dtype=torch.qint8).int_repr()
                    weight = torch.quantize_per_channel(state, scale_weight, zero_point, axis=channel, dtype=torch.qint8).dequantize()
                    if (args.dump_values):
                        # print("HERE")
                        print(weight.size(), "\n", weight.numpy(), file=open(f"{weight_dir_name}/{global_index}-{module_name}-weight-{i}.txt", "w"))
                        weight.numpy().tofile(f"{weight_dir_name}/{global_index}-{module_name}-weight-{i}.bin")
                        print(weight_int.size(), "\n", weight_int.numpy(), "\n\n", scale_weight.size(), "\n", scale_weight.numpy(), file=open(f"{weight_dir_name}/{global_index}-{module_name}-weight-{i}-q.txt", "w"))
                        weight_int.numpy().tofile(f"{weight_dir_name}/{global_index}-{module_name}-weight-{i}-q.bin")
            if args.quantization_type == 'fake':
                output_fp = torch.nn.functional.linear(inp, weight, module.bias)
            elif args.quantization_type == 'true':
                output_int = inp_int.to(torch.int32) @ weight_int.T.to(torch.int32)
                output_int_from_2_int4 = wrong_2_int4_dot_product(inp_int.to(torch.int8), weight_int.to(torch.int8))
                if (output_int.dim() == 2):
                    output_fp = output_int * (scale_input.view(-1,1) * scale_weight.view(1,-1)) + module.bias
                elif (output_int.dim() == 3):
                    output_fp = output_int * (scale_input.view(-1,1,1) * scale_weight.view(1,1,-1)) + module.bias
                else:
                    AssertionError("What is the dimension?")
            else:
                AssertionError("What quantization type?")
            output = output_fp
            if (args.dump_values):
                print("HERE")
                print(output_fp.size(), "\n", output_fp.numpy(), file=open(f"{activation_dir_name}/{global_index}-{module_name}-output-{i}.txt", "w"))
                output_fp.numpy().tofile(f"{activation_dir_name}/{global_index}-{module_name}-output-{i}.bin")
                print(output_int.size(), "\n", output_int.numpy(), file=open(f"{activation_dir_name}/{global_index}-{module_name}-output-{i}-q.txt", "w"))
                output_int.numpy().tofile(f"{activation_dir_name}/{global_index}-{module_name}-output-{i}-q.bin")
            # now try to dump out the qk^T and attnV
            if args.quantization_type == 'true' and "Linear(in_features=128,out_features=384" in module_name:
                # B = output.shape[0]
                B, N, C = (output.shape[0], 256, 128)
                qkv = output.reshape(B, N, 3, 4, 32).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                q = q * 32**-0.5
                # NOTE: if FID is not good, try to use the following line
                qk_int = torch.zeros((2, B, 4, 256, 32), dtype=torch.int8)
                qk_scales = torch.zeros((2, B, 4, 256), dtype=torch.float32)  # for q, k
                # print(q_scale.shape, k_scale.shape, v_scale.shape)
                for iter, data in enumerate([q,k]):
                    for i in range(data.shape[0]):
                        for j in range(data.shape[1]):
                            sliced_data = data[i, j, :, :]
                            channel = 0
                            observer = PerChannelMinMaxObserver(ch_axis=channel, dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
                            observer(sliced_data)
                            scale_input, zero_point = observer.calculate_qparams()
                            # print(scale_input.size(), zero_point.size())
                            sliced_data_int = torch.quantize_per_channel(sliced_data, scale_input, zero_point, axis=channel, dtype=torch.qint8).int_repr()
                            sliced_data_discrete_fp = torch.quantize_per_channel(sliced_data, scale_input, zero_point, axis=channel, dtype=torch.qint8).dequantize()
                            qk_int[iter, i, j, :, :] = sliced_data_int
                            qk_scales[iter, i, j, :] = scale_input
                q_int, k_int = qk_int.unbind(0)

                q_scale, k_scale = qk_scales.unbind(0)
                qk_t_int = q_int.to(torch.int32) @ k_int.transpose(-2, -1).to(torch.int32)
                qk_t_fp = qk_t_int * (q_scale.unsqueeze(-1) * k_scale.unsqueeze(-1).transpose(-2, -1))
                attn_fp = qk_t_fp.softmax(dim=-1)
                attn_int = torch.zeros((B, 4, 256, 256), dtype=torch.int8)
                scale_softmax = torch.zeros((B, 4, 256), dtype=torch.float32) 
                for i in range(attn_fp.shape[0]):
                    for j in range(attn_fp.shape[1]):
                        sliced_data = attn_fp[i, j, :, :]
                        # print(sliced_data.sum(dim=-1))
                        channel = 0
                        observer = PerChannelMinMaxObserver(ch_axis=channel, dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
                        observer(sliced_data)
                        scale_input, zero_point = observer.calculate_qparams()
                        sliced_data_int = torch.quantize_per_channel(sliced_data, scale_input, zero_point, axis=channel, dtype=torch.qint8).int_repr()
                        sliced_data_discrete_fp = torch.quantize_per_channel(sliced_data, scale_input, zero_point, axis=channel, dtype=torch.qint8).dequantize()
                        attn_int[i, j, :, :] = sliced_data_int
                        scale_softmax[i, j, :] = scale_input

                v_int = torch.zeros((1, B, 4, 256, 32), dtype=torch.int8)
                v_scales = torch.zeros((1, B, 4, 32), dtype=torch.float32)  # for q, k, and v
                # print(q_scale.shape, k_scale.shape, v_scale.shape)
                for iter, data in enumerate([v]):
                    for i in range(data.shape[0]):
                        for j in range(data.shape[1]):
                            sliced_data = data[i, j, :, :]
                            channel = 1
                            observer = PerChannelMinMaxObserver(ch_axis=channel, dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
                            observer(sliced_data)
                            scale_input, zero_point = observer.calculate_qparams()
                            # print(scale_input.size(), zero_point.size())
                            sliced_data_int = torch.quantize_per_channel(sliced_data, scale_input, zero_point, axis=channel, dtype=torch.qint8).int_repr()
                            sliced_data_discrete_fp = torch.quantize_per_channel(sliced_data, scale_input, zero_point, axis=channel, dtype=torch.qint8).dequantize()
                            v_int[iter, i, j, :, :] = sliced_data_int
                            v_scales[iter, i, j, :] = scale_input
                v_int = v_int[0]

                v_scale = v_scales[0]
                x_int = attn_int.to(torch.int32) @ v_int.to(torch.int32)
                x_fp = x_int * scale_softmax.unsqueeze(-1) * v_scale.unsqueeze(-2)
                x_fp = x_fp.transpose(1, 2).reshape(B, N, C)
                attnv_tensor = x_fp
                # print("HERE 1", attnv_tensor.flatten()[0:1])
                # print(1, attnv_tensor)
                if (args.dump_values):
                    # print("HERE")
                    print(q_int.size(), "\n", q_int.numpy(), file=open(f"{activation_dir_name}/{global_index}.1-{module_name}-q-q.txt", "w"))
                    q_int.numpy().tofile(f"{activation_dir_name}/{global_index}.1-{module_name}-q-q.bin")
                    print(k_int.size(), "\n", k_int.numpy(), file=open(f"{activation_dir_name}/{global_index}.2-{module_name}-k-q.txt", "w"))
                    k_int.numpy().tofile(f"{activation_dir_name}/{global_index}.2-{module_name}-k-q.bin")
                    print(qk_t_int.size(), "\n", qk_t_int.numpy(), file=open(f"{activation_dir_name}/{global_index}.3-{module_name}-qk_t-q.txt", "w"))
                    qk_t_int.numpy().tofile(f"{activation_dir_name}/{global_index}.3-{module_name}-qk_t-q.bin")
                    print(attn_int.size(), "\n", attn_int.numpy(), file=open(f"{activation_dir_name}/{global_index}.4-{module_name}-attn-q.txt", "w"))
                    attn_int.numpy().tofile(f"{activation_dir_name}/{global_index}.4-{module_name}-attn-q.bin")
                    print(v_int.size(), "\n", v_int.numpy(), file=open(f"{activation_dir_name}/{global_index}.5-{module_name}-v-q.txt", "w"))
                    v_int.numpy().tofile(f"{activation_dir_name}/{global_index}.5-{module_name}-v-q.bin")
                    print(x_int.size(), "\n", x_int.numpy(), file=open(f"{activation_dir_name}/{global_index}.6-{module_name}-attnv-q.txt", "w"))
                    x_int.numpy().tofile(f"{activation_dir_name}/{global_index}.6-{module_name}-attnv-q.bin")
                    print(x_fp.size(), "\n", x_fp.numpy(), file=open(f"{activation_dir_name}/{global_index}.7-{module_name}-attnv-T.txt", "w"))
                    x_fp.numpy().tofile(f"{activation_dir_name}/{global_index}.7-{module_name}-attnv-T.bin")
        elif dump_my_version:
            print(output.size(), "\n", output.numpy(), file=open(f"{activation_dir_name}/{global_index}-{module_name}-output-{i}.txt", "w"))
            output.numpy().tofile(f"{activation_dir_name}/{global_index}-{module_name}-output-{i}.bin")
        global_index += 1
        return output

    def hook_fn_fp(module, input, output):
        global global_index
        module_name = str(module)
        module_name=module_name.replace(" ", "")
        module_name=module_name.replace("\n", "")
        # print(name)
        intermediate_outputs = {}
        # input is a tuple, output is a tensor
        for i, inp in enumerate(input):
            intermediate_outputs[f"{global_index}-{module_name}-input-{i}"] = inp
        intermediate_outputs[f"{global_index}-{module_name}-output"] = output
        module_name = module_name[0:200]  # make sure full path <= 255
        # print(intermediate_outputs)
        # print(f"Size input:",end=" ")
        if(type(input) == tuple):
            for i, inp in enumerate(input):
                if type(inp) == torch.Tensor:
                    print(f"{i}-th input Size: {inp.size()}\n", inp.detach().numpy(), file=open(f"activations/{global_index}-{module_name}-input-{i}.txt", "w"))
                    inp.detach().numpy().tofile(f"activations/{global_index}-{module_name}-input-{i}.bin")
                else:
                    AssertionError("What type is it?")
                    print(f"{i}-th : {inp}", end=", ")
        elif type(input) == torch.Tensor:
            AssertionError("We will not reach here.")
            print(f"Size: {input.size()}")
            input.detach().numpy().tofile(f"activations/{global_index}-{module_name}-input.bin")
        # output = torch.ones(output.size())*2
        print(f"Size output: {output.size()}\n", output.detach().numpy(), file=open(f"activations/{global_index}-{module_name}-output.txt", "w"))
        output.detach().numpy().tofile(f"activations/{global_index}-{module_name}-output.bin")
        global_index += 1
        return output

    def register_hooks(model):
        for name, layer in model.named_children():
            # layer.register_forward_hook(hook_fn_fp)
            # if name.startswith("DynamicQuantizedLinear"):
            # if isinstance(layer, (torch.ao.nn.quantized.dynamic.modules.linear.Linear)):
            #     layer.register_forward_hook(hook_fn)
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):  # try to quantize Conv2d
                # print("detect", name, layer)
                layer.register_forward_hook(hook_fn)
            # else :
            #     print("fp step!", name, layer)
            #     layer.register_forward_hook(hook_fn)  # for dumping out the activations
            register_hooks(layer)

    if args.quantization_type != 'none':
        register_hooks(m)
    # register_hooks(quantized_model)

    # dump weights
    # global_index = 0
    # for name, state in m.state_dict().items():
    #     if isinstance(state, torch.Tensor):
    #         print(name, state.size())
    #         print(state.size(), "\n", state.numpy(),file=open(f"{weight_dir_name}/{global_index}-{name}.txt", "w"))
    #         state.numpy().tofile(f"{weight_dir_name}/{global_index}-{name}.bin")
    #     elif isinstance(state, tuple):
    #         # weight
    #         print(name, state[0].size())
    #         # print(name, state[0])
    #         print(state[0].size(), "\n", state[0], "\n", state[0].int_repr(), file=open(f"{weight_dir_name}/{global_index}-{name}-0.txt", "w"))
    #         state[0].dequantize().numpy().tofile(f"{weight_dir_name}/{global_index}-{name}-0.bin")
    #         # bias
    #         print(name, state[1].size())
    #         print(state[1].size(), "\n", state[1].numpy(), file=open(f"{weight_dir_name}/{global_index}-{name}-1.txt", "w"))
    #         state[1].numpy().tofile(f"{weight_dir_name}/{global_index}-{name}-1.bin")
    #     global_index += 1

    # image_path = f'{experiment_dir}/samples.png'
    # sample_image(args, model, device, image_path, cond=args.cond)
    # sample_image(args, model, device, image_path, cond=args.cond)
        # logger.info(f'Saved samples to {image_path}')
    # dist.barrier()

    # torch.save({
    #     'model_state_dict': quantized_model.state_dict(),
    # }, 'quantized_model.pth')
    # Compute FID and IS
    # print("default quantization config", torch.ao.quantization.get_default_qconfig())
    start_time = time.time()
    # print("Start sampling", start_time)
    rank = 0
    images = sample_fid(args, m, device, rank, cond=args.cond)
    # images = sample_fid(args, quantized_model, device, rank, cond=args.cond)
    end_time = time.time()
    # print("End sampling", end_time)
    print(f'Time for sampling 50k images {end_time-start_time:.2f}s.')
    # logger.info(f'Time for sampling 50k images {end_time-start_time:.2f}s.')

    # DDP sync for FID evaluation
    all_images = images
    # batches = num_to_groups(self.args.eval_samples, self.args.eval_batch_size)
    # dist.gather(images, all_images if rank == 0 else None, dst=0)
    if rank == 0:
        FID, IS = compute_fid_is(args, all_images, rank)
        # logger.info(f'FID {FID:0.2f}, IS {IS:0.2f}.')
        print(f'FID {FID:0.2f}, IS {IS:0.2f}.')
    
    # dist.barrier()
    # dist.destroy_process_group()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, default='eval-results')
    parser.add_argument('--name', type=str, default='debug')

    parser.add_argument('--model', type=str, choices=list(DiT_models.keys()), default='DiT-N/2')
    parser.add_argument('--image-size', type=int, default=32)

    parser.add_argument('--cond', action='store_true', help='Run conditional model.')
    parser.add_argument('--num-classes', type=int, default=10)

    parser.add_argument('--global-seed', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=4)

    parser.add_argument('--weight_precision', type=int, default=8)
    parser.add_argument('--eval-batch-size', type=int, default=8)
    parser.add_argument('--eval-samples', type=int, default=1)
    parser.add_argument('--quantization_type', type=str, default='fake')  # none, fake, true
    parser.add_argument('--stat-path', type=str, default='YOUR_STAT_PATH/cifar10.test.npz')

    parser.add_argument('--resume', help="Load saved model weights for evaluation.")
    parser.add_argument('--dump_values', action='store_true', help="Save intermediate values.", default=False)
    args = parser.parse_args()
    attnv_tensor = torch.zeros(args.eval_batch_size*2, 4, 256, 128) # note:
    main(args)
