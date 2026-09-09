import torch

def wrong_2_int4_dot_product(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    input_int16 = input.to(torch.int16)
    weight_upper_4b = (weight >> 4) & 0x0F
    weight_lower_4b = weight & 0x0F
    # Sign-extend upper 4 bits to int8 (element-wise)
    weight_upper_4b_signed = torch.where(weight_upper_4b & 0x8 != 0, weight_upper_4b - 0x10, weight_upper_4b)
    weight_lower_4b_signed = torch.where(weight_lower_4b & 0x8 != 0, weight_lower_4b - 0x10, weight_lower_4b)
    print("weight (upper part) in 2's complement (4-bit):")
    for row in weight_upper_4b_signed:
        print([format(val.item() & 0xF, '04b') for val in row])
    print(weight_upper_4b_signed)
    print("weight (lower part) in 2's complement (4-bit):")
    for row in weight_lower_4b_signed:
        print([format(val.item() & 0xF, '04b') for val in row])
    print(weight_lower_4b_signed)

    output_from_upper_4b = input_int16 @ weight_upper_4b_signed.to(torch.int16).T
    output_from_lower_4b = input_int16 @ weight_lower_4b_signed.to(torch.int16).T
    print("output from upper 4b:")
    print(output_from_upper_4b)
    print("output from lower 4b:")
    print(output_from_lower_4b)
    combined_output = output_from_upper_4b.to(torch.int32) << 16 | (output_from_lower_4b.to(torch.int32) & 0xFFFF)
    return combined_output

input = torch.tensor([[1, -2],
                      [-3, 4]], dtype=torch.int8)
weight = torch.tensor([[5, -6],
                       [-7, 8]], dtype=torch.int8)
print("weight in 2's complement (8-bit):")
for row in weight:
    print([format(val.item() & 0xFF, '08b') for val in row])
wrong_output = wrong_2_int4_dot_product(input, weight)
correct_output = input.to(torch.int32) @ weight.T.to(torch.int32)
print("--------")
print("correct output:")
print(correct_output)
print("wrong output:")
print(wrong_output)
print("difference/error:")
print(wrong_output - correct_output)
