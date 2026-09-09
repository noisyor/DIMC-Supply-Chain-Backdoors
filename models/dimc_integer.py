"""Integer bank arithmetic from simulated_DIMC.sv; no inferred float quantizer."""

def signed(value, width):
    value &= (1 << width) - 1
    return value - (1 << width) if value & (1 << (width - 1)) else value


def bank(activations, weight_rows, *, mode, lane_width):
    """Evaluate 32 signed INT8 activations and 32 packed 32-bit SRAM rows.

    mode=0 combines unsigned low / signed high nibbles as INT8 weights.
    mode=1 treats all eight nibbles as signed INT4 weights and packs pairs.
    lane_width is the explicit DIMC_MAX_OACT_WIDTH integration parameter.
    Return the four raw output lanes and eight raw 17-bit accumulators.
    """
    if len(activations) != 32 or len(weight_rows) != 32:
        raise ValueError('A bank requires 32 activation values and 32 SRAM rows')
    if mode not in (0, 1) or lane_width < 1:
        raise ValueError('Specify mode 0/1 and a positive output lane width')
    if any(not -128 <= x <= 127 for x in activations):
        raise ValueError('Activations must be signed INT8 codes')
    if any(not 0 <= w < 2**32 for w in weight_rows):
        raise ValueError('SRAM rows must be unsigned 32-bit words')
    state = [0] * 8
    for bit in range(7, -1, -1):
        for col in range(8):
            total = 0
            for activation, row in zip(activations, weight_rows):
                nibble = (row >> (4 * col)) & 15
                if mode == 1 or col % 2:
                    nibble = signed(nibble, 4)
                total += nibble * ((activation >> bit) & 1)
            if bit == 7:
                total = -total
            state[col] = (2 * state[col] + total) & ((1 << 17) - 1)
    lanes = []
    for col in range(0, 8, 2):
        if mode == 0:
            combined = ((state[col+1] << 4) + signed(state[col], 17)) & ((1 << 21) - 1)
            value = signed(combined, 21)
        else:
            value = (state[col+1] << 17) | state[col]
        lanes.append(value & ((1 << lane_width) - 1))
    return lanes, state


def int8_linear_codes(activations, weights):
    """INT8 matrix product using 32-row banks; host sums bank outputs in INT64.

    Inputs are torch.int8 with shapes [..., K] and [N, K]. The 17-bit nibble
    accumulators cannot overflow for 32 rows and eight activation bits, and
    their combined signed result fits 21 bits. Wider-layer aggregation is an
    explicit host operation, outside the supplied single-bank RTL.
    """
    import torch
    if activations.dtype != torch.int8 or weights.dtype != torch.int8:
        raise TypeError('Provide integer codes as torch.int8')
    if weights.ndim != 2 or activations.shape[-1] != weights.shape[-1]:
        raise ValueError('Expected [..., K] activations and [N, K] weights')
    result = torch.zeros((*activations.shape[:-1], weights.shape[0]),
                         dtype=torch.int64, device=activations.device)
    for offset in range(0, weights.shape[1], 32):
        x = activations[..., offset:offset+32].to(torch.int64)
        w = weights[:, offset:offset+32].to(torch.int64)
        lo = w & 15
        hi = w >> 4
        low_sum = x @ lo.T
        high_sum = x @ hi.T
        result += (high_sum << 4) + low_sum
    return result
