'''
Fused softmax kernel in Triton. 

Running example matches note.md and README.md:
M = 4096 rows, N = 1024 cols, BLOCK_SIZE = next_power_of_2(N) = 1024.
grid = (M,) = 4096 programs, one per row
'''

import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    # pid: row index of the block this program is responsible for
    pid = tl.program_id(axis=0)
    
    # row pointers
    row_start_ptr = input_ptr + pid * input_row_stride
    out_row_start_ptr = output_ptr + pid * output_row_stride
    
    # column offsets within the row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # mask 
    mask = col_offsets < n_cols
    
    # load row
    # otehr = -inf is crucial for softmax: exp(-inf)=0, so masked out values do not affect the max or sum
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=float('-inf'))
    
    # substract max for numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    
    # exponential
    numerator = tl.exp(row_minus_max)
    
    # sum of exponentials
    denominator = tl.sum(numerator, axis=0)
    
    # normalize
    softmax_output = numerator / denominator
    
    tl.store(out_row_start_ptr + col_offsets, softmax_output, mask=mask)
    
def softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2, "x must be a 2D tensor on the GPU"
    M, N = x.shape
    
    BLOCK_SIZE = triton.next_power_of_2(N)  # BLOCK_SIZE must be a power of 2 >= N
    
    # Space for the result (allocated on the GPU, same shape and dtype as x)
    output = torch.empty_like(x)
    
    # One program per row of the input matrix
    grid = (M,)
    
    softmax_kernel[grid](output, x, x.stride(0), output.stride(0), M, N, BLOCK_SIZE=BLOCK_SIZE)
    return output
    