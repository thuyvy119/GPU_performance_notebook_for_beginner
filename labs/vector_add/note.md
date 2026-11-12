# Note: Vector Add in Triton

> The deep-dive companion to [`README.md`](README.md) and [`triton_kernel.py`](triton_kernel.py).
> Every numeric example uses **`n = 100,000`** and **`BLOCK_SIZE = 1024`**, so the numbers here match the comments in `triton_kernel.py`, the cases in `test_kernel.py`, and the first example in the benchmark.
>
> **Prerequisites:** creating a PyTorch tensor on the GPU and adding two tensors. No prior Triton, CUDA, or GPU architecture knowledge is needed.
> **Related chapters:** [Chapter 1](../../docs/part-1-foundations/01-latency-throughput-flops-bytes-bandwidth.md) (counting bytes), [Chapter 2](../../docs/part-1-foundations/02-arithmetic-intensity-roofline-amdahls-law.md) (roofline), [Chapter 3](../../docs/part-1-foundations/03-compute-memory-bound-launch-overhead.md) (what limits it).

## Contents

1. [The problem](#1-the-problem)
2. [The idea: programs, blocks, and masks](#2-the-idea-programs-blocks-and-masks)
3. [A worked example at n = 100,000](#3-a-worked-example-at-n--100000)
4. [Syntax notes](#4-syntax-notes)
5. [Run it and read the results](#5-run-it-and-read-the-results)
6. [Experiments to try](#6-experiments-to-try)
7. [Open questions](#7-open-questions)

---

## 1. The problem

We have two float32 arrays `x` and `y` of length **100,000** on the GPU, and we want `z = x + y` without writing a Python `for` loop over 100,000 elements.

PyTorch already does this with `x + y`, and that is our reference. For an operation this simple, PyTorch and Triton should end up about equally fast at large sizes, because both are limited by memory (Chapter 2). So the point of this lab is **not speed**. It is to learn Triton's programming model, which we will reuse for every later kernel (softmax, layer norm, matmul, attention).

---

## 2. The idea: programs, blocks, and masks

### 2.1 Many copies of one function

A GPU has thousands of small execution units that work best when they all do the same kind of work at the same time, each on different data. So instead of one loop, we write **one function** and the GPU runs **many copies of it at once**, each copy working on a different slice of the data.

Triton calls each copy a **program**. This pattern is called *Single Program, Multiple Data* (SPMD): every program runs the same code, and only its **id** differs.

### 2.2 Four words to know

| Term | Meaning in Triton | Analogy |
| --- | --- | --- |
| **`pid`** (`tl.program_id`) | Which program am I? An integer `0, 1, 2, ...` | A worker's ID number |
| **`BLOCK_SIZE`** | How many elements one program handles | How many boxes one worker carries |
| **`offsets`** | The specific element positions this program reads and writes | The worker's list of box numbers |
| **`mask`** | `True`/`False` per position: does this position really exist? | A note saying "skip the boxes that aren't there" |

Everything else in the kernel is arithmetic on these four things.

### 2.3 Why do we need a mask?

Say we have **10 numbered boxes** (positions `0` to `9`) and a worker who carries **3 boxes per trip**. We need

```
ceil(10 / 3) = 4 trips
```

But `4 × 3 = 12` slots, and there are only 10 boxes. Trip 4's list says "boxes 9, 10, 11", and boxes 10 and 11 do not exist. If the worker reaches for them anyway, that is an **out-of-bounds memory access**. It does not always crash. Sometimes it quietly reads or writes a wrong value, which is worse.

The mask is a note on each trip's list:

```
trip 4: [ ✅ box 9,   ❌ box 10,   ❌ box 11 ]
```

In Triton it is one line:

```python
mask = offsets < n_elements
```

`offsets` is a list of candidate positions and `n_elements` is one number, so the comparison is done per element and gives a list of `True`/`False`.

### 2.4 How they combine

For `n` elements and block size `B`:

- The number of programs is `ceil(n / B)`.
- Program `p` owns positions `p*B` to `p*B + B - 1`.
- Only the **last** program can have `False` entries in its mask. (If `n` is an exact multiple of `B`, none do.)

For `n = 100,000` and `B = 1024`:

```
ceil(100000 / 1024) = ceil(97.65625) = 98 programs,  pid ∈ {0 ... 97}
```

---

## 3. A worked example at n = 100,000

`n = 100,000`, `BLOCK_SIZE = 1024`, so **`grid = 98`** programs, `pid ∈ {0, 1, ..., 97}`.

### 3.1 What each program owns

| pid | `block_start` (= pid × 1024) | `offsets` range | Valid count | `mask` |
| --- | --- | --- | --- | --- |
| 0 | 0 | 0 … 1,023 | 1024 | all True |
| 1 | 1,024 | 1,024 … 2,047 | 1024 | all True |
| 2 | 2,048 | 2,048 … 3,071 | 1024 | all True |
| ... | ... | ... | ... | all True |
| 96 | 98,304 | 98,304 … 99,327 | 1024 | all True |
| 97 | 99,328 | 99,328 … 100,351 | 672 | True × 672, False × 352 |

Only `pid = 97` has any `False`.

### 3.2 Zoom into the last program (`pid = 97`)

```python
pid          = 97
block_start  = 97 * 1024 = 99,328
offsets      = [99,328, 99,329, ..., 100,351]          # 1024 entries
mask         = offsets < 100,000
             = [True, True, ..., True,  False, False, ..., False]
               └── 672 valid ──────┘  └── 352 invalid ──┘
               (99,328 … 99,999)      (100,000 … 100,351)
```

Two sanity checks:

- `100,000 − 99,328 = 672` real elements left.
- `1024 − 672 = 352` phantom slots that do not correspond to real memory.

### 3.3 The kernel, line by line, for `pid = 97`

| Line | Value for `pid = 97` |
| --- | --- |
| `pid = tl.program_id(axis=0)` | `97` |
| `block_start = pid * BLOCK_SIZE` | `99,328` |
| `offsets = block_start + tl.arange(0, BLOCK_SIZE)` | `[99,328, ..., 100,351]` (1024 values) |
| `mask = offsets < n_elements` | `True × 672, False × 352` |
| `x = tl.load(x_ptr + offsets, mask=mask, other=0.0)` | 672 real values of `x`, then 352 × `0.0` |
| `y = tl.load(y_ptr + offsets, mask=mask, other=0.0)` | same for `y` |
| `output = x + y` | 1024 additions; the last 352 are `0.0 + 0.0` |
| `tl.store(output_ptr + offsets, output, mask=mask)` | writes `output[99,328 … 99,999]` (672 values); skips the rest |

Note that:
- Where `mask` is `False`, `tl.load` does **not touch memory**. What the masked slots hold is only defined because we passed `other=0.0`. Without `other`, the Triton documentation says the masked-out values are undefined, so we should not rely on them being zero.
- The 352 phantom results are never written, so the output array is never touched past index 99,999.

### 3.4 Sanity check: every element is written exactly once

- `pid = 0 … 96` write `output[0 … 99,327]`, which is `97 × 1024 = 99,328` elements.
- `pid = 97` writes `output[99,328 … 99,999]`, which is 672 elements.
- Total: `99,328 + 672 = 100,000`. No gaps and no double writes.

### 3.5 Why not choose `BLOCK_SIZE` to divide `n` evenly?

We could set `BLOCK_SIZE = 100,000` and launch one program, but then:

- One program cannot hold 100,000 values at once.
- We lose all the parallelism (1 worker instead of 98).
- A different `n` would need a different `BLOCK_SIZE`.

So the practical pattern is: pick a power-of-two `BLOCK_SIZE` (256, 512, 1024, ...; Triton requires the length given to `tl.arange` to be a power of two) and always write a mask for the tail. This same pattern shows up in the later operations.

---

## 4. Syntax notes

The comments in `triton_kernel.py` explain each line. These notes cover the Triton ideas that do not fit in a one-line comment.

### 4.1 `@triton.jit`

Triton compiles the function for the GPU, just in time. It looks like Python, but it is **not** ordinary Python: the body can only use the Triton language and simple arithmetic. For example, a plain `print` does not run on the GPU (Triton has `tl.device_print` for that).

### 4.2 Pointer arguments

`x_ptr`, `y_ptr`, and `output_ptr` are **memory addresses**, not arrays. The kernel receives where each array starts and uses pointer arithmetic: `x_ptr + i` means "the address `i` elements past the start of `x`". When we call the kernel with a `torch.Tensor`, Triton passes the address of its first element automatically.

### 4.3 `tl.arange` is not `range`

`tl.arange(0, BLOCK_SIZE)` is a **tensor** of 1024 values, `[0, 1, ..., 1023]`, processed together. Python's `range` is a loop helper. Adding a number to a tensor (`block_start + tl.arange(...)`) adds it to every entry.

### 4.4 `tl.constexpr`

`BLOCK_SIZE: tl.constexpr` is a **compile-time constant**. The shape of a tensor like `tl.arange(0, BLOCK_SIZE)` must be known when the kernel is compiled, and each different `BLOCK_SIZE` produces its own compiled version. We pass it by keyword (`BLOCK_SIZE=1024`) when we launch, as the official Triton tutorial does.

### 4.5 The wrapper: `grid`, `cdiv`, and the launch

```python
output = torch.empty_like(x)
n_elements = output.numel()
grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=block_size)
```

- **`torch.empty_like(x)`** reserves space for the result. "Empty" means uninitialized: the kernel fills it.
- **`triton.cdiv(a, b)`** is ceiling division: `triton.cdiv(100000, 1024) = 98`. It is ceiling, not floor, because 97 full blocks leave a 672-element tail that needs one more program. With floor division the tail would never be written, and the last 672 outputs would contain uninitialized memory.
- **`grid`** is the number of programs to launch, written as a tuple because grids can have up to three dimensions: `(98,)`, not `98`. We write it as a function of `meta` (a dictionary of the compile-time parameters) so it can read `BLOCK_SIZE`. For this lab a plain tuple would also work. The function form matters later, when autotuning tries several block sizes.
- **`add_kernel[grid](...)`** launches the programs. **The launch is asynchronous**: the Python call returns before the GPU is finished. Kernels on the same CUDA stream run in order, and copying results back to the CPU (for example `print(output)`) waits for them. That is also why timing needs synchronization (Chapter 1, section 1.5).

---
