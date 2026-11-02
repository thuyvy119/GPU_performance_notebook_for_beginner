# Lab: Vector Addition (PyTorch vs Triton)

Compute `z = x + y` with PyTorch and with a Triton kernel, check if they are equal, and compare their speed across sizes.

**Read next:** [`note.md`](note.md) explains the Triton kernel with a worked example (`n = 100,000`, `BLOCK_SIZE = 1024`, 98 programs).

**Related chapters:** [Chapter 1](../../docs/part-1-foundations/01-latency-throughput-flops-bytes-bandwidth.md) (bytes and bandwidth), [Chapter 2](../../docs/part-1-foundations/02-arithmetic-intensity-roofline-amdahls-law.md) (roofline), [Chapter 3](../../docs/part-1-foundations/03-compute-memory-bound-launch-overhead.md) (what limits it).

| File | Purpose |
| --- | --- |
| `note.md` | Deep-dive note: programs, blocks, masks, worked numbers |
| `pytorch_ref.py` | Baseline: `x + y` |
| `triton_kernel.py` | The Triton kernel and its wrapper, with a comment on every line |
| `test_kernel.py` | Correctness tests, including sizes that are not a multiple of the block size |
| `benchmark.py` | Times both versions and saves a CSV and a plot |

## Run it

From the repo root (this also works in Google Colab after cloning the repo and moving into it):

```bash
python -m pytest labs/01_vector_add -v                 # correctness
python -m labs.01_vector_add.triton_kernel             # quick demo with n = 100,000
python -m labs.01_vector_add.benchmark                 # FP32 sweep, saves CSV + PNG
python -m labs.01_vector_add.benchmark --dtype fp16 --peak-bw 320
```

If Python cannot find `labs` or `benchmarks`, add an empty `__init__.py` to `labs/`, `labs/vector_add/`, and `benchmarks/`.

## Results

Results are saved per GPU in `benchmarks/results/`:

- `vector_add_fp32.csv`
- `vector_add_fp32.png`


```markdown
![Vector add: PyTorch vs Triton](../../benchmarks/results/vector_add_fp32.png)
```

## How the benchmark stays fair

- Both versions use the same inputs, dtype, and timing function (`benchmarks/benchmark_utils.py`).
- Correctness is checked before timing.
- Allocation of the output tensor is inside the timed region for both.
- `triton.testing.do_bench` warms up, synchronizes the GPU, and clears the L2 cache between runs.