# Lab: ReLU (PyTorch vs Triton)

Compute `y = max(0, x)` with PyTorch and with a Triton kernel, check that they agree (including at zero and for all-negative/all-positive inputs), and compare their speed across sizes.

**Read next:** [`note.md`](note.md). It assumes you've already read [`labs/vector_add/note.md`](../vector_add/note.md) and focuses only on what's new here: `tl.maximum`, and why ReLU has no good FLOP count.

**Related chapters:** [Chapter 2](../../docs/part-1-foundations/02-arithmetic-intensity-roofline-amdahls-law.md) (arithmetic intensity), [Chapter 3](../../docs/part-1-foundations/03-compute-memory-bound-launch-overhead.md) (what limits it).

| File | Purpose |
| --- | --- |
| `note.md` | Deep-dive note: what's new versus vector add, worked numbers |
| `pytorch_ref.py` | Baseline: `torch.maximum(x, 0)` |
| `triton_kernel.py` | The Triton kernel and its wrapper, with a comment on every line |
| `test_kernel.py` | Correctness tests: size sweep, zero boundary, all-negative, all-positive |
| `benchmark.py` | Times both versions and saves a CSV and a plot |

## Run it

From the repo root:

```bash
python -m pytest labs/02_relu -v                 # correctness
python -m labs.02_relu.triton_kernel             # quick demo with n = 100,000
python -m labs.02_relu.benchmark                 # FP32 sweep, saves CSV + PNG
python -m labs.02_relu.benchmark --dtype fp16 --peak-bw 320
```

If Python cannot find `labs` or `benchmarks`, add an empty `__init__.py` to `labs/`, `labs/relu/`, and `benchmarks/`.

## Results

Results are saved per GPU in `benchmarks/results/<gpu-name>/`:

- `relu_fp32.csv`
- `relu_fp32.png`

After the first run, embed the plot here (replace `GPU-NAME`):

```markdown
![ReLU: PyTorch vs Triton](../../benchmarks/results/GPU-NAME/relu_fp32.png)
```

## How the benchmark stays fair

- Both versions use the same input, dtype, and timing function (`benchmarks/benchmark_utils.py`).
- The input uses `torch.randn` (not `torch.rand`), so it has a realistic mix of negative and positive values — an all-positive input would turn ReLU into a no-op and hide bugs.
- Correctness is checked before timing.
- Allocation of the output tensor is inside the timed region for both.
- `triton.testing.do_bench` warms up, synchronizes the GPU, and clears the L2 cache between runs.