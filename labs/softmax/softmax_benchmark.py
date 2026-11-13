''' Compare three providers:
    1. PyTorch torch.softmax: reference, single fused C++ kernel
    2. Naive PyTorch (5 ops): the un-fused Python version, shows the cost of materializing intermediates to DRAM
    3. Our Triton fused softmax: one read, one write, all reductions on-chip '''


from benchmark.benchmark_utils import (
    gbps, results_dir, save_csv, time_ms, plot_two_panels,
)
from .pytorch_ref import softmax_ref, naive_softmax
from .triton_kernel import softmax
import torch
import argparse

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

N_COLS = 1024   # fixed, only M grows. Matches row_sum for cross-lab comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=DTYPES, default="fp32")
    parser.add_argument("--min-pow", type=int, default=10)
    parser.add_argument("--max-pow", type=int, default=18)
    parser.add_argument("--peak-bw", type=float, default=None,
                        help="published peak memory bandwidth in GB/s (dashed line)")
    parser.add_argument("--no-naive", action="store_true",
                        help="skip the un-fused PyTorch provider (it is ~5x slower)")
    args = parser.parse_args()

    dtype = DTYPES[args.dtype]

    providers = {
        "Pytorch (torch.softmax)": softmax_ref,
        "Triton (fused)":          softmax,
    }
    if not args.no_naive:
        # Insert naive first so the legend reads "naive -> torch -> triton"
        # (slowest -> fastest), which is the pedagogical order
        providers = {"Naive PyTorch": naive_softmax, **providers}

    rows = []

    for p in range(args.min_pow, args.max_pow + 1):
        M = 2 ** p
        N = N_COLS
        x = torch.randn(M, N, device=DEVICE, dtype=dtype)

        # Correctness gate: every provider must match torch.softmax.
        # Slightly looser tolerances than vector_add because exp/div accumulate
        # small float rounding differences.
        ref = softmax_ref(x)
        torch.testing.assert_close(softmax(x), ref, rtol=1e-5, atol=1e-5)
        if not args.no_naive:
            torch.testing.assert_close(naive_softmax(x), ref, rtol=1e-5, atol=1e-5)

        # Logical I/O for fused softmax: read X once, write Y once
        # use the SAME denominator for all three providers 
        logical_bytes = 2 * M * N * x.element_size()

        for name, fn in providers.items():
            median, p20, p80 = time_ms(lambda: fn(x))
            rows.append({
                "provider": name,
                "dtype": args.dtype,
                "n_rows": M,
                "n_cols": N,
                "logical_bytes": logical_bytes,
                "time_ms": median,
                "time_p20_ms": p20,
                "time_p80_ms": p80,
                "gbps": gbps(logical_bytes, median),
            })
        del x

    out = results_dir("softmax")
    stem = f"softmax_{args.dtype}"
    save_csv(rows, out / f"{stem}.csv")
    plot_two_panels(
        rows, list(providers),
        f"Fused softmax ({args.dtype}, N={N_COLS}): PyTorch vs Triton",
        out / f"{stem}.png",
        args.peak_bw,
    )
    print(f"saved {out / (stem + '.csv')} and {out / (stem + '.png')}")


if __name__ == "__main__":
    main()
