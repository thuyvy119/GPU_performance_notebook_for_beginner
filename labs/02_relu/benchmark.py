import argparse
import torch

from benchmark.benchmark_utils import gbps, results_dir, save_csv, time_ms, plot_two_panels
from pytorch_ref import relu_ref
from triton_kernel import relu

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=DTYPES, default="fp32")
    parser.add_argument("--min-pow", type=int, default=10, help="smallest size is 2**min_pow")
    parser.add_argument("--max-pow", type=float, default=None, help="largest size is 2**max_pow")
    parser.add_argument("--peak-bw", type=float, default=None, help="published peak memory bandwidth in GB/s (draws a dashed line)")
    
    args = parser.parse_args()
    dtype = DTYPES[args.dtype]
    providers = {"Pytorch": relu_ref, "Triton": relu}
    rows = []
    
    for p in range(args.min_pow, args.max_pow+1):
        n = 2**p
        x = torch.rand(n, device="cuda", dtype=dtype)
        
        torch.testing.assert_close(relu(x), relu_ref(x))
        
        logical_bytes = 2 * n * x.element_size()
        
        for name, fn in providers.items():
            median, p20, p80 = time_ms(lambda: fn(x))
            rows.append({
                "provider": name, "dtype": args.dtype, "n_elements": n,
                "logical_bytes": logical_bytes, 
                "time_ms": median, "time_p20_ms": p20, "time_p80_ms": p80,
                "gbps": gbps(logical_bytes, median),
            })
        del x
        
    out = results_dir("relu")
    stem = f"relu_{args.dtype}"
    save_csv(rows, out/f"{stem}.csv")
    plot_two_panels(rows, list(providers), f"ReLU ({args.dtype}): Pytorch vs Triton",
                    out/f"{stem}.png", args.peak_bw)
    print(f"saved {out/(stem + '.csv')} and {out/(stem + '.png')}")
    
if __name__ == "__main__":
    main()
    