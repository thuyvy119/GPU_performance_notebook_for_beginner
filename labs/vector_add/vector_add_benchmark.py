from benchmark.benchmark_utils import gbps, results_dir, save_csv, time_ms, plot_two_panels
from .pytorch_ref import vector_add_ref
from .triton_kernel import vector_add
import torch
import argparse

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=DTYPES, default="fp32")
    parser.add_argument("--min-pow", type=int, default=10, help="smallest size is 2**min_pow")
    parser.add_argument("--max-pow", type=int, default=24, help="largest size is 2**max_pow")
    parser.add_argument("--peak-bw", type=float, default=None, help="published peak memory bandwidth in GB/s (draws a dashed line)")
    
    args = parser.parse_args()
    dtype = DTYPES[args.dtype]
    providers = {"Pytorch": vector_add_ref, "Triton": vector_add}
    rows = []
    
    for p in range(args.min_pow, args.max_pow+1):
        n = 2**p
        x = torch.rand(n, device=DEVICE, dtype=dtype)
        y = torch.rand(n, device=DEVICE, dtype=dtype)
        
        torch.testing.assert_close(vector_add(x, y), vector_add_ref(x, y))
        
        logical_bytes = 3 * n * x.element_size()
        
        for name, fn in providers.items():
            median, p20, p80 = time_ms(lambda: fn(x, y))
            rows.append({
                "provider": name, "dtype": args.dtype, "n_elements": n,
                "logical_bytes": logical_bytes, 
                "time_ms": median, "time_p20_ms": p20, "time_p80_ms": p80,
                "gbps": gbps(logical_bytes, median),
            })
        del x, y
        
    out = results_dir("vector_add")
    stem = f"vector_add_{args.dtype}"
    save_csv(rows, out/f"{stem}.csv")
    plot_two_panels(rows, list(providers), f"Vector add ({args.dtype}): Pytorch vs Triton",
                    out/f"{stem}.png", args.peak_bw)
    print(f"saved {out/(stem + '.csv')} and {out/(stem + '.png')}")
    
if __name__ == "__main__":
    main()
