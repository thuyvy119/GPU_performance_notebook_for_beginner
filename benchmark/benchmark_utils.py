import csv
import torch
import triton
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "benchmark" / "results"

def gpu_name():
    return re.sub(r"[^A-Za-z0-9]+", "-", torch.cuda.get_device_name(0)).strip("-")

def time_ms(fn, quantiles=(0.5, 0.2, 0.8)):
    median, p20, p80 = triton.testing.do_bench(fn, quantiles=list(quantiles))
    return median, p20, p80

def gbps(num_bytes, ms):
    return num_bytes/(ms*1e-3)/1e9

def results_dir(op_name):
    path = RESULTS_DIR/op_name
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        
# left: time vs size (log-log), right: effective bandwidth vs size (log x)
def plot_two_panels(rows, providers, title, path, peak_gbps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    fig, (ax_t, ax_b) = plt.subplots(1, 2, figsize=(11, 4))
    for name in providers:
        pts = [r for r in rows if r["provider"] == name]
        sizes = [r["n_elements"] for r in pts]
        ax_t.plot(sizes, [r["time_ms"] for r in pts], marker="o", label=name)
        ax_b.plot(sizes, [r["gbps"] for r in pts], marker="o", label=name)
    
    ax_t.set(xscale="log", yscale="log", xlabel="elements (N)", ylabel="time (ms)", title="Time vs size")
    ax_b.set(xscale="log", xlabel="elements (N)", ylabel="effective bandwidth (GB/s)", title="Effective bandwidth vs size")
    if peak_gbps:
        ax_b.axhline(peak_gbps, linestyle="--", color="gray", label=f"peak {peak_gbps:g} GB/s")
    for ax in (ax_t, ax_b):
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
        
