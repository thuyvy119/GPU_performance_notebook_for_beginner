# GPU Performance Notebook for Beginners

**From GPU fundamentals to kernel optimization and LLM inference performance.**

GPU performance can feel overwhelming when terms such as *warp, SM, bandwidth, arithmetic intensity, occupancy, GEMM, CUDA Core, Tensor Core,* and *kernel fusion* appear without a clear explanation of how they relate. This repository is a beginner-friendly, practical guide to understanding those concepts, implementing GPU operations, measuring their performance, and explaining why an optimization helps or does not.

The goal is not to collect optimization tricks. It is to build a systematic way to reason about GPU workloads:
**Understand the workload → predict the bottleneck → implement a baseline → measure → optimize → explain the result.**

## Why this exists

I started this notebook while learning GPU performance engineering myself, not after mastering it. I know the feeling of being handed ten new terms in one paragraph and not knowing which one to look up first, or finding three tutorials that each assume you already know what the other two explain. This is my attempt to write the resource I wish I had: one place that connects the basics, the foundations, and the surrounding concepts in a systematic order, instead of scattering them across blog posts and docs pages.

I'm sharing it publicly because I'd rather learn in the open than alone. If something here is inaccurate, unclear, or out of date, please open an issue or a pull request. Corrections from more experienced readers are one of the most useful things this project can receive, and beginner questions are just as welcome, since they usually point at a place the explanation isn't clear enough yet.

**A note on how this was written:** I used AI-assisted tools (chat and coding assistants) while researching, drafting, and editing this notebook, alongside my own study and testing. I've tried to verify technical claims against primary sources (official docs, papers) rather than trust them blindly, and references are included so you can check them yourself. If you spot a claim that doesn't hold up, that is exactly the kind of correction this project needs.

## Who is this for?

This guide is for people who:
- Know some Python and basic linear algebra but are new to GPU performance engineering.
- Want to understand what happens underneath PyTorch operations.
- Are learning Triton or CUDA and want to understand the concepts behind kernel code.
- Want to benchmark kernels correctly rather than rely on one timing or speedup number.
- Are interested in GPU computing, deep learning systems, and LLM inference optimization.

You do not need to be a CUDA expert to begin. Some familiarity with Python, tensors, matrix multiplication, and basic neural networks will help.

## What you'll learn

The guide is organized as a progressive book, with implementation labs and performance investigations alongside the theory.

1. **Performance fundamentals** — latency, throughput, FLOPs, bytes moved, bandwidth, arithmetic intensity, the roofline model, bottlenecks, and Amdahl's law.
2. **GPU architecture and execution** — GPCs, SMs, grids, thread blocks, warps, threads, SIMT, and tile programming.
3. **Memory and numerical computation** — memory hierarchy, coalescing, bank conflicts, occupancy, register pressure, and precision formats.
4. **Core GPU operations** — elementwise operations, reductions, softmax, normalization, and matrix multiplication.
5. **Optimization and measurement** — fusion, tiling, autotuning, asynchronous execution, CUDA Graphs, benchmarking, and profiling.
6. **GPU programming systems** — PyTorch dispatch, Triton, CUDA, cuBLAS, cuBLASLt, and CUTLASS.
7. **LLM inference performance** — attention, FlashAttention, KV cache, prefill, decode, batching, and inference case studies.

The roadmap is intentionally broader than the initial implementation set. Chapters and labs will be added incrementally as the project develops; see [`docs/learning-roadmap.md`](docs/learning-roadmap.md) for current status.

## How to use this repository

### If you are new to GPU performance

1. Read [`docs/index.md`](docs/index.md) to see the notebook structure.
2. Follow [`docs/learning-roadmap.md`](docs/learning-roadmap.md) in order.
3. Read the theory before attempting a related lab.
4. Use [`docs/glossary.md`](docs/glossary.md) when you encounter unfamiliar terms.
5. Run the lab, check correctness, and then benchmark it.
6. Read the analysis and try to explain the result in your own words.

### If you want to implement kernels

Start with the PyTorch reference operations, then compare them with Triton implementations. A reference implementation gives you a correctness baseline, a benchmark tells you how the implementations behave, and profiling helps explain why.

Do not assume a custom kernel will always outperform PyTorch or an optimized library. The goal is to understand the trade-offs and the workload conditions under which an optimization is useful.

### If you want to study a specific concept

Use the glossary to find the term, then follow its links to the relevant chapter and lab. The glossary is a navigation aid, not a replacement for the full explanation.

## Repository map

```text
gpu-performance-engineering/
├── README.md
├── LICENSE
├── docs/
│   ├── index.md
│   ├── glossary.md
│   ├── learning-roadmap.md
│   ├── benchmark-methodology.md
│   ├── troubleshooting.md
│   └── part-1-foundations/
│       ├── 01-latency-throughput-flops-bytes-bandwidth.md
│       ├── 02-arithmetic-intensity-ridge-point-roofline-amdahls-law.md
│       └── 03-compute-memory-bound-launch-overhead.md
│
├── labs/
│   ├── 00-environment-check/
│   ├── vector_add/
│   │   ├── README.md
│   │   ├── note.md
│   │   ├── pytorch_ref.py
│   │   ├── triton_kernel.py
│   │   ├── test_kernel.py
│   │   └── benchmark.py
│   ├── relu/
│   ├── row_sum/
│   ├── softmax/
│   ├── matmul/
│   └── layer_norm/
│
├── benchmarks/
└── diagrams/
```

This is the initial documentation layout. Folders for later topics (memory patterns, profiling, attention, and so on) will be introduced as the corresponding chapters and labs are developed.

## Learning philosophy

Each topic connects five questions:

1. **What is it?** — Define the concept in accessible language.
2. **Why does it matter?** — Identify the performance problem it addresses.
3. **How does it connect?** — Link it to hardware, algorithms, and other concepts.
4. **How can we test it?** — Use a small experiment or implementation when appropriate.
5. **What did we learn?** — Report evidence, limitations, and trade-offs.

Not every chapter needs a coding lab. Some topics are best introduced through diagrams, equations, or thought experiments. When code is useful, the preferred path is:

**Theory → PyTorch reference → Triton implementation → correctness test → benchmark → analysis.**

## Benchmarking principles

Performance numbers are meaningful only when the measurement method and environment are clear. This project aims to:

- Verify correctness before measuring speed.
- Separate compilation and warmup from steady-state execution when appropriate.
- Account for asynchronous GPU execution.
- Use repeated measurements and compare like with like.
- Record hardware, software versions, input shapes, dtypes, and benchmark settings.
- Explain results instead of presenting speedup numbers without context.

See [`docs/benchmark-methodology.md`](docs/benchmark-methodology.md).

## Hardware and software

Some concepts can be learned without a dedicated GPU, but running Triton and CUDA labs requires a compatible GPU and software environment. Availability depends on the lab.

The guide focuses primarily on NVIDIA CUDA GPUs and Triton. Hardware-dependent behavior, software versions, and unsupported configurations should be documented alongside each lab. Do not assume that results from one GPU architecture generalize to every GPU.

## References

The guide draws on official documentation and research, including:

- [NVIDIA GPU Performance Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)
- [NVIDIA Matrix Multiplication Performance](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)
- [CUDA Programming Guide: Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/)
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)

Each chapter's own references section cites the sources for its specific technical claims. The documentation distinguishes official guarantees from simplified explanations or observations from a particular experiment; verify links before relying on them, since vendor documentation is reorganized often.

## Project status

This is an evolving educational project, written by someone learning the material alongside everyone reading it. The roadmap describes the intended direction; not every planned chapter or lab is available yet. Examples and benchmark results should be treated as hardware- and environment-specific unless stated otherwise.

## Contributing and feedback

Suggestions, corrections, reproducible benchmark results, and beginner questions are welcome. When reporting a performance result, include the hardware, software versions, input shapes, dtype, measurement method, and code or configuration needed to reproduce it.

The most useful contributions help explain not only **what happened**, but also **why it happened**.