# Glossary

A growing reference for GPU architecture, performance analysis, kernel programming, and inference. Definitions are intentionally concise, follow the linked chapters for deeper explanations.

## Performance fundamentals
- **Latency**: The time required to complete an operation or request. State whether the measurement is kernel-only, device-side, or end-to-end.
- **Throughput**: The amount of work completed per unit time, such as requests/second, tokens/second, or operations/second.
- **FLOP**: A floating-point operation. The exact counting convention should be stated when comparing workloads.
- **FLOP/s**: Floating-point operations per second, a measure of achieved arithmetic throughput. TFLOP/s is (10\^{12}) FLOP/s.
- **Bytes moved**: Data transferred across a specified boundary or memory level. Logical tensor bytes are not necessarily the same as physical DRAM traffic.
- **Bandwidth**: Data transferred per unit time, commonly expressed in GB/s or TB/s. Always clarify which memory level and units are intended.
- **Arithmetic intensity**: Floating-point operations divided by bytes transferred, usually FLOPs/byte. The result depends on which memory transfers are counted.
- **Roofline model**: A performance model that bounds attainable arithmetic throughput by the lesser of peak compute throughput and memory bandwidth multiplied by arithmetic intensity.
- **Ridge point**: The arithmetic intensity where the idealized memory-bandwidth roof meets the peak-compute roof.
- **Amdahl's law**: A model showing that overall speedup is limited by the portion of runtime that an optimization does not accelerate.
- **Compute-bound**: A workload whose performance is limited mainly by available arithmetic throughput.
- **Memory-bound**: A workload whose performance is limited mainly by data movement, often memory bandwidth.
- **Launch-bound**: A workload where dispatch or kernel launch overhead is a substantial part of total time.
- **Speedup**: Baseline runtime divided by optimized runtime, using comparable measurement conditions.
- **P50 / P90 / P95 / P99**: Percentiles of a measured latency distribution. P99 is the value at or below which 99% of observations fall.
