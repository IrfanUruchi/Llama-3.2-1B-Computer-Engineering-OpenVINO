# IULinux setup

The complete OpenVINO conversion, Intel backend setup, and final benchmark workflow was performed on **IULinux 0.1 Development**.

## Test system

| Component | Configuration |
|---|---|
| OS | IULinux 0.1 Development |
| Kernel | 7.0.0-31-generic |
| CPU | Intel Core Ultra 9 275HX |
| GPU | Intel Graphics integrated GPU |
| NPU | Intel AI Boost |
| OpenVINO | 2026.4 |
| OpenVINO GenAI | 2026.4 |
| Power profile | Performance |
| Power source | AC |

The final OpenVINO installation exposed:

```text
['CPU', 'GPU', 'NPU']
```

Device names:

```text
CPU: Intel(R) Core(TM) Ultra 9 275HX
GPU: Intel(R) Graphics (iGPU)
NPU: Intel(R) AI Boost
```

## Intel NPU

The Intel AI Boost NPU was exposed by the kernel through:

```text
/dev/accel/accel0
```

The kernel module was:

```text
intel_vpu
```

The user was added to the render group:

```bash
sudo gpasswd -a "$USER" render
```

The Intel NPU userspace stack installed for the final setup included:

```text
intel-driver-compiler-npu
intel-fw-npu
intel-level-zero-npu
libze1
```

The tested Intel NPU release was:

```text
1.38.0
```

After installation, OpenVINO detected:

```text
NPU: Intel(R) AI Boost
```

## Intel integrated GPU

The Intel integrated GPU was detected as:

```text
Intel Arrow Lake-S [Intel Graphics]
```

PCI ID:

```text
8086:7d67
```

Kernel driver:

```text
i915
```

The DRM render device was available as:

```text
/dev/dri/renderD128
```

Initially, OpenVINO did not expose the integrated GPU as a compute backend.

`clinfo` was installed, but no usable Intel OpenCL device was available.

The Intel OpenCL compute runtime was installed with:

```bash
sudo apt install intel-opencl-icd
```

This installed the Intel OpenCL runtime together with its required graphics compiler components.

After installation:

```text
Platform #0: Intel(R) OpenCL Graphics
 `-- Device #0: Intel(R) Graphics
```

OpenVINO then exposed:

```text
CPU
GPU
NPU
```

No NVIDIA driver modification was required.

## OpenVINO environment

The final Python environment used:

```text
Python 3.14.4

OpenVINO:
2026.4.0-22959-99c81491cc3-releases/2026/4

OpenVINO GenAI:
2026.4.0.0-3407-7ea2546852a

Transformers:
5.5.4

PyTorch:
2.14.0+cu130

bitsandbytes:
0.50.2

accelerate:
1.15.0
```

The complete captured environment is stored at:

```text
results/environment.txt
```

## Benchmark configuration

All final benchmark runs used the same method:

```text
Prompt tokens: 26
Generated tokens: 128
Warm-up runs: 2
Measured runs: 5
do_sample: false
ignore_eos: true
```

Benchmark prompt:

```text
Q: Explain how CPU cache memory improves processor performance, including its relationship with RAM, latency, and memory locality.
A:
```

`ignore_eos=True` forced every backend to generate the same 128-token workload.

The machine was connected to AC power and the IULinux power profile was set to:

```text
performance
```

## Results

Generation throughput:

| Precision | CPU | Intel iGPU | Intel NPU |
|---|---:|---:|---:|
| INT4 | 62.98 tok/s | 62.59 tok/s | 33.89 tok/s |
| INT8 | 50.64 tok/s | 41.86 tok/s | 34.30 tok/s |
| FP16 | 26.34 tok/s | 24.43 tok/s | 20.63 tok/s |

The full raw benchmark results are stored under:

```text
results/
```

The benchmark implementation is:

```text
benchmark/benchmark_release.py
```

No device-level power measurements were collected, so the results should be interpreted as latency and throughput measurements only.
