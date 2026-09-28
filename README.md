# Llama-3.2-1B Computer Engineering - OpenVINO

OpenVINO deployment and benchmarking of the **Llama-3.2-1B Computer Engineering** model across Intel CPU, integrated GPU, and NPU on **IULinux**.

Three OpenVINO variants were produced and published on Hugging Face:

| Variant | Approx. size | Hugging Face |
|---|---:|---|
| INT4 | 758 MB | [OpenVINO INT4](https://huggingface.co/Irfanuruchi/Llama-3.2-1B-Computer-Engineering-OpenVINO-INT4) |
| INT8 | 1.2 GB | [OpenVINO INT8](https://huggingface.co/Irfanuruchi/Llama-3.2-1B-Computer-Engineering-OpenVINO-INT8) |
| FP16 | 2.4 GB | [OpenVINO FP16](https://huggingface.co/Irfanuruchi/Llama-3.2-1B-Computer-Engineering-OpenVINO-FP16) |

Parent model:

[Llama-3.2-1B-Computer-Engineering-LLM](https://huggingface.co/Irfanuruchi/Llama-3.2-1B-Computer-Engineering-LLM)

This GitHub repository does **not** contain model weights. The model files are distributed through Hugging Face. This repository contains conversion notes, benchmark code, environment information, and raw benchmark results.

## Conversion path

The published parent checkpoint is a BitsAndBytes INT8 model.

Inspection of the checkpoint confirmed actual INT8 linear weights, SCB scaling tensors, and 112 `Linear8bitLt` modules.

The deployment path was:

```text
meta-llama/Llama-3.2-1B
        ↓
Computer Engineering fine-tuning
        ↓
BitsAndBytes INT8 checkpoint
        ↓
Dense FP16 reconstruction
        ↓
        ├── OpenVINO FP16
        ├── OpenVINO INT8
        └── OpenVINO INT4
```

After dequantization, the dense intermediate contained:

```text
146 FP16 tensors
0 BitsAndBytes linear modules
0 INT8 weight tensors
0 SCB tensors
```

The FP16 representation in this project is therefore **not the original pre-quantization FP16 checkpoint**.

It is a dense FP16 reconstruction of the published BitsAndBytes INT8 model. Quantization is lossy, so dequantization cannot recover information already discarded by the original INT8 quantization.

## OpenVINO exports

### INT4

```bash
optimum-cli export openvino \
  --model ./source-dequant-fp16 \
  --task text-generation-with-past \
  --weight-format int4 \
  --sym \
  --ratio 1.0 \
  --group-size 128 \
  ./Llama-3.2-1B-Computer-Engineering-OpenVINO-INT4
```

The 112 ratio-defining layers were compressed with symmetric INT4 using group size 128.

### INT8

```bash
optimum-cli export openvino \
  --model ./source-dequant-fp16 \
  --task text-generation-with-past \
  --weight-format int8 \
  --sym \
  ./Llama-3.2-1B-Computer-Engineering-OpenVINO-INT8
```

### FP16

```bash
optimum-cli export openvino \
  --model ./source-dequant-fp16 \
  --task text-generation-with-past \
  --weight-format fp16 \
  ./Llama-3.2-1B-Computer-Engineering-OpenVINO-FP16
```

## Test system

All final OpenVINO benchmarks were collected on **IULinux 0.1 Development**.

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

OpenVINO exposed all three Intel backends:

```text
CPU: Intel(R) Core(TM) Ultra 9 275HX
GPU: Intel(R) Graphics (iGPU)
NPU: Intel(R) AI Boost
```

The Intel AI Boost NPU used the Intel NPU Linux stack and Level Zero runtime.

The integrated GPU used the `i915` kernel driver and Intel OpenCL compute runtime.

No NVIDIA driver modification was required for the OpenVINO deployment.

## Benchmark method

The same workload was used for every precision and every backend.

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

`ignore_eos=True` forced every test to generate the same 128-token workload.

Greedy decoding was used for deterministic benchmarking only.

## Throughput

Generation throughput in tokens per second:

| Precision | CPU | Intel iGPU | Intel NPU |
|---|---:|---:|---:|
| INT4 | **62.98** | **62.59** | 33.89 |
| INT8 | 50.64 | 41.86 | **34.30** |
| FP16 | 26.34 | 24.43 | 20.63 |

For this workload, INT4 produced the highest CPU and integrated-GPU throughput.

INT8 produced the highest measured NPU throughput, although the INT4 and INT8 NPU results were close.

## INT4 results

| Device | TTFT | TPOT | Throughput |
|---|---:|---:|---:|
| CPU | 51.57 ± 9.91 ms | 15.90 ± 0.74 ms/token | 62.98 ± 2.79 tok/s |
| iGPU | 54.24 ± 1.77 ms | 15.98 ± 0.29 ms/token | 62.59 ± 1.14 tok/s |
| NPU | 1147.08 ± 4.16 ms | 29.51 ± 0.10 ms/token | 33.89 ± 0.11 tok/s |

## INT8 results

| Device | TTFT | TPOT | Throughput |
|---|---:|---:|---:|
| CPU | 44.24 ± 4.97 ms | 19.75 ± 0.06 ms/token | 50.64 ± 0.16 tok/s |
| iGPU | 34.60 ± 0.12 ms | 23.89 ± 0.06 ms/token | 41.86 ± 0.11 tok/s |
| NPU | 922.20 ± 1.20 ms | 29.15 ± 0.09 ms/token | 34.30 ± 0.11 tok/s |

## FP16 results

| Device | TTFT | TPOT | Throughput |
|---|---:|---:|---:|
| CPU | 127.76 ± 3.99 ms | 37.96 ± 0.12 ms/token | 26.34 ± 0.09 tok/s |
| iGPU | 46.95 ± 0.53 ms | 40.93 ± 0.02 ms/token | 24.43 ± 0.01 tok/s |
| NPU | 981.90 ± 1.69 ms | 48.49 ± 0.20 ms/token | 20.63 ± 0.09 tok/s |

No device-level power measurements were collected. These results compare latency and generation throughput only.

## Prompt format

During testing, the model performed substantially better using its intended completion-style Q/A format than generic conversational prompting.

Recommended format:

```text
Q: <computer engineering question>
A:
```

Example:

```text
Q: Explain the purpose of CPU cache memory.
A:
```

Recommended generation settings for normal use:

```python
max_new_tokens = 200
temperature = 0.7
top_p = 0.9
do_sample = True
repetition_penalty = 1.1
```

The benchmark itself used greedy decoding so the workload remained deterministic.

## Repetition check

Using the Q/A prompt format, the following average repeated 4-gram ratios were observed:

| Variant | Repeated 4-gram ratio |
|---|---:|
| FP16 | 0.002 |
| INT8 | 0.000 |
| INT4 | 0.003 |

These values are diagnostic repetition measurements, not formal model-quality scores.

The OpenVINO FP16 representation also matched the reconstructed dense FP16 model during the earlier deterministic comparison.

## IULinux

The complete OpenVINO conversion and validation workflow was performed on IULinux.

The initial OpenVINO installation exposed CPU and NPU execution.

The Intel integrated GPU already used the `i915` kernel driver and exposed:

```text
/dev/dri/renderD128
```

Installing the Intel OpenCL compute runtime:

```bash
sudo apt install intel-opencl-icd
```

made the Intel GPU available to OpenVINO as an additional execution backend.

After setup:

```text
['CPU', 'GPU', 'NPU']
```

The NPU environment included the Intel NPU compiler, firmware, Level Zero runtime, and `libze1`.

More IULinux and Intel backend setup details are available in [`docs/iulinux.md`](docs/iulinux.md).

## Repository contents

```text
.
├── README.md
├── benchmark/
│   └── benchmark_release.py
├── results/
│   ├── environment.txt
│   ├── fp16.json
│   ├── fp16.txt
│   ├── int4.json
│   ├── int4.txt
│   ├── int8.json
│   └── int8.txt
└── docs/
    ├── conversion.md
    └── iulinux.md
```

The JSON files contain the measured runs and aggregate timing data.

The text files contain the complete benchmark console output.

The benchmark implementation is available at:

[`benchmark/benchmark_release.py`](benchmark/benchmark_release.py)

## Models

Actual model files are hosted on Hugging Face and are intentionally not duplicated in this GitHub repository.

- [OpenVINO INT4](https://huggingface.co/Irfanuruchi/Llama-3.2-1B-Computer-Engineering-OpenVINO-INT4)
- [OpenVINO INT8](https://huggingface.co/Irfanuruchi/Llama-3.2-1B-Computer-Engineering-OpenVINO-INT8)
- [OpenVINO FP16](https://huggingface.co/Irfanuruchi/Llama-3.2-1B-Computer-Engineering-OpenVINO-FP16)
- [Parent Computer Engineering model](https://huggingface.co/Irfanuruchi/Llama-3.2-1B-Computer-Engineering-LLM)

## Notes

Benchmark results apply only to the tested hardware, software versions, runtime configuration, power profile, thermal conditions, and generation settings.

The three OpenVINO variants derive from a parent checkpoint that had already undergone BitsAndBytes INT8 quantization.

More details:

- [Conversion and quantization](docs/conversion.md)
- [IULinux and Intel backend setup](docs/iulinux.md)

---

## License

The OpenVINO model variants documented in this repository are derivative works of **Llama 3.2** and are subject to the [Llama 3.2 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE) and the [Llama 3.2 Acceptable Use Policy](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/USE_POLICY.md).

**Built with Llama.**

Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.

The OpenVINO model files themselves are distributed through the linked Hugging Face repositories. This GitHub repository contains benchmark code, conversion documentation, environment information, and benchmark results.
