# Conversion

This document describes the conversion path used to produce the OpenVINO FP16, INT8, and INT4 versions of the Llama-3.2-1B Computer Engineering model.

## Parent model

The source checkpoint is:

`Irfanuruchi/Llama-3.2-1B-Computer-Engineering-LLM`

Hugging Face:

https://huggingface.co/Irfanuruchi/Llama-3.2-1B-Computer-Engineering-LLM

The published parent checkpoint uses BitsAndBytes INT8 quantization.

Inspection of the checkpoint confirmed:

```text
112 Linear8bitLt modules
INT8 linear weights
SCB scaling tensors
weight_format metadata
```

The checkpoint therefore could not be treated as a normal dense FP16 source model.

## Config repair

Before conversion, the parent `config.json` contained 68 U+00A0 NO-BREAK SPACE characters in its indentation.

The repair only normalized the invalid whitespace and canonicalized the JSON.

No model weights, tokenizer files, architecture values, or model semantics were changed.

The repaired configuration was validated with `json.tool` and Transformers `AutoConfig`.

SHA-256:

```text
6b15a3a594312d4dd7afb86b53c8746e67be92d82639274028f708e0f820595c
```

## Dequantization

The BitsAndBytes INT8 parent was loaded and dequantized into a dense FP16 representation.

After dequantization:

```text
146 FP16 tensors
0 BitsAndBytes linear modules
0 INT8 weight tensors
0 SCB tensors
```

The dense checkpoint was saved with:

```python
model.save_pretrained(
    OUT,
    safe_serialization=True,
    save_original_format=False,
)
```

The dense FP16 checkpoint is an intermediate representation used for the OpenVINO exports.

It is **not the original pre-quantization FP16 model**.

The conversion path is:

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

Quantization is lossy. Dequantizing the published INT8 checkpoint cannot recover information already discarded during the original INT8 quantization.

## FP16

The reconstructed dense checkpoint was exported to OpenVINO FP16 with:

```bash
optimum-cli export openvino \
  --model ./source-dequant-fp16 \
  --task text-generation-with-past \
  --weight-format fp16 \
  ./Llama-3.2-1B-Computer-Engineering-OpenVINO-FP16
```

Observed local size:

```text
~2.4 GB
```

Published model:

https://huggingface.co/Irfanuruchi/Llama-3.2-1B-Computer-Engineering-OpenVINO-FP16

## INT8

The dense intermediate was exported using symmetric INT8 OpenVINO weight compression:

```bash
optimum-cli export openvino \
  --model ./source-dequant-fp16 \
  --task text-generation-with-past \
  --weight-format int8 \
  --sym \
  ./Llama-3.2-1B-Computer-Engineering-OpenVINO-INT8
```

The export used symmetric per-channel INT8 weight compression.

Observed local size:

```text
~1.2 GB
```

Published model:

https://huggingface.co/Irfanuruchi/Llama-3.2-1B-Computer-Engineering-OpenVINO-INT8

## INT4

The INT4 export used symmetric group-wise compression:

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

NNCF reported:

```text
112 / 112 ratio-defining layers using INT4
group size: 128
symmetric compression
```

Observed local size:

```text
~758 MB
```

Published model:

https://huggingface.co/Irfanuruchi/Llama-3.2-1B-Computer-Engineering-OpenVINO-INT4

## Sizes

| Variant | Approximate size |
|---|---:|
| FP16 | 2.4 GB |
| INT8 | 1.2 GB |
| INT4 | 758 MB |

## Prompt format

During validation, the model worked substantially better using a completion-style Q/A prompt:

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

Greedy decoding was used only for deterministic benchmarking.

## Repetition check

Using the Q/A format:

| Variant | Average repeated 4-gram ratio |
|---|---:|
| FP16 | 0.002 |
| INT8 | 0.000 |
| INT4 | 0.003 |

These values are diagnostic repetition measurements and are not formal model-quality scores.

The OpenVINO FP16 output also matched the reconstructed dense FP16 representation during the earlier deterministic comparison.
