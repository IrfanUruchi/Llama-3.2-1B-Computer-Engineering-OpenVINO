import argparse
import gc
import os
import json
import platform
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path

import openvino as ov
import openvino_genai as ov_genai


MODEL = Path(
    os.environ.get(
        "OV_MODEL",
        "./Llama-3.2-1B-Computer-Engineering-OpenVINO-INT4",
    )
)

PROMPT_TEXT = """Q: Explain how CPU cache memory improves processor performance, including its relationship with RAM, latency, and memory locality.
A:"""

# OpenVINO GenAI returns DecodedResults with perf_metrics
# when generation input is a list.
PROMPT_BATCH = [PROMPT_TEXT]


def command_output(command):
    try:
        return subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def benchmark_device(device, warmups, iterations, tokens):
    core = ov.Core()

    full_name = core.get_property(
        device,
        "FULL_DEVICE_NAME",
    )

    print()
    print("=" * 78)
    print(f" DEVICE: {device}")
    print(f" NAME:   {full_name}")
    print("=" * 78)

    started = time.perf_counter()

    pipe = ov_genai.LLMPipeline(
        str(MODEL),
        device,
    )

    load_ms = (
        time.perf_counter() - started
    ) * 1000

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = tokens

    # Force identical work on every device.
    config.ignore_eos = True

    # Deterministic benchmark path.
    config.do_sample = False
    config.apply_chat_template = False

    encoded = pipe.get_tokenizer().encode(PROMPT_BATCH)
    prompt_tokens = encoded.input_ids.get_shape()[1]

    print(f"Pipeline load: {load_ms:.2f} ms")
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Forced output tokens: {tokens}")

    print()
    print("=== WARMUP ===")

    for i in range(warmups):
        pipe.generate(
            PROMPT_BATCH,
            config,
        )

        print(
            f"Warmup {i + 1}/{warmups}: complete"
        )

    print()
    print("=== MEASURED RUNS ===")

    runs = []

    for i in range(iterations):
        result = pipe.generate(
            PROMPT_BATCH,
            config,
        )

        metrics = result.perf_metrics

        run = {
            "ttft_ms":
                float(metrics.get_ttft().mean),

            "tpot_ms_per_token":
                float(metrics.get_tpot().mean),

            "throughput_tokens_per_second":
                float(metrics.get_throughput().mean),

            "generated_tokens":
                int(metrics.get_num_generated_tokens()),

            "input_tokens":
                int(metrics.get_num_input_tokens()),
        }

        runs.append(run)

        print(
            f"Run {i + 1}: "
            f"TTFT={run['ttft_ms']:.2f} ms | "
            f"TPOT={run['tpot_ms_per_token']:.2f} ms/token | "
            f"Throughput="
            f"{run['throughput_tokens_per_second']:.2f} tok/s"
        )

    def stats(key):
        values = [run[key] for run in runs]

        mean = statistics.mean(values)

        std = (
            statistics.stdev(values)
            if len(values) > 1
            else 0.0
        )

        return {
            "mean": mean,
            "std": std,
            "min": min(values),
            "max": max(values),
        }

    summary = {
        "device": device,
        "full_device_name": full_name,
        "pipeline_load_ms": load_ms,
        "prompt_tokens": prompt_tokens,
        "forced_output_tokens": tokens,
        "warmups": warmups,
        "iterations": iterations,
        "ttft_ms": stats("ttft_ms"),
        "tpot_ms_per_token": stats(
            "tpot_ms_per_token"
        ),
        "throughput_tokens_per_second": stats(
            "throughput_tokens_per_second"
        ),
        "runs": runs,
    }

    print()
    print("=== FINAL ===")

    print(
        f"TTFT:       "
        f"{summary['ttft_ms']['mean']:.2f} ± "
        f"{summary['ttft_ms']['std']:.2f} ms"
    )

    print(
        f"TPOT:       "
        f"{summary['tpot_ms_per_token']['mean']:.2f} ± "
        f"{summary['tpot_ms_per_token']['std']:.2f} ms/token"
    )

    print(
        f"Throughput: "
        f"{summary['throughput_tokens_per_second']['mean']:.2f} ± "
        f"{summary['throughput_tokens_per_second']['std']:.2f} tok/s"
    )

    del pipe
    gc.collect()

    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        choices=["CPU", "GPU", "NPU", "ALL"],
        default="ALL",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--tokens",
        type=int,
        default=128,
    )

    args = parser.parse_args()

    core = ov.Core()
    available = core.available_devices

    requested = (
        ["CPU", "NPU", "GPU"]
        if args.device == "ALL"
        else [args.device]
    )

    selected = [
        device
        for device in requested
        if device in available
    ]

    print("=" * 78)
    print(" OPENVINO RELEASE BENCHMARK")
    print("=" * 78)

    print("Date:", datetime.now().astimezone().isoformat())
    print("Model:", MODEL)
    print("OpenVINO:", ov.__version__)

    print(
        "OpenVINO GenAI:",
        getattr(
            ov_genai,
            "__version__",
            "unknown",
        ),
    )

    print("Available devices:", available)
    print("Benchmark devices:", selected)

    print(
        "Power profile:",
        command_output(
            ["powerprofilesctl", "get"]
        ),
    )

    print("Kernel:", platform.release())

    print()
    print("NOTE:")
    print(
        "Q:/A: prompt format is used because this model "
        "is a completion-style Computer Engineering model."
    )

    print(
        "ignore_eos=True forces every backend to generate "
        f"exactly {args.tokens} tokens."
    )

    results = []

    for device in selected:
        results.append(
            benchmark_device(
                device,
                args.warmup,
                args.iterations,
                args.tokens,
            )
        )

    report = {
        "timestamp":
            datetime.now().astimezone().isoformat(),

        "model":
            str(MODEL),

        "openvino_version":
            ov.__version__,

        "openvino_genai_version":
            getattr(
                ov_genai,
                "__version__",
                "unknown",
            ),

        "power_profile":
            command_output(
                ["powerprofilesctl", "get"]
            ),

        "kernel":
            platform.release(),

        "prompt":
            PROMPT_TEXT,

        "benchmark_method": {
            "warmups": args.warmup,
            "iterations": args.iterations,
            "max_new_tokens": args.tokens,
            "ignore_eos": True,
            "do_sample": False,
        },

        "devices": results,
    }

    Path("results").mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(
        "results/final-release-benchmark.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print(" FINAL COMPARISON")
    print("=" * 78)

    print(
        f"{'DEVICE':<8}"
        f"{'TTFT ms':>14}"
        f"{'TPOT ms/tok':>16}"
        f"{'TOKENS/s':>14}"
    )

    print("-" * 52)

    for item in results:
        print(
            f"{item['device']:<8}"
            f"{item['ttft_ms']['mean']:>14.2f}"
            f"{item['tpot_ms_per_token']['mean']:>16.2f}"
            f"{item['throughput_tokens_per_second']['mean']:>14.2f}"
        )

    print()
    print(
        "Saved: results/final-release-benchmark.json"
    )


if __name__ == "__main__":
    main()
