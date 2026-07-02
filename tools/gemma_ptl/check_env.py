#!/usr/bin/env python3
"""Minimal PTL smoke check for OpenVINO + OpenVINO GenAI runtime readiness.

This script verifies package imports, visible devices, and (optionally) a tiny NPU
compile/infer smoke path using NPUW property.
"""

from __future__ import annotations

import argparse
import json
import traceback

import numpy as np
import openvino as ov
import openvino_genai  # noqa: F401


def run_npu_smoke() -> dict:
    result: dict[str, object] = {}
    core = ov.Core()
    devices = list(core.available_devices)
    result["available_devices"] = devices
    result["npu_present"] = "NPU" in devices

    # Tiny model for runtime sanity (x -> relu(x)).
    p = ov.opset13.parameter([1, 3], ov.Type.f32, name="x")
    r = ov.opset13.relu(p)
    model = ov.Model([r], [p], "npuw_smoke")

    if not result["npu_present"]:
        result["npu_smoke"] = "SKIPPED_NO_NPU_DEVICE"
        return result

    try:
        compiled = core.compile_model(model, "NPU", {"NPU_USE_NPUW": "YES"})
        output = compiled([np.array([[1.0, -2.0, 3.0]], dtype=np.float32)])
        first_key = next(iter(output))
        result["npu_smoke"] = "PASS"
        result["output_sample"] = output[first_key].tolist()
    except Exception as exc:  # noqa: BLE001
        result["npu_smoke"] = "FAIL"
        result["error"] = str(exc)
        result["trace"] = traceback.format_exc(limit=1)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="PTL environment smoke check")
    parser.add_argument(
        "--npu-smoke",
        action="store_true",
        help="Try tiny compile/infer on NPU with NPU_USE_NPUW=YES",
    )
    args = parser.parse_args()

    summary: dict[str, object] = {
        "openvino_version": getattr(ov, "__version__", "unknown"),
        "openvino_genai_version": getattr(openvino_genai, "__version__", "unknown"),
    }

    core = ov.Core()
    summary["available_devices"] = list(core.available_devices)

    if args.npu_smoke:
        summary.update(run_npu_smoke())

    print("GEMMA_PTL_ENV_RESULT=" + json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
