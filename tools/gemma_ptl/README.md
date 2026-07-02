# Gemma PTL Sanity Toolkit

This folder contains lightweight checks to validate that your OpenVINO/OpenVINO GenAI
installation is usable on the PTL machine before running Gemma workloads.

## 1) Activate OpenVINO environment

PowerShell:

```powershell
. C:\path\to\openvino\build-x86_64\install\setupvars.ps1
```

## 2) Run environment check

```powershell
python tools/gemma_ptl/check_env.py
```

Expected output pattern:

- `GEMMA_PTL_ENV_RESULT={...}`
- Includes `openvino_version`, `openvino_genai_version`, and `available_devices`.

## 3) Optional NPU/NPUW smoke

```powershell
$env:OPENVINO_NPUW_LOG_LEVEL="DEBUG"
python tools/gemma_ptl/check_env.py --npu-smoke
```

Expected:

- `npu_smoke=PASS` when NPU runtime is healthy.
- `SKIPPED_NO_NPU_DEVICE` if NPU is not exposed on the host.

## 4) Run Gemma workload

After the above checks pass, run your Gemma inference command/script used by your team
(e.g., your existing GenAI sample, benchmark script, or application pipeline).
