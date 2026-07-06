# Zephyrine Platform Stop Codes

This reference documents the system `STOP` codes used in the BSOD (Blue Screen of Death) display screen during critical boot or runtime failures.

## Reference Map

| Hex Stop Code | Error Name | Description |
|---|---|---|
| `0x0000007B` | `ENV_CHECK_FAILURE` | Critical environment requirements or tools (e.g. package managers, runtime SDK) are missing. |
| `0x00000001` | `CORE_INIT_FAILURE` | Initializing the core backend service failed. |
| `0x00000002` | `FRONTEND_INIT_FAILURE` | Preparing the user interface assets failed. |
| `0x00000003` | `INTEGRITY_CHECK_FAILURE` | Platform self-integrity check failed. |
| `0x00000004` | `VAD_BOOTSTRAP_FAILURE` | Setting up the Voice Activity Detection (VAD) ONNX sidecar environment failed. |
| `0x00000005` | `LSH_BOOTSTRAP_FAILURE` | Setting up the sequence worker runtime environment failed. |
| `0x00000006` | `SERVER_CRASHED` | The core server process exited unexpectedly with a non-zero exit code. |
