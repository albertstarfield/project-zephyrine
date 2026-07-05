# Contributing to Adelaide_Lite

Thank you for your interest in contributing to Adelaide_Lite! We are an open-source platform pushing the boundaries of AI, robotics, and deterministic avionics architectures. 

## Code Standards
- **Python Code**: Use `Ruff` and `Biome` for static analysis and linting. Python code should be strictly PEP8 compliant and formatted with Ruff.
- **Ada Core**: Ada code should target the Ada 2022 standard and prioritize safety. Use `gnatprove` level 4 to ensure strong typing and SPARK adherence.
- **JavaScript/TypeScript**: Ensure `strict: true` in your `tsconfig.json` and validate with Biome before committing.

## ROS2 and Robotics (ELP2/ELP3)
Adelaide_Lite uses a strict prioritization schema for deterministic tasks. 
- All ROS2 integration must occur in the **ELP2** and **ELP3** priority bands.
- **ELP3 (ZenithOrion)** is reserved strictly for safety-critical, time-sensitive actuators (e.g., balancing) and failure-critical reflexes that require consistent 1ms timing. ROS2 integration for ELP3 should be placed in `ZenithOrion/ROS2/`.
- **ELP2 (StellaIcarus)** is for other actuators and robotics components that are not strictly safety-critical or timing-sensitive. ROS2 integration for ELP2 should be placed in `stellaicarus/ROS2/`.
- Do NOT integrate ROS2 publishers into the non-deterministic LLM generative queue unless via curated tool hooks.
- **For a detailed guide on programming PX4 and ROS2 natively in Ada, see [ROS2 and PX4 Integration Guide](docs/ROS2_PX4_INTEGRATION.md).**

## PR Process
1. Run `./run.sh --verbose` locally to ensure the boot sequence passes all environment verification checks, including Pyrefly and CrossHair.
2. Submit a Pull Request with a clear description of your changes and which components (Ada core, Python Sidecars, ROS2) are affected.
3. The automated CI will test your pipeline latency against our 4-7 microsecond (µs) target window for deterministic hooks.
