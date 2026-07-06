# ROS2 and PX4 Integration Guide

Welcome to the Adelaide_Lite robotics integration guide! This document explains how to wire and program ROS2 and PX4 components while adhering to the deterministic safety standards of our architecture.

## Architectural Constraints (CRITICAL)

Adelaide_Lite runs a multi-band priority scheduler:
- **ELP0/ELP1**: Non-deterministic (LLM Generative Tasks, Web Search, Data Processing)
- **ELP2 (StellaIcarus)**: Deterministic low-priority robotics (e.g., Lidar sweeps, path planning)
- **ELP3 (ZenithOrion)**: Deterministic high-priority robotics (e.g., 4000Hz balancing, flight control)

### Golden Rules:
1. **Never** block an ELP3 thread waiting for a network packet or a lock.
2. **Never** put native FFI/ROS2 publishers in ELP0/ELP1 if they can crash the Ada process. The only exceptions are the curated tool hooks (like `px4_gnc` and `ros2_actuate`) which have explicit safety wrappers.
3. Keep ROS2 publishers for safety-critical actuators in `ZenithOrion/ROS2/`.

## Programming PX4 (Native FFI)

To achieve `< 0.25ms` latency for flight dynamics, we use native C FFI to bind to the MAVLink `c_library_v2`.

### Wiring a New PX4 Feature:
1. **Define the Ada Spec (`src/px4_ffi_bindings.ads`)**:
   Use `pragma Import (C, ...)` to link to the corresponding C-library function.
   ```ada
   procedure Set_Flaps (Position : Float);
   pragma Import (C, Set_Flaps, "mavlink_set_flaps");
   ```
2. **LLM Integration (`src/model_manager.adb`)**:
   If you want the LLM to control this, add it to the `px4_gnc` capability list and modify the string parsing in `PX4_FFI_Bindings.Execute_GNC_Tool` to support the new parameter.
3. **Compile**:
   Run `./run.sh --build-px4` to ensure the C-headers are pulled and the firmware compiles.

## Programming ROS2 (Ada Native Publisher)

ROS2 commands are sent natively via the `ZO_ROS2_Actuator` package. 

### Wiring a New ROS2 Topic:
1. **Locate the Publisher (`src/zo_ros2_actuator.adb`)**:
   Add your new DDS/ROS2 topic string to the initialization phase.
2. **Buffer Injection**:
   If the command comes from a deterministic sensor loop (ELP3), push it directly to the native publisher.
   If the command comes from the LLM (ELP1), you must use the existing async buffer (`Zenith_Orion.ROS2_Command_Buffer`) OR the explicitly safe `ros2_actuate` LLM tool hook.
   
   Example LLM usage (Hybrid_Generate):
   ```ada
   ZO_ROS2_Actuator.Publish_Actuator_Command ("cmd_vel", Velocity_Value);
   ```

## Local Testing
Always run `./run.sh --test-build-integrity-check --verbose` before submitting a PR. This ensures that any C/C++ linking issues with MAVLink or ROS2 RoboStack are caught immediately.
