# cFS Manual Building for Zephyrine

**NASA core Flight System (cFS) Integration Guide**

---

## What is cFS?

NASA's core Flight System (cFS) is a generic flight software architecture framework used on flagship spacecraft, human spacecraft, cubesats, and Raspberry Pi. It provides:

- **Software Bus** — Publish/subscribe message passing between flight apps
- **Event System** — Telemetry and event reporting to ground stations
- **Executive Service** — Application lifecycle management
- **Time Service** — Spacecraft time management
- **Table Services** — Runtime configuration tables
- **File Manager** — Non-volatile storage management

cFS is the **flight software infrastructure** — it handles telemetry aggregation, command routing, health monitoring, and data storage. PX4 handles the **flight control** (GNC, attitude). They're complementary.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                    ZEPHY (GNC Layer)                  │
│         Cognitive ELP0/ELP1 → Tool Bridge             │
│         Deterministic ELP2/ELP3 → cFS Telemetry       │
└────────────────────┬─────────────────┬───────────────┘
                     │                 │
              Ada FFI (Interfaces.C)   │
                     │                 │
                     ▼                 ▼
              ┌─────────────┐  ┌─────────────────┐
              │  cFS cFE    │  │     PX4         │
              │  (Software  │  │  (Flight        │
              │   Bus, TLM) │  │   Controller)   │
              └─────────────┘  └─────────────────┘
                     │
                     ▼
              ┌─────────────────────────────────────┐
              │         cFS Apps                     │
              │  CI_LAB │ TO_LAB │ HS │ FM │ DS     │
              └─────────────────────────────────────┘
```

### Component Responsibilities

| Component | Role | Protocol |
|-----------|------|----------|
| **PX4** | Flight control (GNC, attitude, motor mixing) | MAVLink UDP |
| **cFS** | Flight software (telemetry, commands, health, data) | Software Bus (pub/sub) |
| **ROS2** | Actuator/sensor middleware (servos, gimbal, LiDAR) | DDS |
| **Zephy** | Cognitive GNC layer (learning, adaptation) | Ada FFI |

### RTOS Compatibility

cFS is RTOS-agnostic through its OS Abstraction Layer (OSAL). It runs on any OS that provides threading, semaphores, and file I/O.

| RTOS | cFS Status | Formal Verification | Use Case |
|------|-----------|---------------------|----------|
| **VxWorks** | Production (NASA) | DO-178C certified | Flight-proven spacecraft |
| **RTEMS** | Production | No | Real-time embedded |
| **FreeRTOS** | Available | No | Low-resource MCUs |
| **Linux** | Native | No | Ground simulation, SITL |
| **Zephyr** | Available | No | Modern embedded, PX4 native |
| **seL4** | Needs OSAL port | Yes (mathematical proof) | High-assurance flight |
| **NuttX** | Available | No | POSIX-like, PX4 current |

**seL4** is the most promising for formal verification — it's a mathematically proven correct microkernel. cFS + seL4 would give a formally verified OS + formally verified flight software stack. Writing a cFS OSAL for seL4 is required (threading, semaphores, file I/O).

**Path to seL4:**
1. Write cFS OSAL for seL4
2. Port cFS apps to seL4 environment
3. Validate with Ada/SPARK core via POSIX API

---

## Quick Start

### 1. Clone and Build cFS

```bash
# Clone cFS + initialize submodules + build natively
./run.sh --build-cfs
```

This runs:
1. `git clone https://github.com/nasa/cFS.git vendor/cFS`
2. `git submodule update --init --recursive` (25+ submodules)
3. `cmake -DSIMULATION=native build && make -j$(nproc)`

Build output: `vendor/cFS/build/core-cpu1`

### 2. Verify cFS Installation

```bash
# Check cFS binary exists
ls -la vendor/cFS/build/core-cpu1

# Run cFS manually (optional)
cd vendor/cFS/build && ./core-cpu1
```

### 3. Use cFS from Zephy

```bash
# Via tool calls (ELP0/ELP1 cognitive layer)
cfs status              # Overall system status
cfs telemetry hk        # Send housekeeping telemetry
cfs health              # Check system health
cfs command gnc <data>  # Route command through Software Bus
cfs info                # Version and configuration
```

---

## Manual Build Steps

If you need to build cFS manually (not via `run.sh`):

```bash
# 1. Navigate to cFS directory
cd vendor/cFS

# 2. Initialize submodules (if not already done)
git submodule update --init --recursive

# 3. Create build directory
mkdir -p build && cd build

# 4. Configure with CMake (native simulation)
cmake -DSIMULATION=native ..

# 5. Build all targets
make -j$(nproc)

# 6. Verify build
ls -la core-cpu1  # Should exist
```

### Build Options

| Option | Description |
|--------|-------------|
| `-DSIMULATION=native` | Build for native Linux/macOS execution |
| `-DCMAKE_BUILD_TYPE=Debug` | Debug build with symbols |
| `-DOMIT_DEPRECATED=ON` | Exclude deprecated APIs |

---

## cFS Apps Included

The following cFS apps are built and available:

| App | Purpose | Tool Call |
|-----|---------|-----------|
| **CI_LAB** | Command Ingest — receives commands from ground | `cfs command` |
| **TO_LAB** | Telemetry Output — sends telemetry to ground | `cfs telemetry` |
| **SCH_LAB** | Scheduler — periodic task execution | Internal |
| **HS** | Health & Safety — watchdog, health monitoring | `cfs health` |
| **FM** | File Manager — non-volatile storage | Internal |
| **DS** | Data Storage — data packet routing | Internal |
| **HK** | Housekeeping — system status telemetry | `cfs telemetry hk` |
| **LC** | Limit Checker — sensor limit monitoring | Internal |
| **SC** | Stored Command — command uplink/execution | Internal |
| **MD** | Memory Dwell — memory read/write monitoring | Internal |
| **MM** | Memory Manager — memory access commands | Internal |
| **CS** | Checksum — memory/software verification | Internal |
| **CF** | CFDP — file transfer protocol | Internal |
| **SBN** | Software Bus Network — inter-CPU routing | Internal |

---

## Ada Integration

### FFI Bindings

cFS is accessed from Ada via `Interfaces.C` FFI bindings:

```ada
-- src/interfaces/cfe_ffi_bindings.ads
package CFE_FFI_Bindings is
   -- Pipe Management
   function CFE_SB_CreatePipe (...) return CFE_Status_t;
   function CFE_SB_DeletePipe (...) return CFE_Status_t;

   -- Subscription
   function CFE_SB_Subscribe (...) return CFE_Status_t;
   function CFE_SB_Unsubscribe (...) return CFE_Status_t;

   -- Message Send/Receive
   function CFE_SB_TransmitMsg (...) return CFE_Status_t;
   procedure CFE_SB_TimeStampMsg (...);
   procedure CFE_SB_SetUserDataLength (...);

   -- Event Service
   function CFE_EVS_SendEvent (...) return CFE_Status_t;
end CFE_FFI_Bindings;
```

### Tool Bridge (ELP0/ELP1)

The LLM cognitive layer accesses cFS through the tool bridge:

```ada
-- src/interfaces/cfs_tool_bridge.ads
package CFS_Tool_Bridge is
   function Execute_CFS_Tool (Params : String) return Tool_Result;
end CFS_Tool_Bridge;
```

Tool call routing in `tool_manager.adb`:
```ada
elsif Name = "cfs" or else Name = "cfe" or else Name = "flight_software" then
   return CFS_Tool_Bridge.Execute_CFS_Tool (Params);
```

### Integration Points

| Layer | cFS Integration | File |
|-------|----------------|------|
| **ELP3 (ZenithOrion)** | Sends periodic housekeeping telemetry | `zenith_orion.adb` |
| **ELP0/ELP1 (Cognitive)** | Routes commands through Software Bus | `cfs_tool_bridge.adb` |
| **Health Monitor** | Wraps HS app for system health | `cfs_health_monitor.adb` |
| **Telemetry Aggregator** | Wraps TO_LAB for structured output | `cfs_telemetry.adb` |
| **Command Router** | Wraps CI_LAB for command dispatch | `cfs_command_router.adb` |

---

## Tool Call Reference

### `cfs status`
Returns overall cFS system health, command count, and telemetry status.

```
[cFS Status]
  Health:    Healthy
  Commands:  42 routed
  Telemetry: Active
  SW Bus:    Initialized
```

### `cfs telemetry <type>`
Send telemetry through the Software Bus.

| Type | Description |
|------|-------------|
| `hk` / `housekeeping` | CPU, memory, uptime telemetry |
| `sensor` | Sensor reading telemetry |
| `attitude` / `att` | Roll/pitch/yaw telemetry |

### `cfs health [app_name]`
Check health of a specific app or overall system.

```bash
cfs health              # System health
cfs health CI_LAB       # Check specific app
```

### `cfs command <type> <data>`
Route a command through the Software Bus.

| Type | Description |
|------|-------------|
| `gnc` | GNC command (guidance, navigation, control) |
| `telemetry` / `tlm` | Telemetry configuration |
| `health` | Health/safety command |
| `config` | Configuration command |
| `custom` | Custom command (default) |

### `cfs info`
Returns cFS version, components, and integration details.

---

## File Structure

```
vendor/cFS/
├── cfe/                    # Core Flight Executive (submodule)
│   └── modules/
│       ├── core_api/       # Public API headers
│       │   └── fsw/inc/    # cfe_sb.h, cfe_evs.h, cfe_es.h, etc.
│       ├── sb/             # Software Bus module
│       ├── evs/            # Event Service module
│       ├── es/             # Executive Service module
│       └── msg/            # Message module
├── osal/                   # OS Abstraction Layer (submodule)
├── psp/                    # Platform Support Package (submodule)
├── apps/                   # Flight applications
│   ├── ci_lab/             # Command Ingest (Lab)
│   ├── to_lab/             # Telemetry Output (Lab)
│   ├── sch_lab/            # Scheduler (Lab)
│   ├── hs/                 # Health & Safety
│   ├── fm/                 # File Manager
│   ├── ds/                 # Data Storage
│   └── ...
├── libs/                   # Libraries
│   └── sample_lib/
├── tools/                  # Build tools
│   ├── cFS-GroundSystem/   # Ground system
│   └── ...
├── sample_defs/            # Sample build configurations
│   ├── targets.cmake       # Target board definitions
│   └── ...
└── build/                  # Build output (created by cmake)
    └── core-cpu1           # Native build binary
```

---

## GPR Configuration

The Ada GPR file (`adelaide_zephyrine_system.gpr`) includes:

### C Compiler Switches (include paths)
```
-I vendor/cFS/cfe/modules/core_api/fsw/inc
-I vendor/cFS/cfe/modules/core_api/config
-I vendor/cFS/cfe/modules/sb/fsw/inc
-I vendor/cFS/cfe/modules/es/fsw/inc
-I vendor/cFS/cfe/modules/evs/fsw/inc
-I vendor/cFS/cfe/modules/msg/fsw/inc
-I vendor/cFS/osal/src/os/inc
```

### Linker Flags
```
vendor/cFS/build/core-api/libcore-api.a
vendor/cFS/build/cfe-core/libcfe-core.a
```

---

## Troubleshooting

### Build fails with "cfe/cfe.h not found"
```bash
# Reinitialize submodules
cd vendor/cFS
git submodule update --init --recursive
```

### Linker errors "undefined symbol CFE_SB_*"
```bash
# Rebuild cFS
cd vendor/cFS/build
cmake -DSIMULATION=native .. && make -j$(nproc)
```

### cFS binary not found
```bash
# Check build output
ls -la vendor/cFS/build/core-cpu1
# If missing, rebuild:
./run.sh --build-cfs
```

### Submodules not initialized
```bash
cd vendor/cFS
git submodule status  # Check which submodules are initialized
git submodule update --init --recursive  # Initialize all
```

---

## References

- [cFS User's Guide](https://github.com/nasa/cFS/blob/gh-pages/cfe-usersguide.pdf)
- [OSAL User's Guide](https://github.com/nasa/cFS/blob/gh-pages/osal-apiguide.pdf)
- [cFE App Developer's Guide](https://github.com/nasa/cFE/blob/main/docs/cFE%20Application%20Developers%20Guide.md)
- [cFS Website](https://cfs.gsfc.nasa.gov)
- [cFS GitHub](https://github.com/nasa/cFS)
