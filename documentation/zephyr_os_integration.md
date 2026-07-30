# Zephyr RTOS Integration

## What is Zephyr?

Zephyr is a scalable real-time operating system (RTOS) for resource-constrained devices — microcontrollers with as little as 8KB RAM. Maintained by the Linux Foundation, it provides threading, memory management, device drivers, and networking for embedded systems.

## Why Zephyr for Adelaide?

| Need | Zephyr provides |
|------|-----------------|
| Deployment target | Builds for ARM Cortex-M, RISC-V, x86, and 200+ boards |
| PX4 RTOS | PX4 natively runs on Zephyr (replaces NuttX) |
| Simulation | `native_posix` board runs full stack on macOS/Linux |
| Drivers | IMU, GPS, UART, SPI, I2C, CAN — all built-in |
| Networking | MAVLink, DDS, TCP/UDP — native net stack |
| Threading | `k_thread` API with deterministic priority scheduling |
| Memory | Slab pools, heap, memory protection units |

## Architecture

```
┌─────────────────────────────────────────────┐
│              Ada/SPARK Core                  │
│         (GNC, Cognitive, cFS)               │
└──────────────────┬──────────────────────────┘
                   │ POSIX API / Interfaces.C FFI
                   ▼
┌─────────────────────────────────────────────┐
│            Zephyr RTOS                       │
│  Threading │ Drivers │ Networking │ Memory   │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│         Hardware (ARM Cortex-M, RISC-V)     │
│         or native_posix (simulation)        │
└─────────────────────────────────────────────┘
```

## Development vs Deployment

| Mode | Board | Use |
|------|-------|-----|
| **Development** | `native_posix` | Run on macOS/Linux, no hardware needed |
| **SITL** | `native_posix` + PX4 | PX4 Software-In-The-Loop on Zephyr |
| **Deployment** | `stm32f4_disco`, `nrf52840dk`, etc. | Cross-compile for real hardware |

## Setup

### Prerequisites

- Zephyr SDK (installed via `west`)
- ARM GCC toolchain (for cross-compilation)
- CMake 3.20+
- Python 3.10+

### Quick Start (native_posix)

```bash
# 1. Build Zephyr with native_posix board
./run.sh --build-zephyr

# 2. Run (simulates full RTOS on your Mac)
./run.sh --no-gui

# 3. Verify
curl http://localhost:11420/api/telemetry
```

### Cross-compile for Hardware

```bash
# 1. Build for specific board
./run.sh --build-zephyr --board stm32f4_disco

# 2. Flash via OpenOCD/J-Link
west flash --runner openocd
```

## Board Support

Zephyr supports 200+ boards. Common targets for Adelaide:

| Board | MCU | RAM | Flash | Use Case |
|-------|-----|-----|-------|----------|
| `native_posix` | Host CPU | Host | Host | Development/simulation |
| `stm32f4_disco` | STM32F407 | 192KB | 1MB | Flight controller dev |
| `nrf52840dk` | nRF52840 | 256KB | 1MB | Low-power UAV |
| `mimxrt1060_evk` | i.MX RT1060 | 1MB | 8MB | High-performance GNC |

## Ada Integration

Ada/SPARK code runs on Zephyr via:

1. **POSIX API** — Zephyr's POSIX compatibility layer (`CONFIG_POSIX_API=y`)
2. **Direct FFI** — `Interfaces.C` calls Zephyr's C API (`k_thread_create`, `k_sem_give`, etc.)
3. **Alire crates** — Wrap Zephyr APIs in Ada packages

```ada
-- Example: Creating a Zephyr thread from Ada
procedure Spawn_GNC_Thread is
   procedure C_K_Thread_Create
     (Stack : access C_Stack;
      Entry_Point : System.Address;
      Priority : C.int;
      Arg : System.Address)
     with Import, Convention => C, External_Name => "k_thread_create";
begin
   C_K_Thread_Create
     (GNC_Stack'Access,
      GNC_Entry'Address,
      Priority => 5,
      Arg => System.Null_Address);
end Spawn_GNC_Thread;
```

## Kconfig Configuration

Zephyr uses Kconfig for build-time configuration. Key options for Adelaide:

```ini
# Adelaide Zephyr config
CONFIG_POSIX_API=y                    # POSIX compatibility for Ada FFI
CONFIG_NETWORKING=y                   # MAVLink/cFS networking
CONFIG_NET_IPV4=y                     # IPv4 support
CONFIG_NET_UDP=y                      # UDP for MAVLink
CONFIG_SERIAL=y                       # UART for sensors
CONFIG_I2C=y                          # I2C bus (IMU, barometer)
CONFIG_SPI=y                          # SPI bus (fast sensors)
CONFIG_GPIO=y                         # GPIO control
CONFIG_HEAP_MEM_POOL_SIZE=8192        # 8KB heap
CONFIG_MAIN_STACK_SIZE=2048           # 2KB main stack
CONFIG_THREAD_STACK_SIZE=2048         # 2KB default thread stack
CONFIG_SYSTEM_WORKQUEUE_STACK_SIZE=2048
```

## Networking Stack

Zephyr's networking supports MAVLink, cFS, and ROS2 DDS:

| Protocol | Zephyr support | Port |
|----------|----------------|------|
| MAVLink UDP | `CONFIG_NET_UDP=y` | 14580 |
| cFS Software Bus | Custom app over UDP/TCP | 5005 |
| ROS2 DDS | Micro-XRCE-DDS agent | 8888 |

## Threading Model

Zephyr uses preemptive priority-based threading:

```
Priority 0 (highest):  Interrupt handlers
Priority 1:            ELP3 flight control (250µs loop)
Priority 2:            ELP2 sensor polling (250µs loop)
Priority 3:            cFS Software Bus routing
Priority 4:            GNC advisory generation
Priority 5:            Health monitoring, telemetry logging
Priority 6 (lowest):   Background tasks, idle
```

## RTOS Comparison

| RTOS | Formal Verification | cFS Support | Use Case |
|------|---------------------|-------------|----------|
| **Zephyr** | No | Available | Development, simulation, deployment |
| **seL4** | Yes (mathematical proof) | Needs OSAL port | High-assurance flight |
| **VxWorks** | DO-178C certified | Production (NASA) | Flight-proven spacecraft |
| **RTEMS** | No | Production | Real-time embedded |
| **FreeRTOS** | No | Available | Low-resource MCUs |
| **NuttX** | No | Available | POSIX-like, PX4 current |

### seL4 — Future Option

seL4 is the world's first formally verified microkernel — mathematically proven correct and secure. cFS on seL4 would give a formally verified OS + formally verified flight software stack.

seL4 has no native OSAL for cFS — writing one would be required. This is a significant but worthwhile effort for high-assurance missions.

**Path to seL4:**
1. Write cFS OSAL for seL4 (threading, semaphores, file I/O)
2. Port cFS apps to seL4 environment
3. Validate with Ada/SPARK core via POSIX API

## References

- [Zephyr Documentation](https://docs.zephyrproject.org/latest/)
- [Zephyr Boards](https://docs.zephyrproject.org/latest/boards/index.html)
- [Zephyr POSIX API](https://docs.zephyrproject.org/latest/services/portability/posix.html)
- [PX4 on Zephyr](https://docs.px4.io/main/en/dev_setup/dev_env_linux.html)
- [native_posix Board](https://docs.zephyrproject.org/latest/boards/posix/native_posix/doc/index.html)
- [seL4 Documentation](https://docs.sel4.systems/)
- [seL4 Proofs](https://docs.sel4.systems/projects/sel4/api-doc.html)
