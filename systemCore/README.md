# README: System Core Architecture (systemCore)

## Overview
The `systemCore` directory contains the foundational architecture for the Zephyrine framework. The system is strictly partitioned based on Design Assurance Levels (DAL) to ensure that mission-critical, hard real-time operations are fully isolated from general-purpose, shared-resource computing tasks.

## Terminology
Before navigating the architecture, note the distinction between the two primary compute environments:
*   **MacroController:** A full-blown computer system utilized for high-level management, complex non-critical logic, and heavy UI/UX operations.
*   **uC (MicroController):** A smaller, dedicated controller. This can be either a physical silicon chip or a strict Virtual Machine (VM) instance dedicated to real-time, deterministic control.

## Architecture Tree and DAL Mapping

```text
systemCore/
│
├── DAL A (Assurance Level A) - High Criticality, Strict WCET, Pre-allocated CPU Time
│   │
│   ├── federated_uC_RT_Engine/
│   │   └── Core real-time dispatcher for the uC. Requires Ada 2012 + SPARK.
│   │
│   └── InterfaceEngine_uC_RT_Engine/
│       └── Dedicated high-integrity interface and RT embedded display logic.
│
└── DAL B/C (Assurance Level B/C) - Shared Resources, Dynamic Scheduling, Multi-language
    │
    ├── mainEngineFrame_MacroController_EngineSharedResources/
    │   └── Core mainframe logic for the MacroController (Python/C++/Ada).
    │
    └── InterfaceEngine_MacroController_SharedResources/
        └── Supervisory interface and general UI/UX communication layer.
```

## Modular Extensions and Hooks Implementation
The framework is designed to be modular. Mission-specific logic, proprietary quests, and hardware interactions should be implemented in their designated environments:

### 1. federated_uC_RT_Engine (DAL A)
*   **Implementation Focus:** Hardware-to-software interfacing.
*   **Details:** Use this for low-level, critical hardware interactions that require formal proofs and guaranteed execution times.

### 2. InterfaceEngine_uC_RT_Engine (DAL A)
*   **Implementation Focus:** RT Embedded Displays.
*   **Details:** Use this for specific mission-critical embedded display rendering and logic that are *not* inherently provided by the base Zephyrine framework.

### 3. mainEngineFrame_MacroController_EngineSharedResources (DAL B/C)
*   **Implementation Focus:** StellaIcarus Daemons and API Hooks.
*   **Details:** Implement mission-specific or quest-specific operations, bridges, and ML model inferencing here. You can utilize StellaIcarus Ada daemons for reliable background tasks, and StellaIcarus Python/C++ deterministic hooks for specific hardware communications. These can be proprietary extensions.

### 4. InterfaceEngine_MacroController_SharedResources (DAL B/C)
*   **Implementation Focus:** UI/UX to Hardware/Software bridging.
*   **Details:** Implement standard, non-critical graphical interfaces and user experience layers that interact with the system logic.

## Virtualization and Hypervisor Constraints (DAL A)
For DAL A modules (`federated_uC_RT_Engine` and `InterfaceEngine_uC_RT_Engine`), you are permitted to use Virtual Machine Integrated Modular Frameworks (VM IMF) or Hypervisors, such as:
*   **VxWorks 653**
*   **Gunyah (Snapdragon)**
*   **Any Other Type-1 Hypervisor that able to do this using Hardware level timing Preallocation**
*   **ONLY FOR TESTING (Putting RT Priority kernel process on a recycled hardware)**

**Strict Constraint:** If a VM or hypervisor is utilized for DAL A components, the CPU time **must** be strictly pre-allocated, restricted, and run on a constant, real-time timing budget. Federated or partition-scheduled execution is mandatory to satisfy Worst-Case Execution Time (WCET) reliability.

## Licensing and Proprietary Software Sidenote
### Zephyrine Framework License
The core base software provided within this repository is licensed under the **HL3-BDS-BOD-LAW-MEDIA-MIL-SOC-SUP-SV** license.

### Modular Extensions and Proprietary Exceptions
*   **Mission-Specific Modules:** Any modular extensions, StellaIcarus hooks, or custom daemons created specifically for your mission or operations remain under your proprietary license. They are treated as proprietary blobs and do not automatically inherit the framework license.
*   **Hardware Drivers:** Hardware-specific proprietary drivers bound to the vendor (especially in the `federated_uC_RT_Engine`) remain the intellectual property of the hardware vendor and are entirely exempt from the Zephyrine framework license.

---
**Path:** `systemCore/`  
**Classification:** Root Architecture Directory
