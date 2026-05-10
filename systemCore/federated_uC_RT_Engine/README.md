# README: Federated Microcontroller Real-Time Engine (federated_uC_RT_Engine)

## Overview
The `federated_uC_RT_Engine` is a high-criticality execution environment designed for microcontrollers (uC) operating within a federated architecture. This module is engineered specifically for mission-critical applications where deterministic behavior and formal safety proofs are mandatory.

## Safety and Assurance Requirements
This component is classified for **Design Assurance Level A (DAL A)**. All development within this folder must adhere to the following technical constraints:

### 1. Language and Formal Verification
*   **Mandatory:** Implementation must be written in **Ada 2012 + SPARK**.
*   **Verification:** All code must be formally verified using the **gnatproven** toolset.
*   **Optional:** The use of **roq_proven** is supported but remains optional depending on specific mission verification requirements.

### 2. Determinism and Scheduling (WCET)
Strict adherence to **Worst-Case Execution Time (WCET)** principles is mandatory. This is required to ensure reliability in scheduling and to prevent execution overruns that could compromise the integrity of the federated system.

### 3. CPU Resource Allocation
If the uC RT Dispatcher is required to run on shared hardware resources, it **must** utilize pre-allocated partitioning of CPU time. This is formally defined as:
*   Pre-allocated timing budgets within a partition-scheduled system.
*   Fixed-time slices that guarantee the availability of resources regardless of the load on other system partitions.

## Licensing and Proprietary Drivers
### Framework License
The core framework components provided in this directory are governed by the **HL3-BDS-BOD-LAW-MEDIA-MIL-SOC-SUP-SV** license.

### Hardware-Specific Proprietary Drivers
A specific exception is made for hardware-level integration:
*   **Vendor-Specific Drivers:** Proprietary drivers provided by hardware vendors for specific silicon or peripheral implementations are **not** bound to the Zephyrine framework license. 
*   **Intellectual Property:** These drivers remain the property of the respective vendor or developer. They are treated as external dependencies or proprietary blobs to ensure compatibility with restricted hardware documentation and IP protections.

## Development Standards
Any contributions or modifications to this engine must undergo rigorous static analysis and formal proof checks to maintain DAL A certification. Developers should ensure that no non-deterministic code paths are introduced.

---
**Path:** `systemCore/federated_uC_RT_Engine`  
**Classification:** High-Criticality System Component - DAL A  
**Environment:** Federated / Partition-Scheduled RTOS
