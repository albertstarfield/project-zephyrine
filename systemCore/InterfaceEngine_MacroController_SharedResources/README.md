# README: Interface Engine MacroController Shared Resources

## Overview
The `InterfaceEngine_MacroController_SharedResources` is a communication and interface module specifically architected for the MacroController. This version is designed to operate within a **Shared Resource Environment**, meaning CPU time is not federated or pre-allocated. It is intended for systems where strict time-partitioning is not a requirement and resource sharing is acceptable for supervisory tasks.

## Safety and Assurance Levels
This directory and its contents are strictly limited to **Assurance Level B and Level C** systems.

*   **Level B/C:** Fully supported. Optimized for shared resource management and general-purpose control interfacing.
*   **Level A:** Not supported. This module is not certified for mission-critical functions requiring formal proof or zero-jitter execution.

## Architectural Selection Criteria
If your system requirements dictate higher safety levels or specific hardware constraints, please select the appropriate module as follows:

### 1. Requirements for Assurance Level A
If the application requires an **RT Embedded Display** and must meet **Assurance Level A** standards, do not use this folder. Instead, use:
*   **Project:** `InterfaceEngine_uC_RT_Engine`
*   **Requirements:** This project mandates the use of **Ada 2012 + SPARK** for formal verification and safety.

### 2. Main Frame Logic
For the core management logic of the MacroController running on shared resources, refer to:
*   **Project:** `mainEngineFrame_MacroController_EngineSharedResources`

## Licensing and Proprietary Modular Extensions
### Framework License
The Zephyrine framework software provided in this directory is governed by the **HL3-BDS-BOD-LAW-MEDIA-MIL-SOC-SUP-SV** license. 

### Mission-Specific Extensions
As defined by the system architecture, any Modular Extensions developed specifically for your mission or unique hardware implementation are not automatically covered by the Zephyrine license. These components remain your dedicated proprietary license "blob" and are not considered part of the open-source or framework-licensed core.

## Contribution Guidelines
For further information regarding code standards, submission processes, and safety validation for Level B/C systems, please refer to the `CONTRIBUTING.MD` file in the project root.

---
**Path:** `systemCore/InterfaceEngine_MacroController_SharedResources`  
**Classification:** Interface Layer - Assurance Level B/C  
**Resource Model:** Non-Federated / Shared CPU Allocation
