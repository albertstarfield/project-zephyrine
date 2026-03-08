# README: Main Engine Frame - MacroController (Shared Resources)

## Overview
The `mainEngineFrame_MacroController_EngineSharedResources` serves as the primary execution framework for the MacroController when operating within a Shared Resources environment. This architectural choice implies that CPU time is not federated or pre-allocated; instead, the system relies on dynamic scheduling. 

This engine is specifically engineered to meet **Assurance Level B and C** requirements. It utilizes a multi-language architecture, integrating the safety and formal verification of Ada/SPARK with the flexibility of C++ and compiled Python.

## System Architecture and Components
The framework is divided into specific functional areas to manage core logic, deterministic API hooks, and background processes.

### 1. Core Mainframe Management
*   **Entry Point:** `AdelaideAlbertCortex.py`
*   **Description:** This is the central management program for the core mainframe. It orchestrates the high-level logic and coordinates between the various sub-modules.

### 2. Deterministic API Hooks
*   **Directory:** `./StellaIcarus`
*   **Description:** This module handles deterministic requests for the API. It serves as the primary interface layer for external calls, ensuring that requests are processed within the constraints of the shared resource environment.

### 3. Background Daemons (Ada/SPARK)
*   **Directory:** `./StellaIcarus/[YourAlireProjectFile]`
*   **Description:** Background processes are implemented using Ada/SPARK within an Alire project structure. This ensures high reliability for daemon-level tasks.
*   **Developer Note:** Ensure that mission-specific proprietary code or Alire project configurations are added to your `.gitignore` to prevent the accidental commitment of sensitive data.

## Interfacing and ML Integration
This framework allows for the implementation of bridges or Machine Learning (ML) models for inferencing. By utilizing the **StellaIcarus** daemons and hooks, developers can create an interface between this Level B/C shared resource environment and the **Assurance Level A federated controller**. This allows for sophisticated data processing and decision-making without compromising the integrity of the Level A core.

## Licensing and Proprietary Extensions
### Zephyrine Framework License
The core framework is licensed under the **HL3-BDS-BOD-LAW-MEDIA-MIL-SOC-SUP-SV** license. This license applies strictly to the underlying Zephyrine framework software.

### Modular Extensions
Any modular extensions developed specifically for a mission or unique use case are not automatically subject to the Zephyrine license. These extensions remain the proprietary property of the developer/organization as a "proprietary blob," allowing for the protection of mission-specific intellectual property.

---
**Path:** `systemCore/mainEngineFrame_MacroController_EngineSharedResources`  
**Assurance:** Level B/C  
**Resource Model:** Non-Federated / Shared CPU Time
