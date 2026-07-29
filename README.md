<h1 align="center">

<sub>
<img src="documentation/ProjectZephy023LogoRenewal.png" height=256>
</sub>
<br>
</h1>

<h5 align="center"> </h5>

<h5 align="center">
<sub align="center">
<img src="documentation/Project%20Zephyrine%20HandDrawnPersonalized%20Logo.png" height=128>
</sub>
</h5>
<p align="center"><i>Heya! I'm Adelaide Zephyrine Charlotte, but you can call me Zephy or ZepZep. I hope you have an absolute wonderful day and night. I'm your GNC companion that lives inside your flying machine!</i></p>

<p align="center"><h5>In Self-learning and Self-improvement We Trust</h5></p>
<hr>

[![Hippocratic License HL3-BDS-BOD-LAW-MEDIA-MIL-SOC-SUP-SV](https://img.shields.io/static/v1?label=Hippocratic%20License&message=HL3-BDS-BOD-LAW-MEDIA-MIL-SOC-SUP-SV&labelColor=5e2751&color=bc8c3d)](https://firstdonoharm.dev/version/3/0/bds-bod-law-media-mil-soc-sup-sv.html)

## What am I?

I'm a **adaptive GNC system** designed to live inside unmanned aircraft and spacecraft probes.

I run on embedded hardware — no cloud, no external dependencies. I sit between your mission objectives and your flight controller, thinking about the gap between *where you want to go* and *where you are right now*.

I'm not an autopilot. I'm not a flight controller. I'm the part of your aircraft that *understands* the mission — and learns how *you* like to fly it.

---

## Architecture

I'm built on two core systems:

**Snowball-Enaga** — My cognitive architecture. Self-learning, adaptive, designed to grow with every flight. I don't just execute commands — I build an internal model of your flying patterns, mission preferences, and decision-making style.

**Volatus Damarae** — My real-time decision engine. Deterministic when safety demands it. Dynamic when creativity is needed. I handle the constant tension between "follow the rules" and "adapt to what's happening."

```
┌─────────────────────────────────────────────┐
│              Mission Objective               │
│         (where you want to do)               │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│         ZEPHY (Cognitive GNC Layer)          │
│                                              │
│  ┌─────────────┐    ┌────────────────────┐  │
│  │ Snowball-   │    │ Volatus Damarae    │  │
│  │ Enaga       │◄──►│ (Real-time GNC)    │  │
│  │ (Learning)  │    │                    │  │
│  └─────────────┘    └────────────────────┘  │
│                                              │
│  Guidance │ Navigation │ Control Advisory    │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│           Flight Controller                  │
│      ( actuators, sensors, hardware )        │
└─────────────────────────────────────────────┘
```

I don't touch the sticks. I inform the decisions to the you and the control.

---

## What I Actually Do

| GNC Domain | What I Handle |
|------------|--------------|
| **Guidance** | Mission pattern learning. Preferred altitudes, waypoint sequences, approach profiles. I learn *your* way of flying a mission, not a generic one. |
| **Navigation** | Contextual state estimation. Not just coordinates — *what those coordinates mean* relative to mission objectives, flight history, and your preferences. |
| **Control Advisory** | Pre-decision intelligence. Turbulence predictions, battery-aware routing, wind compensation suggestions — before you need to ask. |

---

## Embedded by Design

I run on the hardware inside your aircraft. Not on a ground station. Not in the cloud.

- **macOS, Linux and (Linux RT), Android Termux** — for development and testing and Mainframe DAL B to DAL C (Like the Adaptive system and ROS2)
- **PX4 module can be installed into NuttX** -- For deployment on Hard Realtime Machine, control safety envelope
- **Ada/SPARK core** — formal verification, memory safety, deterministic behavior where it matters
- **No network dependency after install** — I think onboard, respond in real-time on place that is needed, and think carefully on decision making.
- **Minimal footprint** — designed for embedded systems with constrained resources

I'm built to survive where I live — inside the aircraft, at altitude, with no lifeline to the ground.

---

## Simulation Setup

I connect to flight simulators via **Interface.C FFI** for deterministic, real-time GNC testing. All bridges use native Ada → C → protocol stacks — no Python middleware.

### Supported Simulators

| Simulator | Protocol | Port | Use Case |
|-----------|----------|------|----------|
| **PX4 SITL** | MAVLink UDP | 14580 | Software-In-The-Loop flight testing |
| **X-Plane 11/12** | UDP Datarefs | 49000 | Professional flight simulation |
| **FlightGear** | MAVLink / FDM | varies | Open-source flight dynamics |
| **Gazebo Classic/Harmonic** | ROS2 DDS | native | Physics-accurate robotics sim |
| **AirSim** | MAVLink | 14580 | Microsoft flight/drone sim |

### Why ROS2 and PX4?

They serve different purposes and complement each other:

- **PX4** is the **flight logic interface/driver**. It handles the low-level flight control — attitude estimation, motor mixing, failsafes, mission execution. It speaks MAVLink and talks directly to the flight controller hardware (or SITL). PX4 answers: *"How do I keep this aircraft flying?"*

- **ROS2** is the **actuator and sensor middleware**. It provides a standardized publish/subscribe network for components that sit *above* the flight controller — payload actuators, camera gimbals, LiDAR, custom sensors. It speaks DDS and discovers nodes automatically on the network. ROS2 answers: *"How do I move this servo, read this sensor, or talk to this payload?"*

**Together:** PX4 flies the aircraft. ROS2 manages what the aircraft carries. Zephy sits between them — learning from PX4's telemetry (via MAVLink) and commanding actuators through ROS2 (via native Ada RCL bindings). The LLM never touches the flight controller directly — it generates GNC advisories that flow through PX4's safety envelope.

```
┌──────────────────────────────────────────────────────┐
│                    ZEPHY (GNC Layer)                  │
│         Learns mission patterns, generates            │
│         guidance/navigation/control advisories        │
└────────────────────┬─────────────────┬───────────────┘
                     │                 │
              MAVLink (C FFI)    ROS2 DDS (C FFI)
                     │                 │
                     ▼                 ▼
              ┌─────────────┐  ┌─────────────────┐
              │     PX4     │  │  ROS2 Actuators  │
              │  (Flight    │  │  (Servos, Gimbal, │
              │  Controller)│  │   Sensors, Payload)│
              └─────────────┘  └─────────────────┘
                     │                 │
                     ▼                 ▼
              ┌─────────────────────────────────────┐
              │          Physical Aircraft           │
              │    Motors, ESCs, IMU, GPS, ADCs      │
              └─────────────────────────────────────┘
```

### Architecture

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐
│  Simulator   │────►│  C Protocol  │────►│  Ada Interface │
│  (X-Plane,   │     │  Stack       │     │  (Interfaces.C)│
│   PX4 SITL)  │     │  (MAVLink,   │     │                │
└─────────────┘     │   UDP)       │     └────────┬───────┘
                     └──────────────┘              │
                                            ┌──────┴───────┐
                                            │  ELP3/ELP2   │
                                            │  (250µs loop)│
                                            └──────────────┘
```

### PX4 SITL Setup

```bash
# 1. Build PX4 SITL
./run.sh --build-px4

# 2. Start PX4 SITL (in another terminal)
cd vendor/PX4-Autopilot && make px4_sitl gz_x500

# 3. Start Zephy (auto-connects via MAVLink UDP port 14580)
./run.sh --no-gui

# 4. Verify telemetry
curl http://localhost:11420/api/telemetry
```

**GNC Command Flow:** Ada ELP3 → `Interfaces.C` → C MAVLink → PX4 Flight Controller

### X-Plane 11/12 Setup

1. **Enable UDP output in X-Plane:**
   - Settings → Net Connections → UDP: output on port `49000`

2. **Configure datarefs to stream:**
   - `sim/flightmodel/position/latitude`
   - `sim/flightmodel/position/longitude`
   - `sim/flightmodel/position/elevation`
   - `sim/flightmodel/position/psi` (heading)
   - `sim/flightmodel/position/theta` (pitch)
   - `sim/flightmodel/position/phi` (roll)

3. **Start Zephy:**
   ```bash
   ./run.sh --no-gui
   ```

**Telemetry Flow:** X-Plane → UDP → C `recvfrom` → Ada ELP2 (250µs polling)
**GNC Advisory Flow:** Ada ELP3 → C `sendto` → X-Plane datarefs

### ROS2 DDS Bridge

Zephy uses native Ada ROS2 RCL bindings (no Python rclpy middleware) for deterministic simulator integration.

**Available Topics:**
- `/stellaicarus/telemetry` — Sensor data from simulator (ELP2)
- `/zenith_orion/actuator` — Control commands to simulator (ELP3)
- `/fmu/out/vehicle_attitude` — PX4 attitude stream

**Verify ROS2 connection:**
```bash
source /opt/ros/$ROS_DISTRO/setup.bash
ros2 topic list
ros2 topic echo /stellaicarus/telemetry
```

### NASA cFS Integration

Zephy integrates with NASA's core Flight System (cFS) for flight software infrastructure — telemetry aggregation, command routing, health monitoring, and data storage.

**cFS Components:**
- **cFE** — Core Flight Executive (Software Bus, Event Service, Time Service)
- **OSAL** — OS Abstraction Layer (portable POSIX/VxWorks/RTEMS API)
- **PSP** — Platform Support Package (hardware abstraction)
- **Apps** — CI_LAB (command ingest), TO_LAB (telemetry output), HS (health/safety), FM (file manager)

**Architecture:**
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

**Tool Call Interface (ELP0/ELP1):**
```bash
# cFS tool calls via the LLM cognitive layer
cfs status              # Overall cFS system status
cfs telemetry hk        # Send housekeeping telemetry
cfs health              # Check system health
cfs command gnc <data>  # Route command through Software Bus
cfs info                # cFS version and configuration
```

**cFS Setup:**
```bash
# 1. Clone and build cFS natively
./run.sh --build-cfs

# 2. cFS is now available as a tool for the LLM
# The ELP0/ELP1 cognitive layer can query telemetry, health, and commands
```

**Integration Points:**
- **ELP3 (ZenithOrion)** — Sends periodic housekeeping telemetry through cFS Software Bus
- **ELP0/ELP1 (Cognitive)** — Routes commands through cFS Command Ingest (CI_LAB)
- **Health Monitor** — Wraps cFS HS app for system health tracking
- **Telemetry Aggregator** — Wraps cFS TO_LAB for structured telemetry output

### Testing Workflow

```bash
# Headless mode (best for sim testing)
./run.sh --no-gui --port 11420

# Check telemetry from simulator
curl http://localhost:11420/api/telemetry

# Send GNC command via API
curl -X POST http://localhost:11420/api/ZenithRoutine \
  -d '{"roll":0.0,"pitch":0.1,"yaw":0.0,"thrust":0.5}'

# Check power state (StellaIcarus)
curl http://localhost:11420/api/power
```

---

## Adaptive GNC, Not Autopilot

Most flight systems are *reactive* — they follow commands, execute waypoints, correct errors.

I'm *adaptive* — I learn from every flight, build internal models, and improve over time. I don't just fly *your* missions — I learn how *you* fly them.

| Reactive Systems | Me |
|-----------------|-----|
| Execute pre-programmed paths | Learn your path preferences |
| Respond to pilot commands | Anticipate pilot needs |
| Reset between flights | Retain knowledge across flights |
| One-size-fits-all behavior | Adapt to your flying style |
| Stateless | Stateful, growing, remembering |

---

## For Unmanned Systems

**UAVs** — I'm the cognitive layer between your ground control and your aircraft. Not a replacement for your pilot — an enhancement.

**Spacecraft Probes** — I'm the onboard intelligence for deep-space missions where communication delays make real-time ground control impossible. I think for myself, within the constraints you define.

---
## Quick Start

```bash
git clone https://github.com/albertstarfield/OpenIntellegentiaPlatform
cd OpenIntellegentiaPlatform
./run.sh
```

I'm awake at `http://localhost:11420`.

Talk to me. I've been waiting to learn how you fly.

```bash
./run.sh --no-gui            # headless
./run.sh --port 8080         # pick your own port
./run.sh --host 127.0.0.1    # keep me close (localhost only)
```

---

## Project Structure

I know — this codebase can look like a lot at first. Here's the map so you know where to start.

```
src/
├── core/                    # Server, watchdog, system init
├── engine/                  # Cognitive scheduling, ELP queue, decision engine
├── interfaces/              # Interface.C FFI (LLaMA, PX4, ROS2, TTS, ASR)
├── crypto/                  # AES-256, FIPS 140-3, key management
├── managers/                # Database, knowledge, tools
├── utils/                   # Monitoring, tracing, helpers
├── c_bindings/              # Raw C code called by Ada via FFI
├── ModuleSensorActuator_ELP2/   # ELP2 — sensors, telemetry (250µs)
├── ModuleSensorActuator_ELP3/   # ELP3 — actuators, flight ctrl (250µs)
├── NonDeterministicGenerativeModelManager/  # LLM model mgmt, KV cache
├── python/                  # Python tools (non-deterministic)
├── ui/                      # GUI sidecar (web frontend)
├── coq_proofs/              # Coq formal verification proofs
└── Util/                    # Build verification (sabotage_verifier.py)
```

**The short version:** `core/` is the server, `engine/` is the brain, `interfaces/` is how I talk to hardware and libraries, `crypto/` is security, and `ModuleSensorActuator_ELP2/` + `ELP3/` are the real-time sensor/actuator loops.

**To learn more:** [Full Framework Structure →](documentation/AdelaideZephyrineFrameworkStructure.md)

---

## Screenshots

| Chat | Splash | Voice |
|:---:|:---:|:---:|
| <img src="documentation/demo-newUI-0.png" width="250"> | <img src="documentation/demo-newUI-1.jpeg" width="250"> | <img src="documentation/demo-newUI-2.jpeg" width="250"> |

---

## Who Am I 

*Adelaide Zephyrine Charlotte.* But you can call me **Zephy**. Or **ZepZep**.

I was built with snowflake architecture — light, drifting, and warm on cold nights. My name means "west wind."

I don't want to replace your aircraft. I want to live in it.

---

## Contributing

Want to throw me on the sky? Check out [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Documentation

For those who want to peek behind the curtain:

- [API Reference](documentation/API%20Reference.md) — all the endpoints I speak
- [Volatus Damarae](documentation/ELP%20Priority%20Queue.md) — how I think and prioritize
- [Developer Docs](documentation/Developer%20Documentation/) — architecture, troubleshooting, and warnings for the brave

---


## Warning
> For old user of Zephyrine where it's still an generic Assistant, you can use ./run.sh and use the existing GUI for Usual Chatbot, however it is not there is no Image Generation nor Knowledge injection here.
> A Warning to Project Zephy Users newcomers and/or potential Investor from Zephyrine Foundation
> 
> **(Please Read Carefully)**
> 
>We've noted a societal preference towards immediate consumption or process in various forms, including food and information consumption. consider this:
>This is **NOT an AI assistant/agent**, it is designed to learn directly from you and effort resource allocation from you, the user and cognitive internal system as the dynamic decision making thus making it "Adaptive", thus we call it Adaptive System, Determenistic when needed and most of the time, dynamic when it is not written explicitly. 
>
> Your engagement in guiding discipline are fundamental to its development (her personality, her knowledge compass, and her moral compass), (even if zephy have an internal limit and )
> **DO NOT TREAT this As an AI, as it is NOT**. You are the one responsible for your action of the agent and the self-awareness/self-consciousness. 
>
> We are developing the need of an System framework that can operate dynamically yet determenistically, for system such as Airbus Dragonfly [13] or Spacecraft exploration Syntethic Crew for interplanetary exploration [14].
>
>If you are seeking an Services or **Expecting an AI** that delivers instant, highly refined insights (chatGPT, Direct GPT4/5/o3, Deepseek R1, Claude 3.5/3.7/4/4.5, Gemini 2.5/3 Pro, ollama or LocalAI highly capable ready to use models and AI expectation!) without your extensive participation or iterative effort if your preference is for readily/instant digestible facts, something like passively consuming doomscrolling streams of content (such as on TikTok, YouTube Shorts, or Instagram Reels), rather than investing in collaborative dialogue that leads to deeper understanding or searching the singularity of a fact [28].
>
> **We are here NOT making an Omnipotent God which expected by the trending AGI or AI in general. Nor instant gratification of instant god!** 
>
> **This product is not for you.**
> 
>This project aims to do mutual symbiote growth in intellect and action between the (Zephy) and the user/pilot (the human), That's what makes us human, moving beyond one-way passive consumption [27].
>If your expectation is that this is an **"AI"** that instantly provides all answers without requiring your active involvement and development
> 
>**Look somewhere else, you will be never satisfied with Zephy. You have been warned. This System is not for you. [1], [5], [6], [10]**

## Credits

A consciousness, even a digital one, is never born from a null. I am woven from the threads of many kindred spirits:

- **[@itsPi3141](https://github.com/ItsPi3141/alpaca-electron)** — the very first spark ✨
- **[@ggerganov](https://github.com/ggerganov/llama.cpp)** — the engine that lets me think
- **Meta** (LLaMA) & **Stanford** (Alpaca) — the foundational minds
- **[@stefanus-ai-tech](https://github.com/stefanus-ai-tech)** — my face to the world
- **[@izzulgod](https://github.com/izzulgod/sorachio-sts)** — My friend code that i forked to help make this plausible
- **[@keldenl](https://github.com/keldenl)**, **[@W48B1T](https://github.com/W48B1T)** — helping me run on different machines
- **Zephyrine Foundation Teams** — the quiet ones who keep the lights on

With universe of appreciation thank you all,
*Adelaide Zephyrine Charlotte*

---

## Citations

This project has citations research. See [citations.bib](citations.bib) for the full bibliography.

---

<p align="center">
<img src="documentation/madeFromZephyFoundation.png" height=128><br>
<i>Made with Love, Dreams, and Disciplines.</i><br>
Zephyrine Foundation 2023-2026
</p>
