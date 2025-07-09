## 📃 Full laundry list of Features & to-do

- [x] Operates locally on your computer, requiring an internet connection solely for web access.
- [x] Can function exclusively on CPU architectures, such as x86_64 and arm64/aarch64.
- [x] Provides compatibility with MacOS, and Linux operating systems.
- [x] Features partial GPU/MPS/FPGA (opencl)/Tensor acceleration using cuBLAS, openBLAS, clBLAST, Vulkan, and Metal.
- [x] Web access. (Requires Reimplementation from Chromium Interfacing)
- [x] Chat history functionality.
- [x] Aggressive model hot-swapping based on ELP0/ELP1 semaphores (Warning: high disk I/O controller).
- [x] Offers context memory capabilities.
- [x] Includes a Granular Toggles Mode within the settings section.
- [x] Implements Markdown Formatted Response support.
- [x] Implement Typing-like experience Response Support.
- [x] Progress bar for internal thoughts rather than leaving the user without any information on the GUI why is it Stuck.
- [x] First gemma.cpp engine model modes integration initial implementation (experimental)
- [x] Re-render old Adelaide Zephyrine Charlotte character from Anything-v3.5 to SDXL+MeinaMix_V11+Anything-v5+ESRGANx2+ESRGANx4 Mix with DDIM DPM2M++ Kerra Euler Ancestral+Euler Kerras scheduler FP16 VAE Fix
- [x] Automata Agent Mode (autoGPT agent like functionality).
- [x] Multi-LLMChild Background Queue Prompt to Experience when can't catch up with QoS time (Improving experience ~22B Parameter Adelaide Paradigm Experience) (Backbrain).
- [x] Aggressive LLMChild QoS via ELP0/ELP1 Task Prioritization.
- [x] Infinite Context Window via Global Vector Database (replaces UMA/MLCMCF design).
- [x] Facilitates access to local literature documents and resources through local files integration, excluding formatted documents, Implemented using UMA MLCMCF NLP Based matching.
- [x] New JSON interaction database that handles emotion sentiment analysis, more data, side of interaction, and multiple SessionID
- [x] Multi-submission input lifelike input rather than flip-flop wait for response type of classic interaction.
- [x] Global Vector Database for Context (divided into Interactions and File Index).
- [x] Auto Idle auto general Model Remorphing (Inserting new knowledge nonSFT and SFT Fragment Variant), specLM or specialized Language Model will NOT recieve this treatment!.
- [x] Autoexec for generated commands (unstable alpha).
- [x] Decentralized model storage via ZephyMesh P2P Distribution.
- [x] Advanced Image Generation Engine (FLUX.1 Schnell hybrid implementation).
- [x] Advanced Audio I/O (Whisper ASR + Dual TTS Engine implementation).
- [x] Mobile HTML5 UI.
- [x] VLM Capabilities via Qwen2.5 VL + ViT (replaces LLaVa-Next).
- [x] Expressive Audio Input and Output Interactive Whisper cpp
- [x] OpenAI API Implementation Bridge for Quality Testing standard of the engine model (Scientific Research Requirement using Standard Testing Method).
- [x] Auto Idle mainLLM Suspension via ELP0 Agent Relaxation "default" Idle Injection to reduce resource/heat load.
- [x] Multi-session Interactions Tab save and restore.
- [x] Agent-based tool invocation (agent.py implementation, inspired by CoPilot/Pinokio).
- [x] Screen context capture via Qwen2.5 VL (replaces original CvT plan).
- [x] StellaIcarus realtime hook response for all API ELP1
- [x] Self Learning iterative global vectoring. (There's better alternative probably, Darwin-Godel Machine by Sakana AI if you want more professional ready to use cloud version setup and being supported by multi institution)
- [x] Initial Flight Interface
- [x] I/O Swappable KV Shared Cache/State (LLM/VLM Qwen2.5) (for more serialized faster loading model swapping)

#### High Priority

- [ ] Implementing Multi-stage Decomposition Append Response ELP1.
- [ ] Implementing Self writing StellaIcarus realtime hook response for ELP1 on Ada and Python code.
- [ ] Building Zephy Spell Moral Compass. ELP0 Vector building
- [ ] I/O Swappable state for Tensor State for other Models for TTS and STT and Stable Diffusion?
- [ ] Paper Referencing Listing to prepare publication
- [ ] Redraw Zephy by Hand instead of AI generated.

#### Low Priority
- [/] Implementing Native Windows Support. (Partially implemented via Conda; requires further testing/scripting).
- [ ] Self-updating, Self-maintaning, and Self-Adapting Code.
- [ ] Multi format communication Real-time communication and collaboration (Audio, Video, Actuator)
- [ ] Adding Docker Support
- [ ] AI Blackbox Issue required Reference if its plausible be like Bing AI on the referencing system (Adapt as factual seeker) (APA7 based referencing system and formatting)
- [ ] Categorize into Classes based on the subsystem drawio for cleaner Code and easier debugging (for the base architecture)


#### Cancelled
- [ ] ~~Electron/v8 Memory Workaround~~ (Obsolete; project no longer uses Electron. Now, uses the mix of python, Golang, Ada, and nodejs for legacy soon deprecated).
- [ ] ~~Mobile static-version precompiled.~~ Will not be implemented (Cancelled)
- [ ] ~~Less-Weeb Graphic mode as default due to request of the industry~~ (Rejected; aesthetic is a core feature)
- [ ] ~~Static compilation Mobile (No AI Dynamics on Self-Modification or Special Specific Hardware Compilation), Very strict jailed version of Mobile version for iOS support.~~ (Canceled)
