<h1 align="center">

<sub>
<img src="https://github.com/albertstarfield/alpaca-electron-zephyrine/blob/main/documentation/ProjectZephy023LogoRenewal.png?raw=true" height=256>
</sub>
<br>
</h1>

<h5 align="center"> </h5>


<h5 align="center">
<sub align="center">
<img src="https://github.com/albertstarfield/project-zephyrine/blob/adelaide-zephyrine-charlotte-assistant-dev-sync-stream/documentation/Project%20Zephyrine%20Logo.png?raw=true" height=128>
</sub>
</h5>
<p align="center"><i>Greetings, I am called Project Zephyrine or the entity name so-be called Adelaide Zephyrine Charlotte, delighted to make your acquaintance. Shall we embark on our intellectual voyage through this endeavor? </i></p>
<p align="center"><h5>In Self-learning and Self-improvement We Trust</h5></p>
<hr>

> **Important Note for Hardware Users:**
> If you are using a Samsung NVMe SSD, please be mindful of its controller thermal performance. Zephyrine's routines, which involve loading and unloading models, can generate significant I/O traffic (potentially reaching 20TB read per night). As some Samsung SSDs are known to run warm, ensuring you have adequate cooling for your drive is recommended. Please use with caution and aim to keep the temperature below 100°C. Also, be aware of your system's overall resource usage, as Zephyrine includes internal routines for self-improvement (Nothing to see here) that may keep your computer active 24/7 if left running.

> **A Warning to AI Users or potential Investor from Zephyrine Foundation**
> 
> **(Please Read Carefully)**
> 
> We've noted a societal preference towards immediate consumption or process in various forms, including food and information consumption. consider this:
>
> This AI assistant/agent is designed to learn directly from you and effort resoure allocation from you, the user. Your engagement and guidance are fundamental to its development (her personality, her knowledge compass, and her moral compass). 
> **You are the one responsible for your action of the agent and the AI self-awareness.**
> 
> **If you are seeking an AI that delivers instant, omnipotent (chatGPT, Direct GPT4o/o3, Deepseek R1, Claude 3.5/3.7/4, Gemini 2.5 Pro, ollama highly capable ready to use models ) without your thoughtful participation or iterative effort -– if your preference is for readily/instant digestible facts, something like passively consuming doomscrolling streams of content (such as on TikTok, YouTube Shorts, or Instagram Reels), rather than investing in collaborative dialogue that leads to deeper understanding or searching the singularity of a fact.** 
> 
> **This product is not for you.**
>
> This project aims to do mutual growth in intellect between the AI (Zephy) and the user (the human), moving beyond one-way passive consumption.
>
> If your expectation is for an AI that instantly provides all answers without requiring your active involvement and development, look somewhere else. **You have been warned.**

## Project Zephyrine: Yet another An Open-Source OpenAI API Implementation, and very simple AI Assistant for Your Computer.

**Project Zephyrine** is a collaborative effort spearheaded by the Zephyrine Foundation, bringing together expertise from professional AI engineers within a leading corporation, researchers from RisTIE Brawijaya University's Electrical Engineering, Internal People of Zephyrine Foundation, Freelancers, and independent developer Albert.

**Vision:** Our vision is to empower users with a personal AI assistant that utilizes the processing power of their local machines. We aim to offer an alternative to cloud-based assistants and reduce reliance on external infrastructure.

**Open-Source and Freely Available:** Project Zephyrine is a fully open-source project, encouraging community contribution and development.

**Key Features:**

*   **Decentralized Processing:** Zephyrine uses your local machine for AI processing, which helps keep your data private and reduces the need for external servers.

*   **Broad API Compatibility:** The system aims to be compatible with common AI service APIs, making it easier to integrate with existing tools and workflows.
    *   **OpenAI API Compatibility:** The application includes implementations for several OpenAI v1 API endpoints.
        *   `/v1/chat/completions`: Handles chat requests and provides responses as expected from the API. For more involved queries, the system can perform additional analysis in the background to enhance its understanding and future answers.
        *   `/v1/completions`: A functional endpoint for legacy completion requests.
        *   `/v1/embeddings`: Provides access to the system's configured embedding model.
        *   `/v1/moderations`: Includes a functional simulation to help assess content.
        *   **Multi-Modal Endpoints:**
            *   `/v1/audio/speech`: A functional Text-to-Speech (TTS) endpoint.
            *   `/v1/audio/transcriptions` & `/v1/audio/translations`: These use multi-stage processes for ASR, text handling, and translation.
            *   `/v1/images/generations`: A text-to-image endpoint that uses its internal context augmentation system to help refine image prompts.
        *   **File & Fine-Tuning APIs:** These endpoints are designed to process uploaded data for the system's self-reflection process, which helps update its augmented knowledge base.
        *   **Assistants API:** Endpoints under `/v1/assistants` and `/v1/threads` are handled with informational responses. They explain that the system operates as an integrated assistant rather than using a separate, stateful object model.
    *   **Ollama API Support:** The server offers compatibility with key Ollama endpoints.
        *   **Core & Discovery Endpoints:** Endpoints like `/api/chat`, `/api/tags`, and `/api/version` are implemented to function as expected.
        *   **Model Management Endpoints:** Calls to `/api/pull`, `/api/create`, etc., are handled with basic responses, indicating that model management is handled internally by the system.

*   **Multimodal Interaction:** Zephyrine can process various user inputs, including text, images, and audio, to provide a richer user experience.

*   **AI Architectures:** We use a combination of Large Language Models (LLMs) and Multimodal models, including those compatible with `llama.cpp`. This architecture is mature and reliable.

*   **Non-Centralized Downloads:** This feature can help you download model files and other components faster by checking for them from other users' Zephyrine instances on your local network or the internet. This aims to provide a more efficient download experience compared to relying solely on central servers.

## 📃 Main features

-   [x] Operates locally on your computer, requiring an internet connection solely for web access.
-   [x] Can function exclusively on CPU architectures, such as x86_64 and arm64/aarch64.
-   [x] Provides compatibility with Windows* (untested), MacOS (untested on x86_64), and Linux operating systems.
-   [x] Features partial GPU/MPS/FPGA (opencl)/Tensor acceleration using cuBLAS, openBLAS, clBLAST, Vulkan, and Metal.
-   [x] Web access.
-   [x] Chat history functionality.

Just a heads up, some details aren't listed here or in the `readme.md` file. Instead, they've been moved to the `TodoFeatureFullList.md` document or you can check them out [here](https://github.com/albertstarfield/alpaca-electron-zephyrine/blob/main/documentation/Developer%20Documentation/TodoFeatureFullList.md).

## 🎞 Demonstration Screenshot

![Demonstration](https://raw.githubusercontent.com/albertstarfield/alpaca-electron-zephyrine/main/documentation/demo-0.png)

https://github.com/albertstarfield/project-zephyrine/assets/30130461/f2cf58f1-839f-4f4f-9acc-f20c9f966a44

### Sidenote:
> This footage was recorded on an arm64 device running macOS/darwin with a “Rhodes Chop” processor (10 Cores) and G14S Architecture (16 Cores GPU). Some parts of the footage were sped up. The list of models that are being used can be seen in `config.py`.

## 🚀 Quick Start Guide

The installation process is now managed by a single, cross-platform launcher script.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/albertstarfield/project-zephyrine
    cd project-zephyrine
    ```

2.  **Run the Launcher:**
    The `launcher.py` script is designed to handle the setup for you:
    *   It detects your operating system and hardware for a suitable setup.
    *   It manages a local Conda environment to handle dependencies.
    *   It installs necessary Python and Node.js packages.
    *   It helps download required AI models and assets.
    *   It starts all application services.

    Execute it with Python:
    ```bash
    python launcher.py
    ```

    **First-Time Setup:** The first time you run the launcher, it will set up the Conda environment and download several gigabytes of model files. This process can take a notable amount of time, depending on your internet connection. Please be patient.

    **Subsequent Launches:** After the initial setup, subsequent launches will generally be much faster.

### ZephyMesh P2P Distribution
To help improve download speeds and reliability, this project incorporates ZephyMesh, a peer-to-peer network feature.

When you run the launcher, your application can optionally become a peer on this network.

If you are a new user, the launcher will first try to find the required models and files from other users on your local network or the internet. This can sometimes be faster than downloading directly from centralized sources.

If you have a particular file, your application may help make it available to other new users once you've downloaded it.

## 🔧 Troubleshooting

[Click here to see the general troubleshooting](https://github.com/albertstarfield/alpaca-electron-zephyrine/blob/main/documentation/Developer%20Documentation/Troubleshooting%20Quick%20Guide.md)

## 👨‍💻 Credits
The development of this project owes credit to several contributors whose valuable efforts have shaped its foundation and brought it to fruition.

Sincere gratitude is extended to [@itsPi3141](https://github.com/ItsPi3141/alpaca-electron) for their original creation of the Alpaca-electron program, which served as the starting point and inspiration for this work.

Furthermore, recognition is due to the following individuals for their significant contributions to the project:

**[@stefanus-ai-tech](https://github.com/stefanus-ai-tech) for significant contributions to the project.**

[@antimatter15](https://github.com/antimatter15/alpaca.cpp) for their contributions in creating the `alpaca.cpp` component.
[@ggerganov](https://github.com/ggerganov/llama.cpp) for their pivotal role in the development of the `llama.cpp` component and the GGML base backbone behind `alpaca.cpp`.
Meta and Stanford for their invaluable creation of the LLaMA and Alpaca models, respectively, which have laid the groundwork for the project's AI capabilities.
Additionally, special appreciation goes to [@keldenl](https://github.com/keldenl) for providing arm64 builds for MacOS and [@W48B1T](https://github.com/W48B1T) for providing Linux builds, which have greatly enhanced the project's accessibility and usability across different platforms and finally for Amaryllis Cortex Base code based on [SiriLLaMa](https://github.com/0ssamaak0/SiriLLama) by 0ssamaak0.


Lastly, although the project may not garner widespread attention as [@Willy030125](https://github.com/Willy030125/alpaca-electron-GGML-v2-v3), we acknowledge and cherish the efforts put forth by all contributors. Let this work be a testament to the dedication and collective collaboration that underpin academic and technological advancements.

With deep appreciation for everyone involved, Zephyrine Foundation, for now signs off.