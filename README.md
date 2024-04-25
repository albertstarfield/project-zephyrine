<h1 align="center">

<sub>
<img src="https://github.com/albertstarfield/alpaca-electron-zephyrine/blob/main/documentation/ProjectZephy023LogoRenewal.png?raw=true" height=256>
</sub>
<br>
</h1>

<h5 align="center"> </h5>

<p align="center">
  <a href="https://nodejs.org">
    <img src="https://img.shields.io/badge/node.js-6DA55F?style=for-the-badge&logo=node.js&logoColor=white">
  </a>
  <a href="https://www.electronjs.org/">
    <img src="https://img.shields.io/badge/Electron-191970?style=for-the-badge&logo=Electron&logoColor=white">
  </a>
</p>
<h5 align="center">
<sub align="center">
<img src="https://github.com/albertstarfield/alpaca-electron-zephyrine/blob/main/usr/icon/png/512x512.png?raw=true" height=128>
</sub>
</h5>
<p align="center"><i>Greetings, I am called Project Zephyrine or the entity name so-be called Adelaide Zephyrine Charlotte, delighted to make your acquaintance. Shall we embark on our intellectual voyage through this endeavor? </i></p>

<hr>

> **Important Note:**
> The software is undergoing a paradigm shift, focusing on a plug-and-play framework similar to Copilot, primarily offline. It aims to avoid direct competition with LMStudio and OLlama GUI that is more successful, focusing on delineated entities with a Mix of Experts (MoE) interface.

## Project Zephyrine: An Open-Source, Decentralized, non-Nvidia exclusive, non-proprietary, AI Assistant for Your Desktop.

**Project Zephyrine** is a collaborative effort spearheaded by the Zephyrine Foundation, bringing together expertise from professional AI engineers within a leading corporation, researchers from RisTIE Brawijaya University's Electrical Engineering department, and independent developer Albert. 

**Vision:** Our vision is to empower users with a truly personal AI assistant that leverages the full processing power of their local machines. We aim to move beyond cloud-based, web-centric assistants and liberate users from dependence on external infrastructure.

**Open-Source and Freely Available:** Project Zephyrine is a fully open-source project, fostering community contribution and development, just like the popular 3D creation software Blender.

**Key Features:**

* **Decentralized Processing:** Zephyrine utilizes the user's local machine for AI processing, eliminating the need for external servers and fostering data privacy.
* **Multimodel Async + Multimodal Interaction:** Zephyrine analyzes various user inputs, including ~~vision~~, ~~sound~~, text, and emotion, enabling a richer and more intuitive user experience.
* **Advanced AI Architectures:** We leverage a combination of Large Language Models (LLMs) and Multimodal models, including cutting-edge options like LLaMa, Whisper, mpt, Falcon, and more (licensing subject to individual model terms).
* **Asynchronous Processing:** Queries are processed asynchronously, allowing utilization of large models (100B+ parameters) even on local hardware, without the need for expensive data center setups (e.g., NVIDIA Exclusive Datacenter GPUs).

## 📃 Main features

- [x] Operates locally on your computer, requiring an internet connection solely for web access.
- [x] Can function exclusively on CPU architectures, such as x86_64 and arm64/aarch64.
- [x] Provides compatibility with Windows* (untested), MacOS (untested on x86_64), and Linux operating systems.
- [x] Features partial GPU/MPS/FPGA (opencl)/Tensor acceleration using cuBLAS, openBLAS, clBLAST, Vulkan, and Metal.
- [x] Web access.
- [x] Chat history functionality.

Just a heads up, some details aren't listed here or in the `readme.md` file. Instead, they've been moved to the `TodoFeatureFullList.md` document or you can check them out [here](https://github.com/albertstarfield/alpaca-electron-zephyrine/blob/main/documentation/Developer%20Documentation/TodoFeatureFullList.md).




## 🎞 Demonstration Screenshot

![Demonstration](https://raw.githubusercontent.com/albertstarfield/alpaca-electron-zephyrine/main/documentation/demo-0.png)
![Demonstration_video](https://raw.githubusercontent.com/albertstarfield/alpaca-electron-zephyrine/main/documentation/demo-1.mp4)

### Sidenote:
> This footage was recorded on a arm64 device running macOS/darwin with an “Rhodes Chop” processor 10 Cores and G14S Architecture 16 Cores GPU. Some parts of the footage were sped up, The list of models that are being used can be seen in [here](https://raw.githubusercontent.com/albertstarfield/alpaca-electron-zephyrine/main/usr/engine_component/LLM_Model_Index.js).
## 🚀 Quick Start Guide

1. **Follow the Guide**: Look at the guide provided and do what it says.
2. **Wait for the Model**: After following the guide, be patient. Keep an eye on your terminal or console for messages about the model downloading itself automatically.
3. **Relaunch the Program**: Once the model has finished downloading, restart the program.

That's it! You're ready to go.

### Windows
> **Note**  
> For Windows the launch sequence command are currently broken, You could try running the following command but it is under heavy development, you can see the update on the Issue section about Windows Native Support. In the meantime you could run it using wsl2 and wslg

### Linux and macOS (Running manually)

> The procedure will encompass an automated compilation process, wherein all components shall be seamlessly and effortlessly installed. Consequently, the necessity for specific release binaries shall be obviated.

>
>```git clone https://github.com/albertstarfield/alpaca-electron-zephyrine```
>
>Change your current directory to alpaca-electron:
>
>```cd alpaca-electron-zephyrine```
>
>Install application specific dependencies: 
>
> ```chmod +x ./launchcontrol/run.sh ```
>
> Run it
>
> ```./launchcontrol/run.sh```


3. The program will commence automatically, affording you the opportunity to initiate a conversation at your convenience.

## 🔧 Troubleshooting

[Click here to see the general troubleshooting](https://github.com/albertstarfield/alpaca-electron-zephyrine/blob/main/documentation/Developer%20Documentation/Troubleshooting%20Quick%20Guide.md)

## 👨‍💻 Credits
The development of this project owes credit to several contributors whose valuable efforts have shaped its foundation and brought it to fruition.

Sincere gratitude is extended to [@itsPi3141](https://github.com/ItsPi3141/alpaca-electron)  for their original creation of the Alpaca-electron program, which served as the starting point and inspiration for this work.

Furthermore, recognition is due to the following individuals for their significant contributions to the project:

[@antimatter15](https://github.com/antimatter15/alpaca.cpp) for their contributions in creating the alpaca.cpp component.
[@ggerganov](https://github.com/ggerganov/llama.cpp) for their pivotal role in the development of the llama.cpp component and the GGML base backbone behind alpaca.cpp.
Meta and Stanford for their invaluable creation of the LLaMA and Alpaca models, respectively, which have laid the groundwork for the project's AI capabilities.
Additionally, special appreciation goes to [@keldenl](https://github.com/keldenl) for providing arm64 builds for MacOS and [@W48B1T](https://github.com/W48B1T) for providing Linux builds, which have greatly enhanced the project's accessibility and usability across different platforms.

Lastly, although the project may not garner widespread attention as [@Willy030125](https://github.com/Willy030125/alpaca-electron-GGML-v2-v3), we acknowledge and cherish the efforts put forth by all contributors. Let this work be a testament to the dedication and collective collaboration that underpin academic and technological advancements.

With deep appreciation for everyone involved, Project Zephyrine signs off.
