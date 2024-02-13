<h1 align="center">
<sub>
<img src="https://github.com/albertstarfield/alpaca-electron-zephyrine/blob/main/documentation/Project%20Zephyrine%20Logo.jpg?raw=true" height=144>
</sub>

<br>
Project Zephyrine
</h1>

<h5 align="center"> </h5>

<p align="center">
  <a href="https://nodejs.org">
    <img src="https://img.shields.io/badge/node.js-6DA55F?style=for-the-badge&logo=node.js&logoColor=white">
  </a>
  <a href="https://www.electronjs.org/">
    <img src="https://img.shields.io/badge/Electron-191970?style=for-the-badge&logo=Electron&logoColor=white">
  </a>
  <a href="https://github.com/antimatter15/alpaca.cpp/">
    <img src="https://img.shields.io/badge/Alpaca.cpp-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white">
  </a>
</p>

<p align="center"><i>Greetings, I am Adelaide Zephyrine Charlotte, delighted to make your acquaintance. Shall we embark on our intellectual voyage through this endeavor? </i></p>

<hr>

> **Important Note:**
> A fundamental paradigmatic transition is underway in the evolution of this software. Acknowledging the eminent positions of LMStudio and OLlama GUI, which have garnered substantial attention and support at opportune junctures, contrasts starkly with my current solitary endeavor. Engaging in direct competition with them would inevitably lead to failure and obsolescence. Therefore, rather than embarking on a futile endeavor, this program is undergoing a metamorphosis towards a plug-and-play, readily deployable framework akin to Copilot, albeit mostly offline, with a concentrated focus on delineated entity that have MoE (Mix of Experts), thus eschewing the conventional model-based interface.

## 📃 Features & to-do

- [x] Operates locally on your computer, requiring an internet connection solely for web access.
- [x] Can function exclusively on CPU architectures, such as x86_64 and arm64/aarch64.
- [x] Provides compatibility with Windows* (untested), MacOS (untested on x86_64), and Linux operating systems.
- [x] Features partial GPU/MPS/FPGA (opencl)/Tensor acceleration using cuBLAS, openBLAS, clBLAST, Vulkan, and Metal.
- [x] Web access.
- [x] Chat history functionality.

Just a heads up, some details aren't listed here or in the `readme.md` file. Instead, they've been moved to the `TodoFeatureFullList.md` document or you can check them out [here](https://github.com/albertstarfield/alpaca-electron-zephyrine/blob/main/documentation/Developer%20Documentation/TodoFeatureFullList.md).




## 🎞 Demonstration Screenshot

![Demonstration](https://raw.githubusercontent.com/albertstarfield/alpaca-electron-zephyrine/main/documentation/demo-0.png)
![Demonstration_video](https://raw.githubusercontent.com/albertstarfield/alpaca-electron-zephyrine/main/documentation/demo-1.gif)

### Sidenote:
> This footage was recorded on a arm64 device running macOS/darwin with an “Rhodes Chop” processor 10 Cores and G14S Architecture 16 Cores GPU. Some parts of the footage were sped up because of the large size of the model nous-hermes-llama2-13b.Q5_K_S that was used. The footage may appear jittery and lack the typing effect that is present in the original version. This is because of the use of markdown, which will be fixed in a later update.

## 🚀 Quick Start Guide

1. **Follow the Guide**: Look at the guide provided and do what it says.
2. **Wait for the Model**: After following the guide, be patient. Keep an eye on your terminal or console for messages about the model downloading itself automatically.
3. **Relaunch the Program**: Once the model has finished downloading, restart the program.

That's it! You're ready to go.

### Windows
> **Note**  
> For Windows the launch sequence command are currently broken, You could try running the following command but it is under heavy development, you can see the update on the Issue section about Windows Native Support. In the meantime you could run it using wsl2 and wslg

### Linux and macOS

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
> ```chmod +x ./run.sh ```
>
> Run it
>
> ```./run.sh```


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
