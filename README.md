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

## 📃 Features & to-do

- [x] Operates locally on your computer, requiring an internet connection solely for web access.
- [x] Accommodates multiple model modes, including llama-2, llama-2-GGUF, llama, and falcon Model Mode.
> **Important Note:**
> It is imperative to acknowledge that the model modes (MPT, GPT-J, GPT-2, GPT-Neox, StarCoder) are currently in an experimental phase, and users may potentially encounter initialization issues. These issues may arise due to the absence of interactive functionality within the compiled binary. We strongly recommend exercising caution and being mindful of this potential limitation.
- [x] Can function exclusively on CPU architectures, such as x86_64 and arm64/aarch64.
- [x] Provides compatibility with Windows (untested), MacOS (untested on x86_64), and Linux operating systems.
- [x] Offers context memory capabilities.
- [x] Features partial GPU/MPS/FPGA (opencl)/Tensor acceleration using cuBLAS, openBLAS, clBLAST, and Metal.
- [x] Integrates with DuckDuckGo for web access.
- [x] Facilitates access to local literature documents and resources through local files integration, excluding formatted documents.
- [x] Includes a Granular Toggles Mode within the settings section.
- [x] Implements Markdown Formatted Response support.
- [ ] Implement Typing-like experience Response Support.
- [ ] Addresses issues with Native Windows Support.
- [ ] Incorporating the Convolutional Transformer (CvT) within the software framework to serve as the conduit for capturing contextual information from the screen.
- [ ] Integrating an Environment/Computer Interaction Pipeline, facilitating the execution of AI-generated commands.
- [ ] Enabling an autonomous cognitive loop in a mode reminiscent of AutoGPT.
- [ ] Introduces Chat history functionality.
- [ ] Adding Docker Support


## 🎞 Demonstration Screenshot

![Demonstration](https://github.com/albertstarfield/alpaca-electron-zephyrine/blob/main/documentation/demo0.png?raw=true)
![Demonstration](https://github.com/albertstarfield/alpaca-electron-zephyrine/blob/main/documentation/demo1.gif?raw=true)

## 🚀 Quick Start Guide

1. You may conveniently acquire GGMLv2, GGMLv3, GGUF, and GGUFv2 from [The Bloke Huggingface Repository](https://huggingface.co/TheBloke). Within this repository, you will find a diverse array of quantization options, exemplified by the Llama-2 Nous Hermes 7B GGUF model, accessible at the following link: [Nous Hermes Llama2 GGUF](https://huggingface.co/TheBloke/Nous-Hermes-Llama2-GGUF).

2. Execute the prescribed commands as delineated in the provided guide, taking into consideration the platform upon which you are operating.


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
> Run it with your desired model mode for instance 
>
> ```./run.sh```


3. Once done installing, it'll ask for a valid path to a model. Now, go to where you placed the model, hold shift, right click on the file, and then click on "Copy as Path". Then, paste this into that dialog box and click `Confirm`. 

4. The program will commence automatically, affording you the opportunity to initiate a conversation at your convenience.

> **Note**  
> The software shall accommodate 2_K bit quantized .bin model files, contingent upon the selected mode. Nonetheless, the utilization of a Q2_K quantized model is discouraged, as it may compromise the operational integrity of the LLMChild, resulting in subpar output quality. If alternative .bin Alpaca model files are available, they may be employed in lieu of the recommended model outlined in the Quick Start Guide for the purpose of exploring diverse model options. Caution is advised when obtaining content from online sources.

## 🔧 Troubleshooting

### General
- In the event that you encounter an error message stating "Invalid file path" when attempting to input the model file path, it is likely attributable to a typographical error. We recommend revisiting the path entry process or employing the file picker function for a more accurate input.
- Should you encounter an error message reading "Couldn't load model," it is plausible that your model file is either corrupted, incompatible, or configured incorrectly for the chosen model mode. We suggest exploring alternative modes that are compatible with the downloaded model, or, if the issue persists, consulting the model repository's readme section for guidance on reacquiring the model.
- For any other unforeseen challenges or issues not covered in the aforementioned scenarios, we kindly request that you create an issue within the "Issues" tab located at the top of this page. Please provide a comprehensive description of the issue, including accompanying screenshots for further assistance.

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
