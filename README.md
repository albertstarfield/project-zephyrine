<h1 align="center">
<sub>
<img src="https://raw.githubusercontent.com/ItsPi3141/alpaca-electron/main/icon/alpaca-chat-logo.png?raw=true" height=144>
</sub>
<br>
Project Zephyrine
</h1>
<br>
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

<p align="center"><i>"Project Zephyrine: Empowering Your Chat Experience with LLaMA (GGML v2 GGML v3), LLM, mpt, and GPU Acceleration in an Updated Alpaca Electron Chatbot Local GUI"</i></p>
<hr>



## 📃 Features & to-do

- [x] Runs locally on your computer, internet connection is not needed except when trying to access the web
- [x] Runs llama-2, llama, mpt, gpt-j, dolly-v2, gpt-2, gpt-neox, starcoder
- [x] Can run purely on CPU
- [x] "Borrowed" UI from *that* popular chat AI Xdd
- [x] Supports Windows(untested), MacOS (untested), and Linux
- [x] Context memory
- [x] Partial GPU/FPGA(opencl)/Tensor acceleration (cuBLAS, openBLAS, clBLAST, Metal) [EXPERIMENTAL WARNING!]
- [x] DuckDuckGo integration for web access
- [x] Pdf-parse Integration for local literature documents resources access
- [ ] Chat history
- [ ] Integration with Stable Diffusion



## 🎞 Demonstration Screenshot

![Demonstration](https://github.com/albertstarfield/alpaca-electron-zephyrine-ggmlv2v3-universal/blob/main/picDemo/demo0.png?raw=true)
![Demonstration](https://github.com/albertstarfield/alpaca-electron-zephyrine-ggmlv2v3-universal/blob/main/picDemo/demo1.png?raw=true)

## 🚀 Quick Start Guide

1. You can easily download GGMLv2 and GGMLv3 from [The Bloke Huggingface Repo](https://huggingface.co/TheBloke) from there you can see multiple and various of quantization for instance the llama-2 7B GGML
> https://huggingface.co/TheBloke/Llama-2-7B-GGML

2. Run the specific commands from the guide below


### Windows
> **Note**  
> The process will involve an automated compilation procedure, with all components being installed seamlessly and effortlessly. As a result, there will be no need for specific release binaries.

> Open Powershell and make sure you have git cli installed
>
>```git clone https://github.com/ItsPi3141/alpaca-electron.git```
>
>Change your current directory to alpaca-electron:
>
>```cd alpaca-electron```
>
> Run it with your desired model mode for instance 
>
> ```./run.ps1```
- If the model has been loaded into RAM but text generation doesn't seem start, [check](https://ark.intel.com/content/www/us/en/ark.html#@Processors) to see if your CPU is compatible with the [AVX2](https://edc.intel.com/content/www/us/en/design/ipla/software-development-platforms/client/platforms/alder-lake-desktop/12th-generation-intel-core-processors-datasheet-volume-1-of-2/002/intel-advanced-vector-extensions-2-intel-avx2/) instruction set. If it does not support AVX2, Alpaca Electron will use AVX instead, which is much slower so be patient. 

### Linux and macOS

> The process will involve an automated compilation procedure, with all components being installed seamlessly and effortlessly. As a result, there will be no need for specific release binaries.

>
>```git clone https://github.com/ItsPi3141/alpaca-electron.git```
>
>Change your current directory to alpaca-electron:
>
>```cd alpaca-electron```
>
>Install application specific dependencies: 
>
> ```chmod +x ./run.sh ```
>
> Run it with your desired model mode for instance 
>
> ```./run.sh llama```


3. Once done installing, it'll ask for a valid path to a model. Now, go to where you placed the model, hold shift, right click on the file, and then click on "Copy as Path". Then, paste this into that dialog box and click `Confirm`. 

4. The program will automatically start and now you can begin chatting!

> **Note**  
> The program will also accept any other 4 bit quantized .bin model files. If you can find other .bin Alpaca model files, you can use them instead of the one recommended in the Quick Start Guide to experiment with different models. As always, be careful about what you download from the internet.

## 🔧 Troubleshooting

### General
- If you get an error that says "Invalid file path" when pasting the path to the model file, you probably have some sort of misspelling in there. Try copying the path again or using the file picker.
- If you get an error that says "Couldn't load model", your model is probably corrupted or incompatible. Try downloading the model again.
- If you face other problems or issues not listed here, create an issue in the "Issues" tab at the top of this page. Describe in detail what happens, and include screenshots. 

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
