This is the implementation of TinyLLMEngine, a memory-efficient and high-performance neural network library for quantized large language model (LLM) on the edge.

## Supported devices

We currently support int8 OPT and fp32/int4 LLaMA models on Intel and Apple M-series CPU:

- Target models: LLaMA, OPT
- Target device: Intel CPU, Apple M-series CPU/GPU, Nvidia edge GPU (on-going)
- Target quantization schemes: w4a16 (GPU), w4a32 (CPU)

## Prerequisites

### MacOS

For MacOS, install boost and llvm by

```bash
brew install boost
brew install llvm
```

For M1/M2 users, install xcode from AppStore to enable metal compiler for GPU support.

### Windows

For Windows, download and install the GCC compiler with MSYS2. Follow this tutorial: https://code.visualstudio.com/docs/cpp/config-mingw for installation.

- Install required dependencies with MSYS2

```
pacman -S --needed base-devel mingw-w64-x86_64-toolchain make unzip git
```

- Add binary directories (e.g., C:\\msys64\\mingw64\\bin and C:\\msys64\\usr\\bin) to the enviroment path

## Demo with LLaMA2 model

```bash
# pull repo
git clone --recursive https://github.com/mit-han-lab/TinyLLMEngine.git
cd TinyLLMEngine/transformer
# download and convert the AWQ model to int4 format, this will take a while...
python download_model.py LLaMA_7B_2_chat
python model_quantizer.py --model_path models/LLaMA_7B_2_chat --method Q4_0 # Use Q4_4 for M1/M2 MacBook
# compile the demo program
make -j
# run the demo
./demo # demo.exe on Windows
Using model: LLaMA7B_2_chat
Using LLaMA's default data format: INT4
Loading model... Finished!
Please enter an instruction: Write a syllabus for Operating Systems.
Of course! Here is a sample syllabus for a college-level course on operating systems:
Course Title: Introduction to Operating Systems
Course Description: This course provides an overview of the fundamental concepts and techniques used in modern operating systems, including process management, memory management, file systems, security, and I/O devices. Students will learn how these components work together to provide a platform for running applications and programs on a computer.
Course Objectives:
* Understand the basic architecture of an operating system
* Learn about processes, threads, and process scheduling algorithms
* Study memory management techniques such as paging and segmentation
* Explore file systems including file organization, storage devices, and file access methods
* Investigate security mechanisms to protect against malicious software attacks
* Analyze input/output (I/O) operations and their handling by the operating system
...
```
