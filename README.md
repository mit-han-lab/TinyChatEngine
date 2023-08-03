# TinyLLMEngine: A Efficient Neural Network Library for LLM

TinyLLMEngine is a powerful neural network library specifically designed for the efficient deployment of quantized large language models (LLMs) on edge devices.

![demo](assets/figures/chat.gif)

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

## Quantization and Model Support

At present, we support int8 OPT and int4 LLaMA models for x86 and ARM CPUs as well as Apple's M-series GPUs. Quantized weights for int8 OPT models originate from [smoothquant](https://github.com/mit-han-lab/smoothquant)  and can be converted to TinyLLMEngine format using the provided conversion script [opt_smooth_exporter.py](transformer/opt_smooth_exporter.py). For LLaMA models, scripts are available for converting Huggingface format checkpoints to our [format](transformer/llama_exporter.py), and for quantizing them to specific methods [based on your device](transformer/model_quantizer.py). We also plan to support edge GPUs, which will be coming soon.

### Device-specific Quantization Methods

Different target devices require different quantization methods due to the variants of kernel implementation that suit the SIMD bit-width and instructions supported by your device. To quantize your LLaMA model to int4, please consult the following table:

| Platforms  | ISA | Quantization methods |
| ------------- | ------------- |  ------------- |
| Intel/AMD |  x86-64  | QM_x86  |
| M1/M2 Mac | arm | QM_ARM  |

Example of quantizing a LLaMA model for a Intel/AMD laptop:

```bash
python model_quantizer.py --model_path models/LLaMA_7B --method QM_x86 --output_path INT4/
```

Example of quantizing a LLaMA model for a M1/M2 Macbook:

```bash
python model_quantizer.py --model_path models/LLaMA_7B --method QM_ARM --output_path INT4/
```

### Download and deploy models from our Model Zoo

We offer a selection of models that have been tested with TinyLLMEngine. These models can be readily downloaded and deployed on your device. To download a model, locate the target model's ID in the table below and use the associated script.

| Models  | Size | ID | Supported Precision |
| ------------- | ------------- |  ------------- |  ------------- |
| LLaMA-2 |  7B-chat  | LLaMA_7B_2_chat  |  INT4 |
| LLaMA | 7B/7B-AWQ | LLaMA_7B/LLaMA_7B_AWQ  |  INT4 |
| OPT | 125m/1.3B/6.7B | OPT_125/OPT_1.3B/OPT_6.7B  | INT8 |

For instance, to download the quantized LLaMA-2-7B-chat model:

- On a Intel/AMD latptop:
  ```bash
  python download_model.py --model LLaMA_7B_2_chat --QM QM_ARM
  ```
- On a M1/M2 Macbook:
  ```bash
  python download_model.py --model LLaMA_7B_2_chat --QM QM_x86
  ```

To deploy the quantized model with TinyLLMEngine, compile the chat program and run it with the model ID and precision.

```
make chat -j
./chat LLaMA_7B_2_chat INT4
```

## Step-by-step to deploy LLaMA2-7B-chat with TinyLLMEngine

Here, we provide step-by-step instructions to deploy LLaMA2-7B-chat with TinyLLMEngine from scratch.

- Download the repo.
  ```bash
  # pull repo
  git clone --recursive https://github.com/mit-han-lab/TinyLLMEngine.git
  ```
- Download the quantized LLaMA2-7B-chat model from our model zoo.
  ```bash
  cd TinyLLMEngine/transformer
  ```
  - On a x86 device (e.g., Intel/AMD laptop)
    ```bash
    python download_model.py --model LLaMA_7B_2_chat --QM QM_x86
    ```
  - On a ARM device (e.g., M1/M2 Macbook)
    ```bash
    python download_model.py --model LLaMA_7B_2_chat --QM QM_ARM
    ```
- Compile and start the chat locally.
  ```bash
  make chat -j
  ./chat # chat.exe on Windows
  Using model: LLaMA7B_2_chat
  Using LLaMA's default data format: INT4
  Loading model... Finished!
  USER: Write a syllabus for Operating Systems.
  ASSISTANT:
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

## Related Projects

[TinyEngine](https://github.com/mit-han-lab/tinyengine).

[Smoothquant](https://github.com/mit-han-lab/smoothquant).

[AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://github.com/mit-han-lab/llm-awq)

## Acknowledgement

[llama.cpp](https://github.com/ggerganov/llama.cpp)

[transformers](https://github.com/huggingface/transformers)
