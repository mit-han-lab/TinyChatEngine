# TinyChatEngine: A Efficient Neural Network Library for LLM

TinyChatEngine is a powerful neural network library specifically designed for the efficient deployment of quantized large language models (LLMs) on edge devices.

![demo](assets/figures/chat.gif)

## Prerequisites

### MacOS

For MacOS, install boost and llvm by

```bash
brew install boost
brew install llvm
```

For M1/M2 users, install Xcode from AppStore to enable the metal compiler for GPU support.

### Windows

For Windows, download and install the GCC compiler with MSYS2. Follow this tutorial: https://code.visualstudio.com/docs/cpp/config-mingw for installation.

- Install required dependencies with MSYS2

```
pacman -S --needed base-devel mingw-w64-x86_64-toolchain make unzip git
```

- Add binary directories (e.g., C:\\msys64\\mingw64\\bin and C:\\msys64\\usr\\bin) to the environment path

### Kernel support list

| Kernel precision | x86 (Intel/AMD CPU) | ARM (Apple M1/M2) | Nvidia GPU | Apple GPU |
| ------ | --------------------------- | --------- | --------- | --------- |
| FP16/FP32   |  ✅    |    ✅  |         |
| W4A16  |      |      |  ✅  | ✅
| W4A32  |  ✅  |  ✅  |      | ✅
| W4A8   |  ✅  |  ✅  |      |
| W8A8   |  ✅  |  ✅  |      |

### Model and quantization method support list

| Models | AWQ (INT4) | SmoothQuant (INT8) |
| ------ | --------- | --------- |
| LLaMA-2   |  ✅  |
| LLaMA  |  ✅  |
| Vicuna |  ✅  |
| OPT  | ✅ |  ✅
| MPT   |    |
| Falcon   |    |

## Quantization and Model Support

The goal of TinyChatEngine is to support various quantization methods on various devices. For example, At present, it supports the quantized weights for int8 OPT models that originate from [smoothquant](https://github.com/mit-han-lab/smoothquant)  and can be converted to TinyChatEngine format using the provided conversion script [opt_smooth_exporter.py](transformer/opt_smooth_exporter.py). For LLaMA models, scripts are available for converting Huggingface format checkpoints to our [format](transformer/llama_exporter.py), and for quantizing them to specific methods [based on your device](transformer/model_quantizer.py). We also plan to more models.

### Device-specific Quantization Methods

Different target devices require different quantization methods due to the variants of kernel implementation that suit the SIMD bit-width and instructions supported by your device. To quantize your LLaMA model to int4, please consult the following table:

| Platforms  | ISA | Quantization methods |
| ------------- | ------------- |  ------------- |
| Intel/AMD |  x86-64  | QM_x86  |
| M1/M2 Mac | arm | QM_ARM  |

Example of quantizing a LLaMA model for an Intel/AMD laptop:

```bash
python model_quantizer.py --model_path models/LLaMA_7B --method QM_x86 --output_path INT4/
```

Example of quantizing a LLaMA model for an M1/M2 Macbook:

```bash
python model_quantizer.py --model_path models/LLaMA_7B --method QM_ARM --output_path INT4/
```

### Download and deploy models from our Model Zoo

We offer a selection of models that have been tested with TinyChatEngine. These models can be readily downloaded and deployed on your device. To download a model, locate the target model's ID in the table below and use the associated script.

| Models  | Size | ID | Supported Precision |
| ------------- | ------------- |  ------------- |  ------------- |
| LLaMA-2 |  7B-chat  | LLaMA_7B_2_chat  |  INT4 |
| LLaMA | 7B/7B-AWQ | LLaMA_7B/LLaMA_7B_AWQ  |  INT4 |
| OPT | 125m/1.3B/6.7B | OPT_125/OPT_1.3B/OPT_6.7B  | INT8 |

For instance, to download the quantized LLaMA-2-7B-chat model:

- On a Intel/AMD latptop:
  ```bash
  python download_model.py --model LLaMA_7B_2_chat --QM QM_x86
  ```
- On a M1/M2 Macbook:
  ```bash
  python download_model.py --model LLaMA_7B_2_chat --QM QM_ARM
  ```

To deploy the quantized model with TinyChatEngine, compile the chat program and run it with the model ID and precision.

```
make chat -j
./chat LLaMA_7B_2_chat INT4
```

## Step-by-step to deploy LLaMA2-7B-chat with TinyChatEngine

Here, we provide step-by-step instructions to deploy LLaMA2-7B-chat with TinyChatEngine from scratch.

- Download the repo.
  ```bash
  # pull repo
  git clone --recursive https://github.com/mit-han-lab/TinyChatEngine.git
  ```
- Download the quantized LLaMA2-7B-chat model from our model zoo.
  ```bash
  cd TinyChatEngine/transformer
  ```
  - On a x86 device (e.g., Intel/AMD laptop)
    ```bash
    python download_model.py --model LLaMA_7B_2_chat --QM QM_x86
    ```
  - On a ARM device (e.g., M1/M2 Macbook)
    ```bash
    python download_model.py --model LLaMA_7B_2_chat --QM QM_ARM
    ```
  - On a CUDA device (e.g., Jetson AGX Orin)
    ```bash
    python download_model.py --model LLaMA_7B_2_chat --QM QM_CUDA
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

## Instructions to run a speech-to-speech chatbot demo

- Follow instructions above to deploy LLaMA2-7B-chat

- Configure whisper.cpp (Note)

  ```bash
  cd whisper.cpp

  # Install SDL2 on Linux
  sudo apt-get install libsdl2-dev
  # Install SDL2 on Mac OS
  brew install sdl2

  git apply ./../clean_up_patch
  bash ./models/download-ggml-model.sh base.en
  # NVIDIA GPU (Note: you may need to change the Makefile of whisper.cpp depending on your environment or device)
  WHISPER_CUBLAS=1 make -j stream
  # Otherwise
  make stream
  cd ../
  ```

- If you have an edge device and want a better TTS program than espeak, download [piper](https://github.com/rhasspy/piper)

  ```bash
    mkdir TTS
    wget https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_arm64.tar.gz
    tar -xvzf piper_arm64.tar.gz
  ```

  - Download your preferred voice from the [huggingface repo](https://huggingface.co/rhasspy/piper-voices/tree/v1.0.0) and drag both the .onxx and .onnx.json files into the TTS directory

- Edit the listen shell file in the transformers directory so whisper.cpp is using your preferred parameters.

  ```bash
  nano listen
  ```

- Edit the speak shell file in the transformers directory so the demo uses your preferred TTS program.

  ```bash
  nano speak
  ```

- Compile and start the voicechat locally.

  ```bash
  make -j voicechat
  ./voicechat # voicechat.exe on Windows
  ```

## Related Projects

[TinyEngine](https://github.com/mit-han-lab/tinyengine)

[Smoothquant](https://github.com/mit-han-lab/smoothquant)

[AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://github.com/mit-han-lab/llm-awq)

## Acknowledgement

[llama.cpp](https://github.com/ggerganov/llama.cpp)

[transformers](https://github.com/huggingface/transformers)
