This is the implementation of TinyLLMEngine, a memory-efficient and high-performance neural network library for quantized large language model (LLM) on the edge.

## Supported devices

We currently support int8 OPT and fp32/int4 LLaMA models on Intel and Apple M-series CPU:

- Target models: LLaMA, OPT
- Target device: Intel CPU, Apple M-series CPU/GPU, Nvidia edge GPU
- Target quantization schemes: w4a16 (GPU), w4a32 (CPU)

## Prerequisites

Install Packages

```bash
conda create -n TinyLLMEngine python=3.8
conda activate TinyLLMEngine
pip install -r requirements.txt
```

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

## Demo with AWQ model

### Get started

```bash
# pull repo
git clone --recursive https://github.com/mit-han-lab/TinyLLMEngine.git
cd TinyLLMEngine/transformer

# download and convert the AWQ model to int4 format, this will take a while...
./download_model.sh LLaMA_7B_AWQ models
```

### Quantize model

- #### Intel CPU

```bash
python model_quantizer.py --model_path models/LLaMA_7B_AWQ --method Q4_0 --data_type fp32 --group_size 128
```

- #### Apple M-series CPU/GPU

```bash
python model_quantizer.py --model_path models/LLaMA_7B_AWQ --method Q4_4 --data_type fp32 --group_size 128
```

- #### Nvidia edge GPU

```bash
python model_quantizer.py --model_path models/LLaMA_7B_AWQ --method Q4_0 --data_type fp32 --group_size 128 --cuda_is_available True
```

### Compile and run the demo

````bash
# compile the demo program
make -j

# run the demo
./demo # demo.exe on Windows

Using model: LLaMA7B_AWQ
Using LLaMA's default data format: INT4
Loading model... Finished!
Please enter an instruction: Write a program to sort an integer array.
Generated:
```python
def sort_array(arr):
    if len(arr) <= 1:
        return arr

    pivot = arr[0]
    left = []
    right = []
    for num in arr:
        if num < pivot:
            left.append(num)
        else:
            right.append(num)

    return sort_array(left) + [pivot] + sort_array(right)
```
Section, Total time(us), Average time(us), Count, GOPs
Token generation, 8345204, 81021, 103, N/A, N/A

````
