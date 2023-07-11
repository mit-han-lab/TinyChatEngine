This is the implementation of TinyLLMEngine, a memory-efficient and high-performance neural network library for quantized large language model (LLM) on the edge.

## Supported devices

We currently support int8 OPT and fp32/int4 LLaMA models on Intel and Apple M-series CPU:

- Target models: LLaMA, OPT
- Target device: Intel CPU, Apple M-series CPU/GPU, Nvidia edge GPU (on-going)
- Target quantization schemes: w4a16 (GPU), w4a32 (CPU)

## Prerequisites

For MacOS, install boost and llvm by

```bash
brew install boost
brew install llvm
```

For M1/M2 users, install xcode from AppStore to enable metal compiler for GPU support.

## Demo with AWQ model

````bash
# pull repo
git clone --recursive https://github.com/mit-han-lab/TinyLLMEngine.git
cd TinyLLMEngine/transformer
# download and convert the AWQ model to int4 format, this will take a while...
./download_model.sh LLaMA_7B_AWQ models
python model_quantizer.py --model_path models/LLaMA_7B_AWQ --method Q4_0 # Use Q4_2 for M1/M2 MacBook
# compile the demo program
make -j
# run the demo
./demo
Using model: LLaMA7B_AWQ
Using LLaMA's default data format: INT4
Loading model... Finished!
Please enter a line of text: Write a program to sort an integer array. # instruction
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
Token generation, 29143198, 282943, 103, N/A, N/A

````
