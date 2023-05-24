This is the implementation of TinyLLMEngine, a memory-efficient and high-performance neural network library for quantized large language model (LLM).

## Road map

We currently support int8 OPT models on Intel CPU and plan to support more models and devices:
- Target models: LLaMA, OPT
- Target device: Intel CPU, Apple M-series CPU/GPU, Nvidia edge GPU
- Target quantization schemes: w4a16 (GPU), w4a32 (CPU)

## Usage
- Example commands to run the int8 OPT demo (currently working on Intel only)
``` bash
sh ./download.sh # This will download 125m, 1.3B, 6.7B
make -j
./demo 6.7B

```


