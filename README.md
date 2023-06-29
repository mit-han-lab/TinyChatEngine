This is the implementation of TinyLLMEngine, a memory-efficient and high-performance neural network library for quantized large language model (LLM).

## Road map

We currently support int8 OPT and fp32/int4 LLaMA models on Intel CPU and plan to support more models and devices:

- Target models: LLaMA, OPT
- Target device: Intel CPU, Apple M-series CPU/GPU, Nvidia edge GPU
- Target quantization schemes: w4a16 (GPU), w4a32 (CPU)

## Prerequisites

For MacOS, install boost by

```bash
brew install boost
```

## Usage

- Example commands to run the int8 OPT demo (currently working on Intel only)

```bash
cd transformer
git submodule update --init
./download.sh # This will download 125m, 1.3B, 6.7B OPT, and 7B LLaMA models
make -j
./demo OPT6.7B INT8
Model: OPT6.7B selected
Data format: INT8 selected
Loading model... Finished!
Please enter a line of text: John went to MIT and study Computer Science.
input:John went to MIT and study Computer Science.
Generated:
 He graduated at the top of his class earning two bachelor's degrees and a master's degree all in 3 years.  In 2004 John started working on the website for FUEL.net which was an online community for racing enthusiasts that helped drivers track their vehicles through GPS, track performance with speed sensors and analyze their driving style. This website later become the basis for the "DriveLogger" product. In 2008 he co-founded a company called DriveLogger based in Cambridge, MA. They were funded by the MIT Technology Review Accelerator and are still active today.  FUEL.net was launched in 2001 as a community site to help NASCAR fans track their favorite drivers through GPS. John's first project with FUEL was implementing their API into the "Driver Tracker" website, which allowed users to see exactly where the driver of their choice was in relation to them on the racetrack and what lap they were currently on.  John then moved on to the more complex task of actually running the site itself and maintaining it. During this time he also started work on a product called "Driver Tracker", which allowed users to track where their favorite drivers were at any given time, even if they weren't racing yet.  In 2003 John was
```

- Example commands to run the w4a32 LLaMA demo (Intel CPU)

```bash
cd transformer
./download.sh # This will download 125m, 1.3B, 6.7B OPT, and 7B LLaMA models
python model_quantizer.py --model_path="models/LLaMA_7B" --method="Q4_0" --data_type="fp32" # Quantize LLaMA7B model from fp32 to int4 by using Q4_0 quantization method
make -j
./demo LLaMA7B INT4 # or `./demo LLaMA7B FP32` to run the fp32 LLaMA demo
Model: LLaMA7B selected
Data format: INT4 selected
Loading model... Finished!
Please enter a line of text: John went to MIT and study Computer Science.
input:John went to MIT and study Computer Science.
Generated:
 A perspective on the admissions process (for an engineering program)
MIT and Stanford have long been considered to be two of the top universities in the country. Stanford is often regarded as a more liberal arts school, while MIT focuses more on its engineering program. When looking at which of these schools you want to apply to, your choice will likely come down to preference and statistics. The average GPA for an applicant to MIT is 4.01 while the same statistic for Stanford is 3.92. Both of these universities accept students with scores in a range between 580-790 on their SATs, which are both above the national average of 663.
The application process to MIT is much more straightforward than that at Stanford. There are no essays for MIT; you just have to answer some questions about yourself and your academic history (the “MIT Application Profile”), submit a resume with your extracurricular activities, volunteer experience, awards, etc., and provide an unofficial transcript. The application profile is available online so that students can get started much earlier on their applications than they would for Stanford.
```

- Example commands to run the w4a32 LLaMA demo (Apple M-series CPU and GPU)

```bash
cd transformer
./download.sh # This will download 125m, 1.3B, 6.7B OPT, and 7B LLaMA models
python model_quantizer.py --model_path="models/LLaMA_7B" --method="Q4_2" --data_type="fp32" # Quantize LLaMA7B model from fp32 to int4 by using Q4_2 quantization method
make -j
./demo LLaMA7B INT4
Model: LLaMA7B selected
Data format: INT4 selected
Loading model... Finished!
Please enter a line of text: Difference between MIT and Stanford:
Generated:
 A perspective on the admissions process (for an engineering program)
MIT and Stanford have long been considered to be two of the top universities in the country. Stanford is often regarded as a more liberal arts school, while MIT focuses more on its engineering program. When looking at which of these schools you want to apply to, your choice will likely come down to preference and statistics. The average GPA for an applicant to MIT is 4.01 while the same statistic for Stanford is 3.92. Both of these universities accept students with scores in a range between 580-790 on their SATs, which are both above the national average of 663.
The application process to MIT is much more straightforward than that at Stanford. There are no essays for MIT; you just have to answer some questions about yourself and your academic history (the “MIT Application Profile”), submit a resume with your extracurricular activities, volunteer experience, awards, etc., and provide an unofficial transcript. The application profile is available online so that students can get started much earlier on their applications than they would for Stanford.
Stanford
```

- Example commands to run the w4a16 LLaMA demo (Nvidia GPU)

```bash
cd matmul_optimization/src/lib/cuda
# Download LibTorch
wget https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu117.zip # Modify the url according to your CUDA and CXX version (https://pytorch.org/get-started/locally/)
unzip libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu117.zip

cd ../../../../transformer
./download.sh # This will download 125m, 1.3B, 6.7B OPT, and 7B LLaMA models
python model_quantizer.py --model_path="models/LLaMA_7B" --method="Q4_0" --data_type="fp32" # Quantize LLaMA7B model from fp32 to int4 by using Q4_0 quantization method
# Before make, please modify Lines 53, 56 and 57 in Makefile according to your GPU architecture and Python development environment
make -j
./demo LLaMA7B INT4 # or `./demo LLaMA7B FP32` to run the fp32 LLaMA demo
Model: LLaMA7B selected
Data format: INT4 selected
Loading model... Finished!
Please enter a line of text: John went to MIT and study Computer Science.
input:John went to MIT and study Computer Science.
Generated:
## TBA
```
