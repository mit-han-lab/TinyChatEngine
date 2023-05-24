This is the implementation of TinyLLMEngine, a memory-efficient and high-performance neural network library for quantized large language model (LLM).

## Road map

We currently support int8 OPT models on Intel CPU and plan to support more models and devices:
- Target models: LLaMA, OPT
- Target device: Intel CPU, Apple M-series CPU/GPU, Nvidia edge GPU
- Target quantization schemes: w4a16 (GPU), w4a32 (CPU)

## Usage
- Example commands to run the int8 OPT demo (currently working on Intel only)
``` bash
cd transformer
./download.sh # This will download 125m, 1.3B, 6.7B OPT models
make -j
./demo OPT6.7B
Model: OPT6.7B selected
Loading model... Finished!
Please enter a line of text: John went to MIT and study Computer Science.
input:John went to MIT and study Computer Science.
Generated: 
 He graduated at the top of his class earning two bachelor's degrees and a master's degree all in 3 years.  In 2004 John started working on the website for FUEL.net which was an online community for racing enthusiasts that helped drivers track their vehicles through GPS, track performance with speed sensors and analyze their driving style. This website later become the basis for the "DriveLogger" product. In 2008 he co-founded a company called DriveLogger based in Cambridge, MA. They were funded by the MIT Technology Review Accelerator and are still active today.  FUEL.net was launched in 2001 as a community site to help NASCAR fans track their favorite drivers through GPS. John's first project with FUEL was implementing their API into the "Driver Tracker" website, which allowed users to see exactly where the driver of their choice was in relation to them on the racetrack and what lap they were currently on.  John then moved on to the more complex task of actually running the site itself and maintaining it. During this time he also started work on a product called "Driver Tracker", which allowed users to track where their favorite drivers were at any given time, even if they weren't racing yet.  In 2003 John was

```


