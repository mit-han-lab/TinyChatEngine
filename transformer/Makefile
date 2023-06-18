# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -pthread -O3

# Executable and source files
TEST_TARGET = test_ops test_Int8OPTAttention test_Int8OPTDecoderLayer test_Int8OPTDecoder test_OPTForCausalLM test_OPTTokenizer test_LLaMATokenizer test_OPTGenerate test_Fp32llamaAttention test_Fp32llamaDecoderLayer test_Fp32llamaDecoder test_Fp32LlamaForCausalLM test_Int4llamaAttention test_Int4llamaDecoderLayer test_Int4llamaDecoder test_Int4LlamaForCausalLM
PROFILE_TARGET = profile_Fp32LlamaForCausalLM profile_Int4LlamaForCausalLM profile_OPTForCausalLM profile_ops
APP_TARGET = demo
TARGET = $(TEST_TARGET) $(PROFILE_TARGET) $(APP_TARGET)

LIB_DIR = ../matmul_optimization/src
LIB_SRC = $(wildcard $(LIB_DIR)/lib/*.cc)
INCLUDE_DIRS = -I$(LIB_DIR) -I./include -I./json/single_include/
LIB =

ifeq ($(shell uname -m),x86_64)
	# For Intel machines with AVX
	LIB_AVX_SRC = $(wildcard $(LIB_DIR)/lib/avx/*.cc)
	LIB_SRC += $(LIB_AVX_SRC)
	CXXFLAGS += -mavx2
else ifeq ($(shell uname -p),arm)
	# For ARM A-series (such as Mac M1) with NEON
	LIB_NEON_SRC = $(wildcard $(LIB_DIR)/lib/neon/*.cc)
	LIB_SRC += $(LIB_NEON_SRC)
	CXXFLAGS += -march=native -framework Accelerate
	INCLUDE_DIRS += -I/opt/homebrew/opt/boost/include
	LIB += -L/opt/homebrew/opt/boost/lib
else
	# Use paltform independent implementation
	# @echo "Device unsupported! Using the reference implementation will largely impacts the performance.
	LIB_REF_SRC = $(wildcard $(LIB_DIR)/lib/ref/*.cc)
	LIB_SRC += $(LIB_REF_SRC)
endif

# $(info $(LIB_SRC))

SRC_DIR = src
SRC = $(wildcard src/*.cc)
OPS =  $(wildcard src/ops/*.cc)
SRC += $(OPS)
SRC += $(LIB_SRC)

# Check if CUDA is available
CUDA_AVAILABLE := $(shell command -v /usr/local/cuda/bin/nvcc 2> /dev/null)
CUDA_SRCS = $(wildcard $(LIB_DIR)/lib/cuda/*.cu)
# General/Server GPUs (e.g., Nvidia RTX A6000, GeForce RTX 3090)
# TORCH_DIR = ../matmul_optimization/src/lib/cuda/libtorch
# Edge GPUs (e.g., Nvidia Jetson AGX Orin, Jetson Orin Nano)
TORCH_DIR = /home/wweichen/archiconda3/envs/TinyLLMEngine_py38/lib/python3.8/site-packages/torch
LDFLAGS =

ifdef CUDA_AVAILABLE
    $(info CUDA is available)
	CXX = /usr/local/cuda/bin/nvcc
# Please modify '-arch=sm_87' according to your GPU architecture/compute capability (https://developer.nvidia.com/cuda-gpus)
	CXXFLAGS = -std=c++17 -O3 -DCUDA_ENABLE -Xcompiler "-pthread" -arch=sm_87
	SRC += $(CUDA_SRCS)
# Please modify the paths in the following two lines ('-I/usr/include/python3.8' & '-lpython3.8') according to your Python development environment
 	INCLUDE_DIRS += -I/usr/local/cuda/include -I$(TORCH_DIR)/include -I$(TORCH_DIR)/include/torch/csrc/api/include -I/usr/include/python3.8
 	LDFLAGS += -L$(TORCH_DIR)/lib -ltorch -lc10 -lcudart -L/usr/local/cuda/lib64 -Xlinker -rpath=$(TORCH_DIR)/lib -lpython3.8
endif

# Default target
all: $(TARGET)

# Linking
test_ops:
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -o test_ops tests/test_ops.cc $(SRC) $(LIB) $(LDFLAGS)

test_Int8OPTAttention:
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -o test_Int8OPTAttention tests/test_Int8OPTAttention.cc $(SRC) $(LIB) $(LDFLAGS)

test_Int8OPTDecoderLayer:
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -o test_Int8OPTDecoderLayer tests/test_Int8OPTDecoderLayer.cc $(SRC) $(LIB) $(LDFLAGS)

test_Int8OPTDecoder:
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -o test_Int8OPTDecoder tests/test_Int8OPTDecoder.cc $(SRC) $(LIB) $(LDFLAGS)

test_OPTForCausalLM:
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -o test_OPTForCausalLM tests/test_OPTForCausalLM.cc $(SRC) $(LIB) $(LDFLAGS)

test_OPTTokenizer:
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -o test_OPTTokenizer tests/test_OPTTokenizer.cc $(SRC) $(LIB) $(LDFLAGS)

test_LLaMATokenizer:
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -o test_LLaMATokenizer tests/test_LLaMATokenizer.cc $(SRC) $(LIB) $(LDFLAGS)

test_OPTGenerate:
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -o test_OPTGenerate tests/test_OPTGenerate.cc $(SRC) $(LIB) $(LDFLAGS)

test_Fp32llamaAttention:
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -o test_Fp32llamaAttention tests/test_Fp32llamaAttention.cc $(SRC) $(LIB) $(LDFLAGS)

test_Fp32llamaDecoderLayer:
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -o test_Fp32llamaDecoderLayer tests/test_Fp32llamaDecoderLayer.cc $(SRC) $(LIB) $(LDFLAGS)

test_Fp32llamaDecoder:
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -o test_Fp32llamaDecoder tests/test_Fp32llamaDecoder.cc $(SRC) $(LIB) $(LDFLAGS)

test_Fp32LlamaForCausalLM:
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -o test_Fp32LlamaForCausalLM tests/test_Fp32LlamaForCausalLM.cc $(SRC) $(LIB) $(LDFLAGS)

test_Int4llamaAttention:
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -o test_Int4llamaAttention tests/test_Int4llamaAttention.cc $(SRC) $(LDFLAGS)

test_Int4llamaDecoderLayer:
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -o test_Int4llamaDecoderLayer tests/test_Int4llamaDecoderLayer.cc $(SRC) $(LDFLAGS)

test_Int4llamaDecoder:
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -o test_Int4llamaDecoder tests/test_Int4llamaDecoder.cc $(SRC) $(LDFLAGS)

test_Int4LlamaForCausalLM:
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -o test_Int4LlamaForCausalLM tests/test_Int4LlamaForCausalLM.cc $(SRC) $(LDFLAGS)

demo:
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -o demo application/demo.cc $(SRC) $(LIB) $(LDFLAGS)

profile_OPTForCausalLM:
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -D PROFILER -o profile_OPTForCausalLM tests/test_OPTForCausalLM.cc $(SRC) $(LIB) $(LDFLAGS)

profile_Fp32LlamaForCausalLM:
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -D PROFILER -o profile_Fp32LlamaForCausalLM tests/test_Fp32LlamaForCausalLM.cc $(SRC) $(LIB) $(LDFLAGS)

profile_Int4LlamaForCausalLM:
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -D PROFILER -o profile_Int4LlamaForCausalLM tests/test_Int4LlamaForCausalLM.cc $(SRC) $(LIB) $(LDFLAGS)

profile_ops:
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -D PROFILER -o profile_ops tests/test_ops.cc $(SRC) $(LIB) $(LDFLAGS)

# Clean up
clean:
	rm -f $(TARGET)
