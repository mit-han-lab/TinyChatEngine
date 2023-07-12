# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -pthread -O3

# Executable and source files
TEST_TARGET = test_ops #test_Int8OPTAttention test_Int8OPTDecoderLayer test_Int8OPTDecoder test_OPTForCausalLM test_OPTTokenizer test_LLaMATokenizer test_OPTGenerate test_Fp32llamaAttention test_Fp32llamaDecoderLayer test_Fp32llamaDecoder test_Fp32LlamaForCausalLM test_Int4llamaAttention test_Int4llamaDecoderLayer test_Int4llamaDecoder test_Int4LlamaForCausalLM
PROFILE_TARGET = #profile_Fp32LlamaForCausalLM profile_Int4LlamaForCausalLM profile_OPTForCausalLM profile_ops
APP_TARGET = #demo
TARGET = $(TEST_TARGET) $(PROFILE_TARGET) $(APP_TARGET)

LIB_DIR = ../matmul_optimization/src
LIB_SRC = $(wildcard $(LIB_DIR)/lib/*.cc)
INCLUDE_DIRS = -I$(LIB_DIR) -I./include -I./json/single_include/ -I./half-2.2.0/include/
LIB =
LDFLAGS =

# Check if CUDA is available
CUDA_AVAILABLE := $(shell command -v /usr/local/cuda/bin/nvcc 2> /dev/null)
ifdef CUDA_AVAILABLE
$(info CUDA is available!)
	CUDA_HOME = /usr/local/cuda
	CXX = $(CUDA_HOME)/bin/nvcc
# Please modify '-arch=sm_87' according to your GPU architecture/compute capability (https://developer.nvidia.com/cuda-gpus)
#	CXXFLAGS = -std=c++17 -O3 -DCUDA_ENABLE -Xcompiler "-pthread" -D_GLIBCXX_USE_CXX11_ABI=1
	CXXFLAGS = -std=c++17 -O3 -DCUDA_ENABLE -Xcompiler "-pthread"
	CUDA_SRCS = $(wildcard $(LIB_DIR)/lib/cuda/*.cu) $(wildcard $(LIB_DIR)/lib/cuda/*.cc)
	LIB_SRC += $(CUDA_SRCS) $(wildcard src/cuda/*.cu) $(wildcard src/cuda/*.cc) $(wildcard src/ops/cuda/*.cu) 
	INCLUDE_DIRS += -I./include/cuda
	IS_CUDA = cuda
else
$(info CUDA is unavailable!)
	LIB_SRC += $(wildcard src/non_cuda/*.cc)
	INCLUDE_DIRS += -I./include/non_cuda
	IS_CUDA = non_cuda
endif

ifeq ($(shell uname -m),x86_64)
# For x86_64 using half-precision (16-bit) floating point
#	INCLUDE_DIRS += -I./half-2.2.0/include/
	ifdef CUDA_AVAILABLE
# General/Server GPUs (e.g., Nvidia RTX A6000, GeForce RTX 3090) use LibTorch
#		PYTHON_DIR = ../../../anaconda3/envs/TinyLLMEngine_py38
#		TORCH_DIR = ../matmul_optimization/src/lib/cuda/libtorch
		CUDNN_DIR = ../../../cudnn
# Please modify the paths in the following two lines ('-I/usr/include/python3.8' & '-lpython3.8') according to your Python development environment
#		INCLUDE_DIRS += -I$(CUDA_HOME)/include -I$(TORCH_DIR)/include -I$(TORCH_DIR)/include/torch/csrc/api/include -I$(PYTHON_DIR)/include/python3.8 -I$(CUDNN_DIR)/include
		INCLUDE_DIRS += -I$(CUDA_HOME)/include -I$(CUDA_HOME)/targets/x86_64-linux/include -I$(CUDNN_DIR)/include
		LDFLAGS += -lcublas -lculibos -lcudart -lcublasLt -lpthread -ldl -lrt -lnvrtc -lcuda -lcudnn -lcurand -lcusolver -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/targets/x86_64-linux/lib -L$(CUDNN_DIR)/lib -Xlinker -rpath=$(CUDA_HOME)/lib64 -Xlinker -rpath=$(CUDA_HOME)/targets/x86_64-linux/lib -Xlinker -rpath=$(CUDNN_DIR)/lib
		CXXFLAGS += -Xcompiler "-mavx2" -arch=sm_86
	else
# For Intel machines with AVX
		LIB_AVX_SRC = $(wildcard $(LIB_DIR)/lib/avx/*.cc)
		LIB_SRC += $(LIB_AVX_SRC)
		CXXFLAGS += -mavx2
	endif
else ifeq ($(shell uname -m),aarch64)
	ifdef CUDA_AVAILABLE
# Edge GPUs (e.g., Nvidia Jetson AGX Orin, Jetson Orin Nano) use PyTorch for JetPack
#		PYTHON_DIR = ../../../archiconda3/envs/TinyLLMEngine_py38
#		TORCH_DIR = ../../../archiconda3/envs/TinyLLMEngine_py38/lib/python3.8/site-packages/torch
		CUDNN_DIR = /usr/include/aarch64-linux-gnu
		CUDNN_LIB = /usr/lib/aarch64-linux-gnu
# Please modify the paths in the following two lines ('-I/usr/include/python3.8' & '-lpython3.8') according to your Python development environment
#		INCLUDE_DIRS += -I$(CUDA_HOME)/include -I$(TORCH_DIR)/include -I$(TORCH_DIR)/include/torch/csrc/api/include -I$(PYTHON_DIR)/include/python3.8 -I$(CUDNN_DIR)
		INCLUDE_DIRS += -I$(CUDA_HOME)/include -I$(CUDA_HOME)/targets/aarch64-linux/include -I$(CUDNN_DIR)
#		LDFLAGS += -L$(TORCH_DIR)/lib -L$(PYTHON_DIR)/lib -L$(CUDNN_LIB) -L$(CUDA_HOME)/lib64 -Xlinker -rpath=$(TORCH_DIR)/lib -Xlinker -rpath=$(PYTHON_DIR)/lib -Xlinker -rpath=$(CUDNN_LIB) -Xlinker -rpath=$(CUDA_HOME)/lib64 -lpython3.8 -ltorch -ltorch_cpu -ltorch_cuda -ltorch_python -lnvrtc -lcuda -lcudart -lcudnn -lcudart -lcublas -lcurand -lcusolver -lc10 -lc10_cuda
		LDFLAGS += -lcublas -lculibos -lcudart -lcublasLt -lpthread -ldl -lrt -lnvrtc -lcuda -lcudnn -lcurand -lcusolver -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/targets/aarch64-linux/lib -L$(CUDNN_LIB) -Xlinker -rpath=$(CUDA_HOME)/lib64 -Xlinker -rpath=$(CUDA_HOME)/targets/aarch64-linux/lib -Xlinker -rpath=$(CUDNN_LIB) 
		CXXFLAGS += -arch=sm_87 --forward-unknown-to-host-compiler
	else
# For ARM aarch64 platforms (such as Nvidia Jetson devices) with NEON
		LIB_NEON_SRC = $(wildcard $(LIB_DIR)/lib/neon/*.cc)
		LIB_SRC += $(LIB_NEON_SRC)
		CXXFLAGS += -march=native
	endif
else ifeq ($(shell uname -p),arm)
# For ARM A-series (such as Mac M1) with NEON
	LIB_NEON_SRC = $(wildcard $(LIB_DIR)/lib/neon/*.cc)
	LIB_SRC += $(LIB_NEON_SRC)
	CXXFLAGS += -march=native -framework Accelerate
	INCLUDE_DIRS += -I/opt/homebrew/opt/boost/include
	LIB += -L/opt/homebrew/opt/boost/lib
else
# Use paltform independent implementation
	@echo "Device unsupported! Using the reference implementation will largely impacts the performance.
	LIB_REF_SRC = $(wildcard $(LIB_DIR)/lib/ref/*.cc)
	LIB_SRC += $(LIB_REF_SRC)
endif

# $(info $(LIB_SRC))

SRC_DIR = src
SRC = $(wildcard src/*.cc)
OPS = $(wildcard src/ops/*.cc) $(wildcard src/ops/$(IS_CUDA)/*.cc) 
SRC += $(OPS)
SRC += $(LIB_SRC)


# Default target
all: $(TARGET)

# Linking
test_ops:
ifdef CUDA_AVAILABLE
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -o test_ops tests/$(IS_CUDA)/test_ops.cu $(SRC) $(LIB) $(LDFLAGS)
else
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -o test_ops tests/$(IS_CUDA)/test_ops.cc $(SRC) $(LIB) $(LDFLAGS)
endif

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

ifdef CUDA_AVAILABLE
test_Int4llamaAttention:
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -o test_Int4llamaAttention tests/cuda/test_Int4llamaAttention.cu $(SRC) $(LDFLAGS)

# TODO: Need to fix the following test to cuda version
test_Int4llamaDecoderLayer:
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -o test_Int4llamaDecoderLayer tests/non_cuda/test_Int4llamaDecoderLayer.cc $(SRC) $(LDFLAGS)

# TODO: Need to fix the following test to cuda version
test_Int4llamaDecoder:
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -o test_Int4llamaDecoder tests/non_cuda/test_Int4llamaDecoder.cc $(SRC) $(LDFLAGS)

# TODO: Need to fix the following test to cuda version
test_Int4LlamaForCausalLM:
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -o test_Int4LlamaForCausalLM tests/non_cuda/test_Int4LlamaForCausalLM.cc $(SRC) $(LDFLAGS)

# TODO: Need to fix the following test to cuda version
profile_Int4LlamaForCausalLM:
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -D PROFILER -o profile_Int4LlamaForCausalLM tests/cuda/test_Int4LlamaForCausalLM.cu $(SRC) $(LIB) $(LDFLAGS)
else
test_Int4llamaAttention:
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -o test_Int4llamaAttention tests/non_cuda/test_Int4llamaAttention.cc $(SRC) $(LDFLAGS)

test_Int4llamaDecoderLayer:
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -o test_Int4llamaDecoderLayer tests/non_cuda/test_Int4llamaDecoderLayer.cc $(SRC) $(LDFLAGS)

test_Int4llamaDecoder:
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -o test_Int4llamaDecoder tests/non_cuda/test_Int4llamaDecoder.cc $(SRC) $(LDFLAGS)

test_Int4LlamaForCausalLM:
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -o test_Int4LlamaForCausalLM tests/non_cuda/test_Int4LlamaForCausalLM.cc $(SRC) $(LDFLAGS)

profile_Int4LlamaForCausalLM:
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -D PROFILER -o profile_Int4LlamaForCausalLM tests/non_cuda/test_Int4LlamaForCausalLM.cc $(SRC) $(LIB) $(LDFLAGS)
endif

demo:
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -o demo application/demo.cc $(SRC) $(LIB) $(LDFLAGS)

profile_OPTForCausalLM:
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -D PROFILER -o profile_OPTForCausalLM tests/test_OPTForCausalLM.cc $(SRC) $(LIB) $(LDFLAGS)

profile_Fp32LlamaForCausalLM:
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -D PROFILER -o profile_Fp32LlamaForCausalLM tests/test_Fp32LlamaForCausalLM.cc $(SRC) $(LIB) $(LDFLAGS)

profile_ops:
ifdef CUDA_AVAILABLE
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -D PROFILER -o profile_ops tests/$(IS_CUDA)/test_ops.cu $(SRC) $(LIB) $(LDFLAGS)
else
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -D PROFILER -o profile_ops tests/$(IS_CUDA)/test_ops.cc $(SRC) $(LIB) $(LDFLAGS)
endif

# Clean up
clean:
	rm -f $(TARGET)
