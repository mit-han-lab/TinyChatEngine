# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -pthread -O3

# Executable and source files
TEST_TARGET = test_ops test_Int8OPTAttention test_Int8OPTDecoderLayer test_Int8OPTDecoder test_OPTForCausalLM test_OPTTokenizer test_LLaMATokenizer test_OPTGenerate test_Fp32llamaAttention test_Fp32llamaDecoderLayer test_Fp32llamaDecoder test_Fp32LlamaForCausalLM test_Int4llamaAttention test_Int4llamaDecoderLayer test_Int4llamaDecoder test_Int4LlamaForCausalLM
PROFILE_TARGET = profile_Fp32LlamaForCausalLM profile_Int4LlamaForCausalLM profile_OPTForCausalLM profile_ops
APP_TARGET = demo
TARGET = $(TEST_TARGET) $(PROFILE_TARGET) $(APP_TARGET)

BUILDDIR := build/transformer
PROFILEDIR := build_profile/transformer
LIB_DIR = ../matmul_optimization/src
LIB_SRC = $(wildcard $(LIB_DIR)/lib/*.cc)
INCLUDE_DIRS = -I$(LIB_DIR) -I./include -I./json/single_include/
LIB =

ifeq ($(shell uname -m),x86_64)
	# For Intel machines with AVX
	LIB_AVX_SRC = $(wildcard $(LIB_DIR)/lib/avx/*.cc)
	LIB_SRC += $(LIB_AVX_SRC)
	CXXFLAGS += -mavx2 -mfma -ffast-math -DUSE_INT8_INT4_PRODUCT
else ifeq ($(shell uname -p),arm)
	# For ARM A-series (such as Mac M1) with Metal GPU
	LIB_ACC_INC = -I$(LIB_DIR)/lib/metal/include -I$(LIB_DIR)/lib/metal/metal-cpp -I$(LIB_DIR)/lib/metal
	LIB_SRC += $(wildcard $(LIB_DIR)/lib/metal/*.cc)
	INCLUDE_DIRS += -I/opt/homebrew/opt/boost/include $(LIB_ACC_INC)
	LIB += -framework Metal -framework Foundation -framework MetalKit
	TARGET += default.metallib library.air
	CXXFLAGS += -stdlib=libc++ -DUSE_METAL
	CXX = /opt/homebrew/opt/llvm/bin/clang++
	# TODO: use NEON if Metal is not supported
	# LIB_SRC += $(wildcard $(LIB_DIR)/lib/neon/*.cc)
	# CXXFLAGS += -march=native -framework Accelerate
	# INCLUDE_DIRS += -I/opt/homebrew/opt/boost/include
	# LIB += -L/opt/homebrew/opt/boost/lib
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
TORCH_DIR = ../matmul_optimization/src/lib/cuda/libtorch
LDFLAGS =

ifdef CUDA_AVAILABLE
    $(info CUDA is available)
	CXX = /usr/local/cuda/bin/nvcc
# Please modify '-arch=sm_86' according to your GPU architecture/compute capability (https://developer.nvidia.com/cuda-gpus)
	CXXFLAGS = -std=c++17 -O3 -DCUDA_ENABLE -Xcompiler "-mavx2" -Xcompiler "-pthread" -arch=sm_86
	SRC += $(CUDA_SRCS)
# Please modify the paths in the following two lines ('-I/usr/include/python3.8' & '-lpython3.8') according to your Python development environment
	INCLUDE_DIRS += -I/usr/local/cuda/include -I$(TORCH_DIR)/include -I$(TORCH_DIR)/include/torch/csrc/api/include -I/usr/include/python3.8
	LDFLAGS += -L$(TORCH_DIR)/lib -ltorch -lc10 -lcudart -L/usr/local/cuda/lib64 -Xlinker -rpath=$(TORCH_DIR)/lib -lpython3.8
endif

# Default target
all: $(TARGET)

# Phony targets
.PHONY: all clean

# Metal lib
library.air: $(LIB_DIR)/lib/metal/kernel/op.metal
	xcrun -sdk macosx metal -ffast-math -fno-fast-math $(LIB_ACC_INC) -c $< -o library.air
default.metallib: library.air
	xcrun -sdk macosx metallib library.air -o default.metallib

$(info $(SRC))
OBJS = $(addprefix $(BUILDDIR)/,$(SRC:.cc=.o))
PROFILE_OBJS = $(addprefix $(PROFILEDIR)/,$(SRC:.cc=.o))

# Pattern rules
$(BUILDDIR)/%.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -c $< -o $@

$(PROFILEDIR)/%.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -DPROFILER -c $< -o $@

# Linking
# Rule for TEST_TARGET
$(TEST_TARGET): %: tests/%.cc $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -o $@ $^ $(LIB) $(LDFLAGS)

# Rule for PROFILE_TARGET
# Here we define explicit rules for each profile target
profile_Fp32LlamaForCausalLM: tests/test_Fp32LlamaForCausalLM.cc $(PROFILE_OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -DPROFILER -o $@ $^ $(LIB) $(LDFLAGS)

profile_Int4LlamaForCausalLM: tests/test_Int4LlamaForCausalLM.cc $(PROFILE_OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -DPROFILER -o $@ $^ $(LIB) $(LDFLAGS)

profile_OPTForCausalLM: tests/test_OPTForCausalLM.cc $(PROFILE_OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -DPROFILER -o $@ $^ $(LIB) $(LDFLAGS)

profile_ops: tests/test_ops.cc $(PROFILE_OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -DPROFILER -o $@ $^ $(LIB) $(LDFLAGS)

# Rule for APP_TARGET
$(APP_TARGET): %: application/%.cc $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -o $@ $^ $(LIB) $(LDFLAGS)

# Clean up
clean:
	rm -f $(TARGET)
	rm -rf *.dSYM
	rm -rf $(BUILDDIR)
	rm -rf $(PROFILEDIR)
