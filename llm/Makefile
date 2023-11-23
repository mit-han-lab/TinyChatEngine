# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++11 -pthread -Ofast

# CUDA flag
DISABLE_CUDA ?= 0
DEC_SHARED_MEM ?= 0

# customize define
DEFINE =
CHATNAME = chat
CXXFLAGS += $(DEFINE)

# Executable and source files
TEST_TARGET_GENERAL = test_Int8OPTAttention test_Int8OPTDecoderLayer test_Int8OPTDecoder test_OPTForCausalLM test_OPTTokenizer test_LLaMATokenizer test_OPTGenerate test_Fp32llamaAttention test_Fp32llamaDecoderLayer test_Fp32llamaDecoder test_Fp32llamaForCausalLM test_Fp32OPTAttention test_Fp32OPTDecoderLayer test_Fp32OPTDecoder test_Fp32OPTForCausalLM
TEST_TARGET_IF_CUDA = test_ops test_Int4llamaAttention test_Int4llamaDecoderLayer test_Int4llamaDecoder test_Int4llamaForCausalLM
PROFILE_TARGET = profile_Fp32llamaForCausalLM profile_Int4llamaForCausalLM profile_OPTForCausalLM profile_ops
CHAT_TARGET = chat
TARGET = $(TEST_TARGET_GENERAL) $(TEST_TARGET_IF_CUDA) $(PROFILE_TARGET) $(CHAT_TARGET)

BUILDDIR := build/transformer
PROFILEDIR := build_profile/transformer
LIB_DIR = ../kernels
LIB_SRC = $(wildcard $(LIB_DIR)/*.cc)
INCLUDE_DIRS = -I$(LIB_DIR) -I./include -I./include/nn_modules -I./json/single_include/ -I./half-2.2.0/include/
LIB =
LDFLAGS =

ifeq ($(DEC_SHARED_MEM), 1)
$(info Shared memory allocation for decoder layers is enabled! (Currently only for CPU, not for CUDA))
	CXXFLAGS += -DDEC_SHARED_MEM
endif

# Check if CUDA is available
ifeq ($(DISABLE_CUDA), 0)
	ifeq ($(OS),Windows_NT)
		CUDA_AVAILABLE := $(shell command -v nvcc 2> /dev/null)
	else
		CUDA_AVAILABLE := $(shell command -v /usr/local/cuda/bin/nvcc 2> /dev/null)
	endif
endif

ifdef CUDA_AVAILABLE
$(info CUDA is available!)
	ifeq ($(OS),Windows_NT)
		CUDA_HOME ?= $(CUDA_PATH)
$(info Detected CUDA_PATH: $(CUDA_HOME))
		ifeq ($(CUDA_HOME),)
			$(error CUDA_PATH is not set. Please set it to your CUDA installation path)
		endif
		CXX = nvcc
# Please modify '-arch=sm_86' according to your GPU architecture/compute capability (https://developer.nvidia.com/cuda-gpus)
		CXXFLAGS = -std=c++17 -O3 -DQM_CUDA -arch=sm_86
		PATH_BRACKET = "
	else
		CUDA_HOME = /usr/local/cuda
		CXX = $(CUDA_HOME)/bin/nvcc
# Please modify 'arch=compute_87,code=sm_87' according to your GPU architecture/compute capability (https://developer.nvidia.com/cuda-gpus)
		CXXFLAGS = -std=c++17 -Xptxas -O3 -gencode arch=compute_87,code=sm_87 --forward-unknown-to-host-compiler -Xcompiler "-pthread" -DQM_CUDA -DENABLE_BF16 -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_BFLOAT16_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__ -U__CUDA_NO_BFLOAT162_OPERATORS__ -U__CUDA_NO_BFLOAT162_CONVERSIONS__ --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math --threads=8
	endif
	# LIB_SRC_CUDA_CC = $(wildcard $(LIB_DIR)/cuda/*.cc) $(wildcard $(LIB_DIR)/cuda/attention/*.cc)
	# LIB_SRC_CUDA_CU = $(wildcard $(LIB_DIR)/cuda/*.cu) $(wildcard $(LIB_DIR)/cuda/attention/*.cu) $(wildcard src/*.cu) $(wildcard src/nn_modules/cuda/*.cu) $(wildcard src/ops/cuda/*.cu)
	# INCLUDE_DIRS += -I./include/ops/cuda -I$(LIB_DIR)/cuda/attention
	LIB_SRC_CUDA_CC = $(wildcard $(LIB_DIR)/cuda/*.cc)
	LIB_SRC_CUDA_CU = $(wildcard $(LIB_DIR)/cuda/*.cu) $(wildcard src/*.cu) $(wildcard src/nn_modules/cuda/*.cu) $(wildcard src/ops/cuda/*.cu)
	INCLUDE_DIRS += -I./include/ops/cuda
else
$(info CUDA is unavailable!)
	LIB_SRC += $(wildcard src/nn_modules/non_cuda/*.cc)
endif

ifeq ($(shell uname -m),x86_64)
	ifdef CUDA_AVAILABLE
		ifeq ($(OS),Windows_NT)
# For Windows platforms with CUDA
			INCLUDE_DIRS += -I"$(CUDA_HOME)/include" -I"$(CUDA_HOME)/targets/x86_64-linux/include"
			LDFLAGS += -lcublas -lcudart -lcublasLt -lnvrtc -lcuda -lcurand -lcusolver -L"$(CUDA_HOME)/lib64" -L"$(CUDA_HOME)/targets/x86_64-linux/lib" -Xlinker -rpath="$(CUDA_HOME)/lib64" -Xlinker -rpath="$(CUDA_HOME)/targets/x86_64-linux/lib"
		else
			CXXFLAGS += -Xcompiler "-mavx2" -mfma -ffast-math -fpermissive
			INCLUDE_DIRS += -I$(CUDA_HOME)/include -I$(CUDA_HOME)/targets/x86_64-linux/include -I/usr/include/x86_64-linux-gnu
			LDFLAGS += -lcublas -lculibos -lcudart -lcublasLt -lpthread -ldl -lrt -lnvrtc -lcuda -lcurand -lcusolver -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/targets/x86_64-linux/lib -L/usr/lib/x86_64-linux-gnu -Xlinker -rpath=$(CUDA_HOME)/lib64 -Xlinker -rpath=$(CUDA_HOME)/targets/x86_64-linux/lib -Xlinker -rpath=/usr/lib/x86_64-linux-gnu
		endif
	else
# For x86_64 platforms with AVX2
		# For Intel machines with AVX
		LIB_AVX_SRC = $(wildcard $(LIB_DIR)/avx/*.cc)
		LIB_SRC += $(LIB_AVX_SRC)
		CXXFLAGS += -mavx2 -mfma -ffast-math -DUSE_INT8_INT4_PRODUCT -fpermissive -DQM_x86
	endif
else ifeq ($(shell uname -m),aarch64)
	ifdef CUDA_AVAILABLE
# For ARM aarch64 platforms with CUDA
		INCLUDE_DIRS += -I$(CUDA_HOME)/include -I$(CUDA_HOME)/targets/aarch64-linux/include -I/usr/include/aarch64-linux-gnu
		LDFLAGS += -lcublas -lculibos -lcudart -lcublasLt -lpthread -ldl -lrt -lnvrtc -lcuda -lcudnn -lcurand -lcusolver -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/targets/aarch64-linux/lib -L/usr/lib/aarch64-linux-gnu -Xlinker -rpath=$(CUDA_HOME)/lib64 -Xlinker -rpath=$(CUDA_HOME)/targets/aarch64-linux/lib -Xlinker -rpath=/usr/lib/aarch64-linux-gnu
		CXXFLAGS += -fPIC
	else
# For ARM aarch64 platforms with NEON
		LIB_NEON_SRC = $(wildcard $(LIB_DIR)/neon/*.cc)
		LIB_SRC += $(LIB_NEON_SRC)
		CXXFLAGS += -march=native -DUSE_INT8_INT4_PRODUCT -DQM_ARM -fPIC
	endif
else ifeq ($(shell uname -p),arm)
	CXX = /opt/homebrew/opt/llvm/bin/clang++
	LIB += -L/opt/homebrew/opt/boost/lib
	# For ARM A-series (such as Mac M1) with Metal GPU
	ifdef USE_METAL
		LIB_ACC_INC = -I$(LIB_DIR)/metal/include -I$(LIB_DIR)/metal/metal-cpp -I$(LIB_DIR)/metal
		LIB_SRC += $(wildcard $(LIB_DIR)/metal/*.cc)
		INCLUDE_DIRS += -I/opt/homebrew/opt/boost/include $(LIB_ACC_INC)
		LIB += -framework Metal -framework Foundation -framework MetalKit
		TARGET += default.metallib library.air
		CXXFLAGS += -std=c++17 -stdlib=libc++ -DQM_METAL
	# Use NEON with int8 runtime quantization is faster
	else
		LIB_SRC += $(wildcard $(LIB_DIR)/neon/*.cc)
		CXXFLAGS += -march=native -DUSE_INT8_INT4_PRODUCT -DQM_ARM -fPIC -march=armv8.2-a
		INCLUDE_DIRS += -I/opt/homebrew/opt/boost/include
	endif
else ifneq ($(or $(filter armv6%,$(UNAME_M)), $(filter armv7%,$(UNAME_M)), $(filter armv8%,$(UNAME_M))),)
	LIB_NEON_SRC = $(wildcard $(LIB_DIR)/neon/*.cc)
	LIB_SRC += $(LIB_NEON_SRC)
	CXXFLAGS += -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access -DUSE_INT8_INT4_PRODUCT -DQM_ARM -fPIC
else
	# Use paltform independent implementation
	# @echo "Device unsupported! Using the reference implementation will largely impacts the performance.
	LIB_REF_SRC = $(wildcard $(LIB_DIR)/ref/*.cc)
	LIB_SRC += $(LIB_REF_SRC)
endif

# $(info $(LIB_SRC))

SRC_DIR = src
SRC = $(wildcard src/*.cc)
SRC += $(wildcard src/nn_modules/*.cc)
OPS =  $(wildcard src/ops/*.cc)
SRC += $(OPS)
SRC += $(LIB_SRC)
SRC_CUDA_CC = $(LIB_SRC_CUDA_CC)
SRC_CUDA_CU = $(LIB_SRC_CUDA_CU)

# Default target
all: $(TARGET)

# Phony targets
.PHONY: all clean

# Metal lib
library.air: $(LIB_DIR)/metal/kernel/op.metal
	xcrun -sdk macosx metal -ffast-math -fno-fast-math $(LIB_ACC_INC) -c $< -o library.air
default.metallib: library.air
	xcrun -sdk macosx metallib library.air -o default.metallib

$(info $(SRC))
OBJS = $(addprefix $(BUILDDIR)/,$(SRC:.cc=.o))
PROFILE_OBJS = $(addprefix $(PROFILEDIR)/,$(SRC:.cc=.o))

# Pattern rules for CUDA
ifdef CUDA_AVAILABLE
$(info $(SRC_CUDA_CC))
OBJS += $(addprefix $(BUILDDIR)/,$(SRC_CUDA_CC:.cc=.o))
PROFILE_OBJS += $(addprefix $(PROFILEDIR)/,$(SRC_CUDA_CC:.cc=.o))

$(info $(SRC_CUDA_CU))
OBJS += $(addprefix $(BUILDDIR)/,$(SRC_CUDA_CU:.cu=.o))
PROFILE_OBJS += $(addprefix $(PROFILEDIR)/,$(SRC_CUDA_CU:.cu=.o))

$(BUILDDIR)/%.o: %.cu
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -c $< -o $@

$(PROFILEDIR)/%.o: %.cu
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -DPROFILER -c $< -o $@
endif

# Pattern rules for non-CUDA
$(BUILDDIR)/%.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -c $< -o $@

$(PROFILEDIR)/%.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -DPROFILER -c $< -o $@

# Linking
# Rule for TEST_TARGET
$(TEST_TARGET_GENERAL): %: tests/%.cc $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -o $@ $^ $(LIB) $(LDFLAGS)

ifdef CUDA_AVAILABLE
$(TEST_TARGET_IF_CUDA): %: tests/cuda/%.cu $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -o $@ $^ $(LIB) $(LDFLAGS)
else
$(TEST_TARGET_IF_CUDA): %: tests/non_cuda/%.cc $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -o $@ $^ $(LIB) $(LDFLAGS)
endif


# Rule for PROFILE_TARGET
# Here we define explicit rules for each profile target
profile_Fp32llamaForCausalLM: tests/test_Fp32llamaForCausalLM.cc $(PROFILE_OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -DPROFILER -o $@ $^ $(LIB) $(LDFLAGS)

profile_OPTForCausalLM: tests/test_OPTForCausalLM.cc $(PROFILE_OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -DPROFILER -o $@ $^ $(LIB) $(LDFLAGS)

ifdef CUDA_AVAILABLE
profile_Int4llamaForCausalLM: tests/cuda/test_Int4llamaForCausalLM.cu $(PROFILE_OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -DPROFILER -o $@ $^ $(LIB) $(LDFLAGS)

profile_ops: tests/cuda/test_ops.cu $(PROFILE_OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -DPROFILER -o $@ $^ $(LIB) $(LDFLAGS)
else
profile_Int4llamaForCausalLM: tests/non_cuda/test_Int4llamaForCausalLM.cc $(PROFILE_OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -DPROFILER -o $@ $^ $(LIB) $(LDFLAGS)

profile_ops: tests/non_cuda/test_ops.cc $(PROFILE_OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -DPROFILER -o $@ $^ $(LIB) $(LDFLAGS)
endif

# Rule for CHAT_TARGET
$(CHAT_TARGET): %: application/%.cc $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -o $(CHATNAME) $^ $(LIB) $(LDFLAGS)

# Clean up
clean:
	rm -f $(TARGET)
	rm -rf *.dSYM
	rm -rf $(BUILDDIR)/../matmul_optimization
	rm -rf $(BUILDDIR)/$(LIB_DIR)
	rm -rf $(BUILDDIR)
	rm -rf $(PROFILEDIR)/$(LIB_DIR)
	rm -rf $(PROFILEDIR)
