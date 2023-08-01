# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++11 -pthread -Ofast

# Executable and source files
TEST_TARGET = test_ops test_Int8OPTAttention test_Int8OPTDecoderLayer test_Int8OPTDecoder test_OPTForCausalLM test_OPTTokenizer test_LLaMATokenizer test_OPTGenerate test_Fp32llamaAttention test_Fp32llamaDecoderLayer test_Fp32llamaDecoder test_Fp32llamaForCausalLM test_Int4llamaAttention test_Int4llamaDecoderLayer test_Int4llamaDecoder test_Int4llamaForCausalLM
PROFILE_TARGET = profile_Fp32llamaForCausalLM profile_Int4llamaForCausalLM profile_OPTForCausalLM profile_ops
APP_TARGET = chat
TARGET = $(TEST_TARGET) $(PROFILE_TARGET) $(APP_TARGET)

BUILDDIR := build/transformer
PROFILEDIR := build_profile/transformer
LIB_DIR = ../kernels
LIB_SRC = $(wildcard $(LIB_DIR)/*.cc)
INCLUDE_DIRS = -I$(LIB_DIR) -I./include -I./include/nn_modules -I./json/single_include/
LIB =

ifeq ($(shell uname -m),x86_64)
	# For Intel machines with AVX
	LIB_AVX_SRC = $(wildcard $(LIB_DIR)/avx/*.cc)
	LIB_SRC += $(LIB_AVX_SRC)
	CXXFLAGS += -mavx2 -mfma -ffast-math -DUSE_INT8_INT4_PRODUCT -fpermissive -DQM_x86
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
profile_Fp32llamaForCausalLM: tests/test_Fp32llamaForCausalLM.cc $(PROFILE_OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -DPROFILER -o $@ $^ $(LIB) $(LDFLAGS)

profile_Int4llamaForCausalLM: tests/test_Int4llamaForCausalLM.cc $(PROFILE_OBJS)
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
	rm -rf $(BUILDDIR)/../matmul_optimization
	rm -rf $(BUILDDIR)
	rm -rf $(PROFILEDIR)
