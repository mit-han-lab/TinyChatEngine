# Check operating system
OS := $(shell uname)

# OneDNN availability
ONEDNN_AVAILABLE =
ifeq ($(OS), Darwin)  # macOS
    $(info Detected macOS)
    ONEDNN_AVAILABLE := $(shell otool -L /usr/local/lib/libdnnl* 2> /dev/null)
else ifeq ($(OS), Linux)  # Ubuntu or other Linux distributions
    $(info Detected Linux)
    ONEDNN_AVAILABLE_CHK := $(shell pkg-config --exists dnnl; echo $$?)
    ifeq ($(ONEDNN_AVAILABLE_CHK), 0)
        ONEDNN_AVAILABLE := $(shell pkg-config --exists onednn 2> /dev/null) # TODO: check this in Linux env
	endif
else
    $(error Unsupported operating system)
endif

# Check if CUDA is available
CUDA_AVAILABLE := $(shell command -v /usr/local/cuda/bin/nvcc 2> /dev/null)

CC_FLAGS = -O3 -std=c++11 #-g
#CC_FLAGS = -O3 -std=c++11 -Xclang -fopenmp -g
# Compiler and flags
ifdef CUDA_AVAILABLE
	CC = /usr/local/cuda/bin/nvcc
	CC_FLAGS += -DCUDA_ENABLE
    $(info CUDA is available)
else
    CC = g++
    CC_FLAGS += -mavx2 -mfma
endif
ifdef ONEDNN_AVAILABLE
	CC_FLAGS += -DONEDNN_ENABLE
    $(info ONEDNN is available)
endif

# Include directories
# INCLUDE_DIRS = -I./ -I/usr/local/opt/libomp/include
INCLUDE_DIRS = -I./

# Library directories
LIBRARY_DIRS = -L/usr/local/cuda/lib64

# Library flag
LDFLAGS =
ifdef ONEDNN_AVAILABLE
LDFLAGS += -ldnnl
endif

# TODO: openmp flag
OMP_FLAGS =  -L/usr/local/opt/libomp/lib/ -lomp
# LDFLAGS += $(OMP_FLAGS

# Files
TARGET = benchmark_run
CUDA_SRCS = lib/matmul.cu
CPP_SRCS = benchmark/main.cc lib/matmul_imp.cc lib/utils.cc lib/matmul_int8.cc lib/matmul_avx_int8.cc
ONEDNN_SRCS = lib/matmul_onednn.cc

# Objects
OBJS = $(CPP_SRCS:.cc=.o)
INT8_OBJS = $(INT8_CPP_SRCS:.cc=.o)
ifdef CUDA_AVAILABLE
OBJS += $(CUDA_SRCS:.cu=.o)
endif
ifdef ONEDNN_AVAILABLE
OBJS += $(ONEDNN_SRCS:.cc=.o)
INT8_OBJS += $(ONEDNN_SRCS:.cc=.o)
endif


# $(info ONEDNN_AVAILABLE: $(ONEDNN_AVAILABLE))
$(info CC_FLAGS: $(CC_FLAGS))


# Targets
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CC_FLAGS) $(INCLUDE_DIRS) $(LDFLAGS) -o $(TARGET) $(OBJS)

%.o: %.cu
	$(CC) $(CC_FLAGS) $(INCLUDE_DIRS) $(LDFLAGS) -c $< -o $@

ifdef CUDA_AVAILABLE
%.o: %.cc
	$(CC) $(CC_FLAGS) $(INCLUDE_DIRS) $(LDFLAGS) -x cu -c $< -o $@
else
%.o: %.cc
	$(CC) $(CC_FLAGS) $(INCLUDE_DIRS) $(LDFLAGS) -c $< -o $@
    #$(CC) $(CC_FLAGS) $(INCLUDE_DIRS) $(LDFLAGS) -c $< -o $@ $(OMP_FLAGS)
endif

clean:
	rm -f $(TARGET) $(OBJS)
