CXX = /opt/homebrew/opt/llvm/bin/clang++
CXXFLAGS = -std=c++17 -stdlib=libc++ -O3

# Executable and source files
TEST_TARGET = benchmark
TARGET = $(TEST_TARGET)

SRC = example_kernel/main.cpp example_kernel/MetalMatmul.cpp example_kernel/MetalMatmulInt4.cpp
INCLUDE_DIRS = -I./metal-cpp -fno-objc-arc
LIB = -L/opt/homebrew/opt/libomp/lib -fopenmp -framework Metal -framework Foundation -framework MetalKit


# Default target
all: $(TARGET)

# Linking
benchmark: build_metallib
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -o example_kernel/benchmark.x $(SRC) $(LIB) $(LDFLAGS)

build_air:
	xcrun -sdk macosx metal -ffast-math -fno-fast-math -c example_kernel/op.metal -o example_kernel/library.air

build_metallib: build_air
	xcrun -sdk macosx metallib example_kernel/library.air -o example_kernel/default.metallib

# Clean up
clean:
	rm -f example_kernel/benchmark.x example_kernel/library.air example_kernel/library.metallib example_kernel/default.metallib
