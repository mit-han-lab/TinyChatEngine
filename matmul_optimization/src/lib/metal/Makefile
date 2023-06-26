CXX = /opt/homebrew/opt/llvm/bin/clang++
CXXFLAGS = -std=c++17 -stdlib=libc++ -O3

# Executable and source files
TEST_TARGET = benchmark
TARGET = $(TEST_TARGET)

SRC = 01-MetalAdder/main.cpp 01-MetalAdder/MetalMatmul.cpp 01-MetalAdder/MetalMatmulInt4.cpp
INCLUDE_DIRS = -I./metal-cpp -fno-objc-arc
LIB = -L/opt/homebrew/opt/libomp/lib -fopenmp -framework Metal -framework Foundation -framework MetalKit


# Default target
all: $(TARGET)

# Linking
benchmark: build_metallib
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -o 01-MetalAdder/benchmark.x $(SRC) $(LIB) $(LDFLAGS)

build_air:
	xcrun -sdk macosx metal -ffast-math -fno-fast-math -c 01-MetalAdder/op.metal -o 01-MetalAdder/library.air

build_metallib: build_air
	xcrun -sdk macosx metallib 01-MetalAdder/library.air -o 01-MetalAdder/default.metallib

# Clean up
clean:
	rm -f 01-MetalAdder/benchmark.x 01-MetalAdder/library.air 01-MetalAdder/library.metallib 01-MetalAdder/default.metallib
