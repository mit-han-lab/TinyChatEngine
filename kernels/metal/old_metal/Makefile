CXX = /opt/homebrew/opt/llvm/bin/clang++
CXXFLAGS = -std=c++17 -stdlib=libc++ -O3

# Executable and source files
TEST_TARGET = benchmark
TARGET = $(TEST_TARGET)
KERNEL_SRC = $(wildcard ./src/*.cpp)

SRC = $(KERNEL_SRC)
INCLUDE_DIRS = -I./metal-cpp -I./include
LIB = -framework Metal -framework Foundation -framework MetalKit


# Default target
all: $(TARGET)

# Linking
benchmark: build_metallib
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -o benchmark.x app/main.cpp $(SRC) $(LIB) $(LDFLAGS)

build_air:
	xcrun -sdk macosx metal -ffast-math -fno-fast-math $(INCLUDE_DIRS) -c kernel/op.metal -o library.air

build_metallib: build_air
	xcrun -sdk macosx metallib library.air -o default.metallib

# Clean up
clean:
	rm -f benchmark.x library.air library.metallib default.metallib
