# Build onednn (enable openmp on mac)

cmake ..  -DOpenMP_C_FLAGS="-Xclang -fopenmp -I/usr/local/opt/libomp/include" -DOpenMP_C_LIB_NAMES="libomp" -DDNNL_CPU_RUNTIME=OMP  -DOpenMP_CXX_FLAGS="-Xclang -fopenmp -I/usr/local/opt/libomp/include"  -DOpenMP_CXX_LIB_NAMES="libomp" -DOpenMP_libomp_LIBRARY=/usr/local/opt/libomp/lib/libomp.dylib -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/opt/libomp/lib/ -lomp -Wl,-rpath,/usr/local/opt/libomp/lib/"
