#include "utils.h"
class MemoryAllocator {
    // TODO: use allocate_aligned_memory instead!
   public:
    MemoryAllocator() { this->counter = 0; }
    float* get_fpbuffer(int size) {
        float* ptr;
        allocate_aligned_memory(ptr, size * sizeof(float));
        return ptr;
    }
    int8_t* get_int8buffer(int size) {
        int8_t* ptr;
        allocate_aligned_memory(ptr, size * sizeof(int8_t));
        return ptr;
    }
    int* get_intbuffer(int size) {
        int* ptr;
        allocate_aligned_memory(ptr, size * sizeof(int));
        return ptr;
    }

   private:
    int counter;
};
