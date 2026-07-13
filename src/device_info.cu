#include <iostream>
#include <cuda_runtime.h>

#include "cuda_check.h"

int main() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count <= 0) {
        std::cerr << "No CUDA device found.\n";
        return EXIT_FAILURE;
    }

    std::cout << "CUDA device count: " << device_count << "\n\n";

    for (int device = 0; device < device_count; ++device) {
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

        std::cout << "Device " << device << ": " << prop.name << "\n";
        std::cout << "  Compute capability: "
                  << prop.major << "." << prop.minor << "\n";
        std::cout << "  Global memory: "
                  << prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)
                  << " GiB\n";
        std::cout << "  SM count: "
                  << prop.multiProcessorCount << "\n";
        std::cout << "  Warp size: "
                  << prop.warpSize << "\n";
        std::cout << "  Max threads per block: "
                  << prop.maxThreadsPerBlock << "\n";
        std::cout << "  Shared memory per block: "
                  << prop.sharedMemPerBlock / 1024.0
                  << " KiB\n";
        std::cout << "  Registers per block: "
                  << prop.regsPerBlock << "\n";
        std::cout << "  Memory bus width: "
                  << prop.memoryBusWidth << " bits\n";
        std::cout << "  Memory clock rate: "
                  << prop.memoryClockRate / 1000.0
                  << " MHz\n\n";
    }

    return EXIT_SUCCESS;
}
