#include "GPUManager.h"
#include <cuda_runtime.h>
#include <iostream>

bool GPUManager::initialize() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    
    if (err != cudaSuccess || deviceCount == 0) {
        std::cerr << "No CUDA-capable devices found" << std::endl;
        return false;
    }
    
    // find device with highest compute capability
    int bestDevice = 0;
    int maxComputeCapability = 0;
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        int computeCapability = prop.major * 10 + prop.minor;
        if (computeCapability > maxComputeCapability) {
            maxComputeCapability = computeCapability;
            bestDevice = i;
        }
    }
    
    cudaSetDevice(bestDevice);
    
    std::cout << "CUDA initialized successfully" << std::endl;
    std::cout << "Using device " << bestDevice << ": " << getDeviceName() << std::endl;
    
    return true;
}

void GPUManager::printDeviceInfo() {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    std::cout << "\n=== GPU Device Info ===" << std::endl;
    std::cout << "Name: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Total Global Memory: " << (prop.totalGlobalMem / (1024*1024)) << " MB" << std::endl;
    std::cout << "Multiprocessors: " << prop.multiProcessorCount << std::endl;
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max Block Dimensions: " << prop.maxThreadsDim[0] << " x " 
              << prop.maxThreadsDim[1] << " x " << prop.maxThreadsDim[2] << std::endl;
    std::cout << "Max Grid Dimensions: " << prop.maxGridSize[0] << " x " 
              << prop.maxGridSize[1] << " x " << prop.maxGridSize[2] << std::endl;
    std::cout << "======================\n" << std::endl;
}

std::string GPUManager::getDeviceName() {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    return std::string(prop.name);
}

size_t GPUManager::getTotalMemory() {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    return prop.totalGlobalMem;
}

size_t GPUManager::getFreeMemory() {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    return free;
}

void GPUManager::cleanup() {
    cudaDeviceReset();
}

