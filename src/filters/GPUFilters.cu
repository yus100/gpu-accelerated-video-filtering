#include "GPUFilters.h"
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

// cuda error checking macro
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
}

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void grayscaleKernel(const uint8_t* input, uint8_t* output,
                               int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = (y * width + x) * channels;
    
    uint8_t r = input[idx];
    uint8_t g = input[idx + 1];
    uint8_t b = input[idx + 2];
    
    // luminosity method
    uint8_t gray = static_cast<uint8_t>(0.299f * r + 0.587f * g + 0.114f * b);
    
    output[idx] = gray;
    output[idx + 1] = gray;
    output[idx + 2] = gray;
}

__global__ void brightnessContrastKernel(const uint8_t* input, uint8_t* output,
                                        int width, int height, int channels,
                                        float brightness, float contrast) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int pixelIdx = y * width + x;
    
    for (int c = 0; c < channels; c++) {
        int idx = pixelIdx * channels + c;
        
        float value = input[idx];
        value = contrast * (value - 128.0f) + 128.0f + brightness;
        
        // clamp
        value = fminf(fmaxf(value, 0.0f), 255.0f);
        
        output[idx] = static_cast<uint8_t>(value);
    }
}

__global__ void boxBlurKernel(const uint8_t* input, uint8_t* output,
                             int width, int height, int channels,
                             int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int halfKernel = kernelSize / 2;
    float sumR = 0, sumG = 0, sumB = 0;
    int count = 0;
    
    // average surrounding pixels
    for (int ky = -halfKernel; ky <= halfKernel; ky++) {
        for (int kx = -halfKernel; kx <= halfKernel; kx++) {
            int nx = x + kx;
            int ny = y + ky;
            
            // clamp to boundaries
            nx = max(0, min(nx, width - 1));
            ny = max(0, min(ny, height - 1));
            
            int idx = (ny * width + nx) * channels;
            sumR += input[idx];
            sumG += input[idx + 1];
            sumB += input[idx + 2];
            count++;
        }
    }
    
    int outIdx = (y * width + x) * channels;
    output[outIdx] = static_cast<uint8_t>(sumR / count);
    output[outIdx + 1] = static_cast<uint8_t>(sumG / count);
    output[outIdx + 2] = static_cast<uint8_t>(sumB / count);
}

// ============================================================================
// Kernel Launchers
// ============================================================================

void launchGrayscaleKernel(const uint8_t* d_input, uint8_t* d_output,
                          int width, int height, int channels) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    grayscaleKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels);
    CUDA_CHECK(cudaGetLastError());
}

void launchBrightnessContrastKernel(const uint8_t* d_input, uint8_t* d_output,
                                   int width, int height, int channels,
                                   float brightness, float contrast) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    brightnessContrastKernel<<<gridSize, blockSize>>>(d_input, d_output, 
                                                      width, height, channels,
                                                      brightness, contrast);
    CUDA_CHECK(cudaGetLastError());
}

void launchBoxBlurKernel(const uint8_t* d_input, uint8_t* d_output,
                        int width, int height, int channels,
                        int kernelSize) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    boxBlurKernel<<<gridSize, blockSize>>>(d_input, d_output, 
                                          width, height, channels, kernelSize);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// GPU Filter Classes
// ============================================================================

GrayscaleGPU::GrayscaleGPU() : d_input(nullptr), d_output(nullptr), allocatedSize(0) {
}

GrayscaleGPU::~GrayscaleGPU() {
    if (d_input) CUDA_CHECK(cudaFree(d_input));
    if (d_output) CUDA_CHECK(cudaFree(d_output));
}

void GrayscaleGPU::allocateMemory(size_t size) {
    if (size > allocatedSize) {
        if (d_input) CUDA_CHECK(cudaFree(d_input));
        if (d_output) CUDA_CHECK(cudaFree(d_output));
        
        CUDA_CHECK(cudaMalloc(&d_input, size));
        CUDA_CHECK(cudaMalloc(&d_output, size));
        allocatedSize = size;
    }
}

double GrayscaleGPU::apply(const Frame& input, Frame& output) {
    auto start = std::chrono::high_resolution_clock::now();
    
    size_t size = input.size();
    allocateMemory(size);
    
    // copy to device
    CUDA_CHECK(cudaMemcpy(d_input, input.data.get(), size, cudaMemcpyHostToDevice));
    
    // launch kernel
    launchGrayscaleKernel(d_input, d_output, input.width, input.height, input.channels);
    
    // synchronize
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // copy back to host
    CUDA_CHECK(cudaMemcpy(output.data.get(), d_output, size, cudaMemcpyDeviceToHost));
    
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// brightness/contrast gpu
BrightnessContrastGPU::BrightnessContrastGPU(float brightness, float contrast)
    : brightness(brightness), contrast(contrast),
      d_input(nullptr), d_output(nullptr), allocatedSize(0) {
}

BrightnessContrastGPU::~BrightnessContrastGPU() {
    if (d_input) CUDA_CHECK(cudaFree(d_input));
    if (d_output) CUDA_CHECK(cudaFree(d_output));
}

void BrightnessContrastGPU::allocateMemory(size_t size) {
    if (size > allocatedSize) {
        if (d_input) CUDA_CHECK(cudaFree(d_input));
        if (d_output) CUDA_CHECK(cudaFree(d_output));
        
        CUDA_CHECK(cudaMalloc(&d_input, size));
        CUDA_CHECK(cudaMalloc(&d_output, size));
        allocatedSize = size;
    }
}

double BrightnessContrastGPU::apply(const Frame& input, Frame& output) {
    auto start = std::chrono::high_resolution_clock::now();
    
    size_t size = input.size();
    allocateMemory(size);
    
    CUDA_CHECK(cudaMemcpy(d_input, input.data.get(), size, cudaMemcpyHostToDevice));
    
    launchBrightnessContrastKernel(d_input, d_output, input.width, input.height,
                                   input.channels, brightness, contrast);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(output.data.get(), d_output, size, cudaMemcpyDeviceToHost));
    
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// box blur gpu
BoxBlurGPU::BoxBlurGPU(int kernelSize)
    : kernelSize(kernelSize), d_input(nullptr), d_output(nullptr), allocatedSize(0) {
    if (kernelSize % 2 == 0) kernelSize++;
}

BoxBlurGPU::~BoxBlurGPU() {
    if (d_input) CUDA_CHECK(cudaFree(d_input));
    if (d_output) CUDA_CHECK(cudaFree(d_output));
}

void BoxBlurGPU::allocateMemory(size_t size) {
    if (size > allocatedSize) {
        if (d_input) CUDA_CHECK(cudaFree(d_input));
        if (d_output) CUDA_CHECK(cudaFree(d_output));
        
        CUDA_CHECK(cudaMalloc(&d_input, size));
        CUDA_CHECK(cudaMalloc(&d_output, size));
        allocatedSize = size;
    }
}

double BoxBlurGPU::apply(const Frame& input, Frame& output) {
    auto start = std::chrono::high_resolution_clock::now();
    
    size_t size = input.size();
    allocateMemory(size);
    
    CUDA_CHECK(cudaMemcpy(d_input, input.data.get(), size, cudaMemcpyHostToDevice));
    
    launchBoxBlurKernel(d_input, d_output, input.width, input.height,
                       input.channels, kernelSize);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(output.data.get(), d_output, size, cudaMemcpyDeviceToHost));
    
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

