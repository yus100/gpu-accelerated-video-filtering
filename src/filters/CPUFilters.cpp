#include "CPUFilters.h"
#include <chrono>
#include <algorithm>
#include <cmath>

// ============================================================================
// Grayscale CPU Implementation
// ============================================================================

double GrayscaleCPU::apply(const Frame& input, Frame& output) {
    auto start = std::chrono::high_resolution_clock::now();
    
    int pixels = input.width * input.height;
    
    for (int i = 0; i < pixels; i++) {
        int idx = i * input.channels;
        uint8_t r = input.data[idx];
        uint8_t g = input.data[idx + 1];
        uint8_t b = input.data[idx + 2];
        
        // luminosity method: Y = 0.299*R + 0.587*G + 0.114*B
        uint8_t gray = static_cast<uint8_t>(0.299f * r + 0.587f * g + 0.114f * b);
        
        // write to output
        int outIdx = i * output.channels;
        output.data[outIdx] = gray;
        output.data[outIdx + 1] = gray;
        output.data[outIdx + 2] = gray;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// ============================================================================
// Brightness/Contrast CPU Implementation
// ============================================================================

BrightnessContrastCPU::BrightnessContrastCPU(float brightness, float contrast)
    : brightness(brightness), contrast(contrast) {
}

double BrightnessContrastCPU::apply(const Frame& input, Frame& output) {
    auto start = std::chrono::high_resolution_clock::now();
    
    int pixels = input.width * input.height;
    
    for (int i = 0; i < pixels; i++) {
        for (int c = 0; c < input.channels; c++) {
            int idx = i * input.channels + c;
            
            // apply contrast and brightness
            // formula: out = contrast * (in - 128) + 128 + brightness
            float value = input.data[idx];
            value = contrast * (value - 128.0f) + 128.0f + brightness;
            
            // clamp to valid range
            value = std::clamp(value, 0.0f, 255.0f);
            
            output.data[idx] = static_cast<uint8_t>(value);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// ============================================================================
// Box Blur CPU Implementation
// ============================================================================

BoxBlurCPU::BoxBlurCPU(int kernelSize) 
    : kernelSize(kernelSize) {
    // ensure kernel size is odd
    if (kernelSize % 2 == 0) {
        kernelSize++;
    }
}

double BoxBlurCPU::apply(const Frame& input, Frame& output) {
    auto start = std::chrono::high_resolution_clock::now();
    
    int halfKernel = kernelSize / 2;
    
    // process each pixel
    for (int y = 0; y < input.height; y++) {
        for (int x = 0; x < input.width; x++) {
            float sumR = 0, sumG = 0, sumB = 0;
            int count = 0;
            
            // average surrounding pixels
            for (int ky = -halfKernel; ky <= halfKernel; ky++) {
                for (int kx = -halfKernel; kx <= halfKernel; kx++) {
                    int nx = x + kx;
                    int ny = y + ky;
                    
                    // clamp to image boundaries
                    nx = std::clamp(nx, 0, input.width - 1);
                    ny = std::clamp(ny, 0, input.height - 1);
                    
                    int idx = (ny * input.width + nx) * input.channels;
                    sumR += input.data[idx];
                    sumG += input.data[idx + 1];
                    sumB += input.data[idx + 2];
                    count++;
                }
            }
            
            // write averaged value
            int outIdx = (y * output.width + x) * output.channels;
            output.data[outIdx] = static_cast<uint8_t>(sumR / count);
            output.data[outIdx + 1] = static_cast<uint8_t>(sumG / count);
            output.data[outIdx + 2] = static_cast<uint8_t>(sumB / count);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

