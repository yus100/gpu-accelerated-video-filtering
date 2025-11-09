#ifndef GPUFILTERS_H
#define GPUFILTERS_H

#include "Filter.h"
#include <cstdint>

// forward declarations for cuda kernels
// these will be implemented in GPUFilters.cu

// grayscale filter kernel launcher
void launchGrayscaleKernel(const uint8_t* d_input, uint8_t* d_output, 
                          int width, int height, int channels);

// brightness/contrast filter kernel launcher
void launchBrightnessContrastKernel(const uint8_t* d_input, uint8_t* d_output,
                                   int width, int height, int channels,
                                   float brightness, float contrast);

// box blur filter kernel launcher
void launchBoxBlurKernel(const uint8_t* d_input, uint8_t* d_output,
                        int width, int height, int channels,
                        int kernelSize);

// GPU filter implementations using CUDA
class GrayscaleGPU : public Filter {
public:
    GrayscaleGPU();
    ~GrayscaleGPU();
    
    double apply(const Frame& input, Frame& output) override;
    const char* getName() const override { return "Grayscale (GPU)"; }
    
private:
    uint8_t* d_input;
    uint8_t* d_output;
    size_t allocatedSize;
    
    void allocateMemory(size_t size);
};

class BrightnessContrastGPU : public Filter {
public:
    BrightnessContrastGPU(float brightness = 0.0f, float contrast = 1.0f);
    ~BrightnessContrastGPU();
    
    double apply(const Frame& input, Frame& output) override;
    const char* getName() const override { return "Brightness/Contrast (GPU)"; }
    
    void setBrightness(float b) { brightness = b; }
    void setContrast(float c) { contrast = c; }
    
private:
    float brightness;
    float contrast;
    uint8_t* d_input;
    uint8_t* d_output;
    size_t allocatedSize;
    
    void allocateMemory(size_t size);
};

class BoxBlurGPU : public Filter {
public:
    BoxBlurGPU(int kernelSize = 5);
    ~BoxBlurGPU();
    
    double apply(const Frame& input, Frame& output) override;
    const char* getName() const override { return "Box Blur (GPU)"; }
    
    void setKernelSize(int size) { kernelSize = size; }
    
private:
    int kernelSize;
    uint8_t* d_input;
    uint8_t* d_output;
    size_t allocatedSize;
    
    void allocateMemory(size_t size);
};

#endif // GPUFILTERS_H

