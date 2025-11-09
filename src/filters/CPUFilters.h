#ifndef CPUFILTERS_H
#define CPUFILTERS_H

#include "Filter.h"

// grayscale filter using luminosity method
class GrayscaleCPU : public Filter {
public:
    double apply(const Frame& input, Frame& output) override;
    const char* getName() const override { return "Grayscale (CPU)"; }
};

// brightness and contrast adjustment
class BrightnessContrastCPU : public Filter {
public:
    BrightnessContrastCPU(float brightness = 0.0f, float contrast = 1.0f);
    
    double apply(const Frame& input, Frame& output) override;
    const char* getName() const override { return "Brightness/Contrast (CPU)"; }
    
    void setBrightness(float b) { brightness = b; }
    void setContrast(float c) { contrast = c; }
    
private:
    float brightness;  // -100 to +100
    float contrast;    // 0.5 to 2.0
};

// box blur filter
class BoxBlurCPU : public Filter {
public:
    BoxBlurCPU(int kernelSize = 5);
    
    double apply(const Frame& input, Frame& output) override;
    const char* getName() const override { return "Box Blur (CPU)"; }
    
    void setKernelSize(int size) { kernelSize = size; }
    
private:
    int kernelSize;  // must be odd (3, 5, 7, etc)
};

#endif // CPUFILTERS_H

