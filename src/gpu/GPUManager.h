#ifndef GPUMANAGER_H
#define GPUMANAGER_H

#include <string>

// gpu device management and utilities
class GPUManager {
public:
    // initialize cuda and select best device
    static bool initialize();
    
    // print gpu info
    static void printDeviceInfo();
    
    // get device properties
    static std::string getDeviceName();
    static size_t getTotalMemory();
    static size_t getFreeMemory();
    
    // cleanup
    static void cleanup();
};

#endif // GPUMANAGER_H

