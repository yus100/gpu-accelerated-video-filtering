#ifndef FRAME_H
#define FRAME_H

#include <cstdint>
#include <memory>

// simple structure to hold raw frame data
struct Frame {
    int width;
    int height;
    int channels;  // 3 for RGB, 4 for RGBA
    int64_t timestamp;  // in milliseconds
    std::unique_ptr<uint8_t[]> data;
    
    Frame(int w, int h, int ch) 
        : width(w), height(h), channels(ch), timestamp(0) {
        data = std::make_unique<uint8_t[]>(w * h * ch);
    }
    
    // copy constructor
    Frame(const Frame& other) 
        : width(other.width), height(other.height), 
          channels(other.channels), timestamp(other.timestamp) {
        size_t size = width * height * channels;
        data = std::make_unique<uint8_t[]>(size);
        std::copy(other.data.get(), other.data.get() + size, data.get());
    }
    
    // size in bytes
    size_t size() const {
        return width * height * channels;
    }
    
    // prevent accidental copies (they're expensive)
    Frame& operator=(const Frame&) = delete;
};

#endif // FRAME_H

