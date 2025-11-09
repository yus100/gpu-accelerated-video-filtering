#ifndef VIDEOREADER_H
#define VIDEOREADER_H

#include "Frame.h"
#include <string>
#include <memory>

// forward declarations to avoid including ffmpeg headers here
struct AVFormatContext;
struct AVCodecContext;
struct AVFrame;
struct AVPacket;
struct SwsContext;

class VideoReader {
public:
    VideoReader();
    ~VideoReader();
    
    // open video file and initialize decoder
    bool open(const std::string& filepath);
    
    // read next frame from video
    // returns nullptr if no more frames or error
    std::unique_ptr<Frame> readFrame();
    
    // video properties
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    double getFPS() const { return fps; }
    int64_t getTotalFrames() const { return totalFrames; }
    
    // cleanup resources
    void close();
    
private:
    AVFormatContext* formatCtx;
    AVCodecContext* codecCtx;
    AVFrame* avFrame;
    AVPacket* packet;
    SwsContext* swsCtx;
    
    int videoStreamIndex;
    int width;
    int height;
    double fps;
    int64_t totalFrames;
    int64_t currentFrame;
    
    // decode one packet and return frame if available
    bool decodePacket(AVPacket* pkt);
    bool currentFrameReady;
};

#endif // VIDEOREADER_H

