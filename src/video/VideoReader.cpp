#include "VideoReader.h"
#include <iostream>

// ffmpeg headers
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}

VideoReader::VideoReader() 
    : formatCtx(nullptr), codecCtx(nullptr), avFrame(nullptr), 
      packet(nullptr), swsCtx(nullptr), videoStreamIndex(-1),
      width(0), height(0), fps(0), totalFrames(0), currentFrame(0),
      currentFrameReady(false) {
}

VideoReader::~VideoReader() {
    close();
}

bool VideoReader::open(const std::string& filepath) {
    // open input file
    if (avformat_open_input(&formatCtx, filepath.c_str(), nullptr, nullptr) < 0) {
        std::cerr << "Failed to open video file: " << filepath << std::endl;
        return false;
    }
    
    // retrieve stream information
    if (avformat_find_stream_info(formatCtx, nullptr) < 0) {
        std::cerr << "Failed to find stream info" << std::endl;
        close();
        return false;
    }
    
    // find video stream
    const AVCodec* codec = nullptr;
    videoStreamIndex = av_find_best_stream(formatCtx, AVMEDIA_TYPE_VIDEO, 
                                          -1, -1, &codec, 0);
    if (videoStreamIndex < 0) {
        std::cerr << "Failed to find video stream" << std::endl;
        close();
        return false;
    }
    
    AVStream* videoStream = formatCtx->streams[videoStreamIndex];
    
    // allocate codec context
    codecCtx = avcodec_alloc_context3(codec);
    if (!codecCtx) {
        std::cerr << "Failed to allocate codec context" << std::endl;
        close();
        return false;
    }
    
    // copy codec parameters to context
    if (avcodec_parameters_to_context(codecCtx, videoStream->codecpar) < 0) {
        std::cerr << "Failed to copy codec parameters" << std::endl;
        close();
        return false;
    }
    
    // open codec
    if (avcodec_open2(codecCtx, codec, nullptr) < 0) {
        std::cerr << "Failed to open codec" << std::endl;
        close();
        return false;
    }
    
    // get video properties
    width = codecCtx->width;
    height = codecCtx->height;
    
    // calculate fps
    AVRational framerate = av_guess_frame_rate(formatCtx, videoStream, nullptr);
    fps = framerate.num / (double)framerate.den;
    
    // estimate total frames
    totalFrames = videoStream->nb_frames;
    if (totalFrames <= 0) {
        // fallback if nb_frames not available
        totalFrames = (int64_t)(videoStream->duration * fps / AV_TIME_BASE);
    }
    
    // allocate frame
    avFrame = av_frame_alloc();
    if (!avFrame) {
        std::cerr << "Failed to allocate frame" << std::endl;
        close();
        return false;
    }
    
    // allocate packet
    packet = av_packet_alloc();
    if (!packet) {
        std::cerr << "Failed to allocate packet" << std::endl;
        close();
        return false;
    }
    
    // initialize sws context for RGB conversion
    // we'll convert to RGB24 format (3 channels)
    swsCtx = sws_getContext(
        width, height, codecCtx->pix_fmt,
        width, height, AV_PIX_FMT_RGB24,
        SWS_BILINEAR, nullptr, nullptr, nullptr
    );
    
    if (!swsCtx) {
        std::cerr << "Failed to initialize sws context" << std::endl;
        close();
        return false;
    }
    
    std::cout << "Video opened: " << width << "x" << height 
              << " @ " << fps << " fps" << std::endl;
    
    return true;
}

std::unique_ptr<Frame> VideoReader::readFrame() {
    if (!formatCtx || !codecCtx) {
        return nullptr;
    }
    
    currentFrameReady = false;
    
    // read packets until we get a decoded frame
    while (!currentFrameReady) {
        int ret = av_read_frame(formatCtx, packet);
        
        if (ret < 0) {
            // end of file or error
            // try flushing decoder
            decodePacket(nullptr);
            if (!currentFrameReady) {
                return nullptr;  // really done
            }
            break;
        }
        
        // check if packet is from video stream
        if (packet->stream_index == videoStreamIndex) {
            decodePacket(packet);
        }
        
        av_packet_unref(packet);
        
        if (currentFrameReady) {
            break;
        }
    }
    
    if (!currentFrameReady) {
        return nullptr;
    }
    
    // create our Frame structure
    auto frame = std::make_unique<Frame>(width, height, 3);
    
    // prepare destination buffers for sws_scale
    uint8_t* destData[1] = { frame->data.get() };
    int destLinesize[1] = { width * 3 };
    
    // convert from codec pixel format to RGB24
    sws_scale(
        swsCtx,
        avFrame->data, avFrame->linesize,
        0, height,
        destData, destLinesize
    );
    
    // set timestamp (in milliseconds)
    frame->timestamp = avFrame->pts * av_q2d(formatCtx->streams[videoStreamIndex]->time_base) * 1000;
    
    currentFrame++;
    
    return frame;
}

bool VideoReader::decodePacket(AVPacket* pkt) {
    // send packet to decoder
    int ret = avcodec_send_packet(codecCtx, pkt);
    if (ret < 0) {
        std::cerr << "Error sending packet to decoder" << std::endl;
        return false;
    }
    
    // receive decoded frame
    ret = avcodec_receive_frame(codecCtx, avFrame);
    if (ret == AVERROR(EAGAIN)) {
        // need more packets
        return false;
    } else if (ret == AVERROR_EOF) {
        // end of stream
        return false;
    } else if (ret < 0) {
        std::cerr << "Error receiving frame from decoder" << std::endl;
        return false;
    }
    
    currentFrameReady = true;
    return true;
}

void VideoReader::close() {
    if (swsCtx) {
        sws_freeContext(swsCtx);
        swsCtx = nullptr;
    }
    
    if (avFrame) {
        av_frame_free(&avFrame);
    }
    
    if (packet) {
        av_packet_free(&packet);
    }
    
    if (codecCtx) {
        avcodec_free_context(&codecCtx);
    }
    
    if (formatCtx) {
        avformat_close_input(&formatCtx);
    }
    
    videoStreamIndex = -1;
    width = 0;
    height = 0;
    fps = 0;
    totalFrames = 0;
    currentFrame = 0;
}

