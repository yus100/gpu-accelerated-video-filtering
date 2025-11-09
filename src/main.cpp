#include "video/VideoReader.h"
#include "display/Display.h"
#include "display/PerformanceOverlay.h"
#include <iostream>
#include <chrono>
#include <SDL2/SDL.h>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video_file>" << std::endl;
        return 1;
    }
    
    std::string videoPath = argv[1];
    
    // create video reader
    VideoReader reader;
    if (!reader.open(videoPath)) {
        std::cerr << "Failed to open video: " << videoPath << std::endl;
        return 1;
    }
    
    // create display
    Display display;
    if (!display.initialize(reader.getWidth(), reader.getHeight(), "GPU Video Filter")) {
        std::cerr << "Failed to initialize display" << std::endl;
        return 1;
    }
    
    // performance tracking
    PerformanceOverlay perfOverlay;
    
    // timing variables
    double targetFrameTime = 1000.0 / reader.getFPS();  // milliseconds per frame
    auto lastFrameTime = std::chrono::high_resolution_clock::now();
    
    bool running = true;
    bool paused = false;
    
    std::cout << "Controls:" << std::endl;
    std::cout << "  SPACE - Pause/Resume" << std::endl;
    std::cout << "  Q/ESC - Quit" << std::endl;
    std::cout << std::endl;
    
    // main loop
    while (running) {
        // handle events
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            } else if (event.type == SDL_KEYDOWN) {
                switch (event.key.keysym.sym) {
                    case SDLK_q:
                    case SDLK_ESCAPE:
                        running = false;
                        break;
                    case SDLK_SPACE:
                        paused = !paused;
                        std::cout << (paused ? "Paused" : "Resumed") << std::endl;
                        break;
                }
            }
        }
        
        if (!paused) {
            // measure frame processing start
            auto frameStart = std::chrono::high_resolution_clock::now();
            
            // read next frame
            auto frame = reader.readFrame();
            if (!frame) {
                std::cout << "End of video reached" << std::endl;
                running = false;
                break;
            }
            
            // render frame
            display.clear();
            display.render(*frame);
            display.present();
            
            // measure frame processing time
            auto frameEnd = std::chrono::high_resolution_clock::now();
            double frameTime = std::chrono::duration<double, std::milli>(frameEnd - frameStart).count();
            
            // update performance stats
            perfOverlay.update(frameTime);
            
            // print stats occasionally (every 30 frames)
            static int frameCount = 0;
            if (++frameCount % 30 == 0) {
                std::cout << "\r" << perfOverlay.getStatsString() << std::flush;
            }
            
            // maintain target frame rate
            double elapsedTime = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - lastFrameTime
            ).count();
            
            if (elapsedTime < targetFrameTime) {
                int delayMs = static_cast<int>(targetFrameTime - elapsedTime);
                SDL_Delay(delayMs);
            }
            
            lastFrameTime = std::chrono::high_resolution_clock::now();
        } else {
            // when paused, just wait a bit to reduce cpu usage
            SDL_Delay(16);
        }
    }
    
    std::cout << std::endl;
    std::cout << "Final stats: " << perfOverlay.getStatsString() << std::endl;
    
    // cleanup
    display.cleanup();
    reader.close();
    
    std::cout << "Shutdown complete" << std::endl;
    
    return 0;
}

