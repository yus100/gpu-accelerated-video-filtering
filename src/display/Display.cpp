#include "Display.h"
#include <iostream>
#include <SDL2/SDL.h>

Display::Display() 
    : window(nullptr), renderer(nullptr), texture(nullptr),
      width(0), height(0) {
}

// cleanup
Display::~Display() {
    cleanup();
}

bool Display::initialize(int w, int h, const std::string& title) {
    width = w;
    height = h;
    
    // initialize SDL video subsystem
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL initialization failed: " << SDL_GetError() << std::endl;
        return false;
    }
    
    // create window
    window = SDL_CreateWindow(
        title.c_str(),
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        width, height,
        SDL_WINDOW_SHOWN
    );
    
    if (!window) {
        std::cerr << "Window creation failed: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return false;
    }
    
    // create renderer with hardware acceleration
    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
        std::cerr << "Renderer creation failed: " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return false;
    }
    
    // create texture for streaming video frames
    // RGB24 format
    texture = SDL_CreateTexture(
        renderer,
        SDL_PIXELFORMAT_RGB24,
        SDL_TEXTUREACCESS_STREAMING,
        width, height
    );
    
    if (!texture) {
        std::cerr << "Texture creation failed: " << SDL_GetError() << std::endl;
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return false;
    }
    
    std::cout << "Display initialized: " << width << "x" << height << std::endl;
    
    return true;
}

void Display::render(const Frame& frame) {
    if (!texture) return;
    
    // update texture with frame data
    SDL_UpdateTexture(texture, nullptr, frame.data.get(), frame.width * frame.channels);
    
    // copy texture to renderer
    SDL_RenderCopy(renderer, texture, nullptr, nullptr);
}

void Display::clear() {
    if (renderer) {
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);
    }
}

void Display::present() {
    if (renderer) {
        SDL_RenderPresent(renderer);
    }
}

void Display::cleanup() {
    if (texture) {
        SDL_DestroyTexture(texture);
        texture = nullptr;
    }
    
    if (renderer) {
        SDL_DestroyRenderer(renderer);
        renderer = nullptr;
    }
    
    if (window) {
        SDL_DestroyWindow(window);
        window = nullptr;
    }
    
    SDL_Quit();
}

