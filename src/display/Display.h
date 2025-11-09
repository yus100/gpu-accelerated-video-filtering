#ifndef DISPLAY_H
#define DISPLAY_H

#include "video/Frame.h"
#include <string>

// forward declarations
struct SDL_Window;
struct SDL_Renderer;
struct SDL_Texture;

class Display {
public:
    Display();
    ~Display();
    
    // create window and renderer
    bool initialize(int width, int height, const std::string& title = "Video Filter");
    
    // render frame to window
    void render(const Frame& frame);
    
    // clear screen
    void clear();
    
    // present rendered content
    void present();
    
    // cleanup
    void cleanup();
    
    // accessors
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    
private:
    SDL_Window* window;
    SDL_Renderer* renderer;
    SDL_Texture* texture;
    
    int width;
    int height;
};

#endif // DISPLAY_H

