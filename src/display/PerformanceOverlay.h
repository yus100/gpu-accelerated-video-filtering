#ifndef PERFORMANCEOVERLAY_H
#define PERFORMANCEOVERLAY_H

#include <string>
#include <deque>

class PerformanceOverlay {
public:
    PerformanceOverlay();
    
    // update with new frame time (in milliseconds)
    void update(double frameTime);
    
    // get current fps
    double getFPS() const { return currentFPS; }
    
    // get average frame time
    double getAvgFrameTime() const { return avgFrameTime; }
    
    // get min/max frame times
    double getMinFrameTime() const { return minFrameTime; }
    double getMaxFrameTime() const { return maxFrameTime; }
    
    // get formatted statistics string
    std::string getStatsString() const;
    
    // reset statistics
    void reset();
    
private:
    std::deque<double> frameTimes;  // rolling window of frame times
    static const size_t WINDOW_SIZE = 60;  // track last 60 frames
    
    double currentFPS;
    double avgFrameTime;
    double minFrameTime;
    double maxFrameTime;
    
    void recalculateStats();
};

#endif // PERFORMANCEOVERLAY_H

