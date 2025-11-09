#include "PerformanceOverlay.h"
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <limits>

PerformanceOverlay::PerformanceOverlay() 
    : currentFPS(0.0), avgFrameTime(0.0), 
      minFrameTime(std::numeric_limits<double>::max()),
      maxFrameTime(0.0) {
}

void PerformanceOverlay::update(double frameTime) {
    frameTimes.push_back(frameTime);
    
    // keep window size limited
    if (frameTimes.size() > WINDOW_SIZE) {
        frameTimes.pop_front();
    }
    
    recalculateStats();
}

void PerformanceOverlay::recalculateStats() {
    if (frameTimes.empty()) {
        currentFPS = 0.0;
        avgFrameTime = 0.0;
        return;
    }
    
    // calculate average
    avgFrameTime = std::accumulate(frameTimes.begin(), frameTimes.end(), 0.0) / frameTimes.size();
    
    // calculate fps (careful with zero)
    currentFPS = avgFrameTime > 0.0 ? 1000.0 / avgFrameTime : 0.0;
    
    // find min/max
    auto minmax = std::minmax_element(frameTimes.begin(), frameTimes.end());
    minFrameTime = *minmax.first;
    maxFrameTime = *minmax.second;
}

std::string PerformanceOverlay::getStatsString() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1);
    oss << "FPS: " << currentFPS 
        << " | Frame: " << avgFrameTime << "ms"
        << " (min: " << minFrameTime << "ms, max: " << maxFrameTime << "ms)";
    return oss.str();
}

void PerformanceOverlay::reset() {
    frameTimes.clear();
    currentFPS = 0.0;
    avgFrameTime = 0.0;
    minFrameTime = std::numeric_limits<double>::max();
    maxFrameTime = 0.0;
}

