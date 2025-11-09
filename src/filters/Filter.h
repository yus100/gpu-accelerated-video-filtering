#ifndef FILTER_H
#define FILTER_H

#include "video/Frame.h"
#include <memory>

// base interface for all filters
class Filter {
public:
    virtual ~Filter() = default;
    
    // apply filter to input frame and produce output frame
    // returns processing time in milliseconds
    virtual double apply(const Frame& input, Frame& output) = 0;
    
    // get filter name
    virtual const char* getName() const = 0;
};

#endif // FILTER_H

