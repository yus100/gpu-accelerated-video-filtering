# GPU-Accelerated Video Filter Pipeline

A real-time video processing application demonstrating GPU acceleration using CUDA. Implements various image filters on both CPU and GPU with performance comparisons.

## Features

- Real-time video playback with FFmpeg
- CPU and GPU implementations of filters:
  - Grayscale conversion
  - Brightness/Contrast adjustment
  - Box blur
- Performance metrics and FPS display
- Side-by-side CPU vs GPU performance comparison

## Requirements

### Hardware
- NVIDIA GPU with compute capability 5.0 or higher
- Minimum 4GB GPU memory

### Software
- CMake 3.18 or later
- C++17 compatible compiler
- CUDA Toolkit 11.0 or later
- FFmpeg development libraries (avformat, avcodec, avutil, swscale)
- SDL2 development library

## Building

### Windows

1. Install CUDA Toolkit from NVIDIA's website
2. Install Visual Studio with C++ support
3. Download FFmpeg development libraries and extract to `C:\ffmpeg`
4. Install SDL2 development libraries

```powershell
mkdir build
cd build
cmake .. -DFFMPEG_INCLUDE_DIR="C:/ffmpeg/include" -DFFMPEG_LIB_DIR="C:/ffmpeg/lib"
cmake --build . --config Release
```

### Linux

1. Install dependencies:
```bash
sudo apt-get install cmake build-essential
sudo apt-get install libavformat-dev libavcodec-dev libavutil-dev libswscale-dev
sudo apt-get install libsdl2-dev
# Install CUDA Toolkit from NVIDIA
```

2. Build:
```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

## Usage

```bash
./video_filter <path_to_video_file>
```

### Controls

- **1, 2, 3**: Switch between filters
  - 1: Grayscale
  - 2: Brightness/Contrast
  - 3: Box Blur
- **0**: No filter (passthrough)
- **M**: Toggle between CPU and GPU processing
- **Space**: Pause/Resume
- **Q** or **ESC**: Quit

## Performance

Expected performance on typical hardware (GTX 1660 Ti, 1080p video):
- CPU: ~10-15 FPS
- GPU: ~150-200 FPS
- Speedup: 10-15x

## Project Structure

```
src/
├── main.cpp                    # Application entry point
├── video/
│   ├── Frame.h                 # Frame data structure
│   └── VideoReader.h/cpp       # FFmpeg video decoder
├── display/
│   ├── Display.h/cpp           # SDL2 rendering
│   └── PerformanceOverlay.h/cpp # FPS overlay
├── filters/
│   ├── Filter.h                # Filter interface
│   ├── CPUFilters.h/cpp        # CPU filter implementations
│   └── GPUFilters.h/cu         # CUDA kernel implementations
└── gpu/
    └── GPUManager.h/cpp        # CUDA memory management
```

## License

Educational project for learning GPU programming and video processing.

