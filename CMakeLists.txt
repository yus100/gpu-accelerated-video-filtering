cmake_minimum_required(VERSION 3.18)
project(GPUVideoFilter LANGUAGES CXX CUDA)

# set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA setting
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# find required packages
find_package(SDL2 REQUIRED)
find_package(CUDAToolkit REQUIRED)

# FFmpeg - manual search since there's no official CMake find module
# Users may need to set FFMPEG_DIR or adjust paths
if(WIN32)
    # typical Windows installation paths
    set(FFMPEG_INCLUDE_DIR "C:/ffmpeg/include" CACHE PATH "FFmpeg include directory")
    set(FFMPEG_LIB_DIR "C:/ffmpeg/lib" CACHE PATH "FFmpeg library directory")
else()
    # typical Linux paths (usually in system directories)
    set(FFMPEG_INCLUDE_DIR "/usr/include" CACHE PATH "FFmpeg include directory")
    set(FFMPEG_LIB_DIR "/usr/lib" CACHE PATH "FFmpeg library directory")
endif()

include_directories(${FFMPEG_INCLUDE_DIR})
link_directories(${FFMPEG_LIB_DIR})

# source files
set(SOURCES
    src/main.cpp
    src/video/VideoReader.cpp
    src/display/Display.cpp
    src/display/PerformanceOverlay.cpp
    src/filters/CPUFilters.cpp
    src/gpu/GPUManager.cpp
)

# CUDA kernel files
set(CUDA_SOURCES
    src/filters/GPUFilters.cu
)

# create executable
add_executable(video_filter ${SOURCES} ${CUDA_SOURCES})

# include directories
target_include_directories(video_filter PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${SDL2_INCLUDE_DIRS}
)

# link libraries
if(WIN32)
    target_link_libraries(video_filter PRIVATE
        ${SDL2_LIBRARIES}
        CUDA::cudart
        avformat
        avcodec
        avutil
        swscale
    )
else()
    target_link_libraries(video_filter PRIVATE
        SDL2::SDL2
        CUDA::cudart
        avformat
        avcodec
        avutil
        swscale
    )
endif()

# CUDA architecture
set_target_properties(video_filter PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "50;61;75;86"  # support for compute capability 5.0+
)

# Platform specific settings
if(WIN32)
    # copy SDL2 DLL to output directory on Windows
    if(TARGET SDL2::SDL2)
        add_custom_command(TARGET video_filter POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:SDL2::SDL2>
            $<TARGET_FILE_DIR:video_filter>
        )
    endif()
endif()

