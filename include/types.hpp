/*
 * types.hpp
 * Shared structs for the pipeline. Everything lives here so we don't
 * end up with header dependency spaghetti later.
 */

#ifndef IMGPIPELINE_TYPES_HPP
#define IMGPIPELINE_TYPES_HPP

#include <cstdint>
#include <vector>
#include <utility>

namespace imgpipe {

struct BBox {
    int x, y, width, height;
};

struct Point2f {
    double x;
    double y;

    Point2f() : x(0.0), y(0.0) {}
    Point2f(double x_, double y_) : x(x_), y(y_) {}
};

struct Detection {
    BBox bbox;
    Point2f centroid;
    double area;
    double confidence;
};

struct Track {
    int id;
    Point2f centroid;
    int age;                       // how many frames we've been tracking this
    int frames_since_seen;         // goes up when we don't find a match
    std::vector<Point2f> history;  // for drawing the trail
};

// simple grayscale wrapper, row-major, one byte per pixel
struct GrayImage {
    int width;
    int height;
    std::vector<uint8_t> data;

    GrayImage() : width(0), height(0) {}
    GrayImage(int w, int h) : width(w), height(h), data(w * h, 0) {}

    uint8_t& at(int row, int col)       { return data[row * width + col]; }
    uint8_t  at(int row, int col) const { return data[row * width + col]; }

    size_t pixel_count() const { return static_cast<size_t>(width) * height; }
};

// BGR interleaved, matches OpenCV's default layout so conversion is trivial
struct ColorImage {
    int width;
    int height;
    std::vector<uint8_t> data;

    ColorImage() : width(0), height(0) {}
    ColorImage(int w, int h) : width(w), height(h), data(3 * w * h, 0) {}

    uint8_t& at(int row, int col, int ch) {
        return data[(row * width + col) * 3 + ch];
    }
    uint8_t at(int row, int col, int ch) const {
        return data[(row * width + col) * 3 + ch];
    }
};

// tunables — these defaults match what the python prototype uses
struct PipelineConfig {
    double bg_alpha        = 0.02;
    int    morph_kernel    = 5;
    int    morph_open_iter = 2;
    int    morph_close_iter= 2;
    int    min_area        = 100;
    int    max_area        = 50000;
    int    max_disappeared = 15;
    double max_distance    = 80.0;
    int    fg_threshold    = 25;
};

}  // namespace imgpipe

#endif  // IMGPIPELINE_TYPES_HPP
