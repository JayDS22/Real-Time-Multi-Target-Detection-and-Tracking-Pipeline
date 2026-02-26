/*
 * background_model.hpp
 * Running-average background subtraction. Basically the same idea as
 * the python prototype's BackgroundModel, but without pulling in OpenCV.
 */

#ifndef IMGPIPELINE_BACKGROUND_MODEL_HPP
#define IMGPIPELINE_BACKGROUND_MODEL_HPP

#include "types.hpp"
#include <vector>

namespace imgpipe {

class BackgroundModel {
public:
    explicit BackgroundModel(double alpha = 0.02, int threshold = 25);

    // feed a new frame, get back a binary fg mask (255/0).
    // first call just seeds the model and returns all zeros.
    GrayImage update(const ColorImage& frame);

    void reset();
    bool is_initialized() const { return initialized_; }

private:
    double alpha_;
    int threshold_;
    bool initialized_;
    std::vector<double> accumulator_;

    // BT.601 luminance, done in fixed-point to avoid float overhead
    static void bgr_to_gray(const ColorImage& src, GrayImage& dst);
};

}  // namespace imgpipe

#endif  // IMGPIPELINE_BACKGROUND_MODEL_HPP
