// background_model.cpp
// Exponentially-weighted running average for background estimation.
// Port of the python prototype — main speedup comes from avoiding
// per-pixel interpreter overhead.

#include "background_model.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace imgpipe {

BackgroundModel::BackgroundModel(double alpha, int threshold)
    : alpha_(alpha), threshold_(threshold), initialized_(false)
{}

void BackgroundModel::bgr_to_gray(const ColorImage& src, GrayImage& dst) {
    dst = GrayImage(src.width, src.height);
    const size_t npx = dst.pixel_count();

    // BT.601 weights: 0.114*B + 0.587*G + 0.299*R
    // scaled by 1024 so we can use shifts instead of floats:
    //   B=117, G=601, R=306  (sum=1024)
    for (size_t i = 0; i < npx; ++i) {
        int b = src.data[i * 3 + 0];
        int g = src.data[i * 3 + 1];
        int r = src.data[i * 3 + 2];
        dst.data[i] = static_cast<uint8_t>((117 * b + 601 * g + 306 * r) >> 10);
    }
}

GrayImage BackgroundModel::update(const ColorImage& frame) {
    GrayImage gray;
    bgr_to_gray(frame, gray);

    const size_t npx = gray.pixel_count();

    if (!initialized_) {
        // first frame just seeds the accumulator, nothing to subtract yet
        accumulator_.resize(npx);
        for (size_t i = 0; i < npx; ++i)
            accumulator_[i] = static_cast<double>(gray.data[i]);
        initialized_ = true;
        return GrayImage(gray.width, gray.height);
    }

    GrayImage mask(gray.width, gray.height);

    const double a = alpha_;
    const double inv_a = 1.0 - a;
    const int thresh = threshold_;

    // this loop is the hottest path in the pipeline, would benefit
    // from SSE/AVX if we ever need more throughput
    for (size_t i = 0; i < npx; ++i) {
        double val = static_cast<double>(gray.data[i]);
        accumulator_[i] = a * val + inv_a * accumulator_[i];

        int diff = static_cast<int>(std::abs(val - accumulator_[i]));
        mask.data[i] = (diff > thresh) ? 255 : 0;
    }

    return mask;
}

void BackgroundModel::reset() {
    initialized_ = false;
    accumulator_.clear();
}

}  // namespace imgpipe
