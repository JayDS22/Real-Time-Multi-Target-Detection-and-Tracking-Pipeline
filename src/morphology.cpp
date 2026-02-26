// morphology.cpp
// Erode/dilate with a circular structuring element.
// The naive nested loop is obviously O(n*k) per pixel — for larger
// kernels we'd want separable approximations or distance transforms,
// but for a 5x5 disk this is fine.

#include "morphology.hpp"
#include <cmath>

namespace imgpipe {

MorphologyFilter::MorphologyFilter(int kernel_size, int open_iter, int close_iter)
    : radius_(kernel_size / 2),
      open_iter_(open_iter),
      close_iter_(close_iter)
{
    build_kernel(kernel_size);
}

void MorphologyFilter::build_kernel(int kernel_size) {
    int r = kernel_size / 2;
    double r_sq = static_cast<double>(r) * r;

    kernel_offsets_.clear();
    for (int dy = -r; dy <= r; ++dy) {
        for (int dx = -r; dx <= r; ++dx) {
            if (dx * dx + dy * dy <= r_sq)
                kernel_offsets_.emplace_back(dy, dx);
        }
    }
}

GrayImage MorphologyFilter::erode(const GrayImage& img) const {
    GrayImage out(img.width, img.height);
    // pixel is 255 only if every kernel neighbor is also 255
    for (int row = 0; row < img.height; ++row) {
        for (int col = 0; col < img.width; ++col) {
            bool all_set = true;
            for (const auto& [dy, dx] : kernel_offsets_) {
                int nr = row + dy;
                int nc = col + dx;
                if (nr < 0 || nr >= img.height || nc < 0 || nc >= img.width) {
                    all_set = false;
                    break;
                }
                if (img.at(nr, nc) == 0) {
                    all_set = false;
                    break;
                }
            }
            out.at(row, col) = all_set ? 255 : 0;
        }
    }
    return out;
}

GrayImage MorphologyFilter::dilate(const GrayImage& img) const {
    GrayImage out(img.width, img.height);
    // pixel is 255 if any kernel neighbor is 255
    for (int row = 0; row < img.height; ++row) {
        for (int col = 0; col < img.width; ++col) {
            bool any_set = false;
            for (const auto& [dy, dx] : kernel_offsets_) {
                int nr = row + dy;
                int nc = col + dx;
                if (nr < 0 || nr >= img.height || nc < 0 || nc >= img.width)
                    continue;
                if (img.at(nr, nc) != 0) {
                    any_set = true;
                    break;
                }
            }
            out.at(row, col) = any_set ? 255 : 0;
        }
    }
    return out;
}

GrayImage MorphologyFilter::apply(const GrayImage& mask) const {
    // opening = erode then dilate — kills small speckles
    GrayImage result = mask;
    for (int i = 0; i < open_iter_; ++i)
        result = erode(result);
    for (int i = 0; i < open_iter_; ++i)
        result = dilate(result);

    // closing = dilate then erode — fills small gaps
    for (int i = 0; i < close_iter_; ++i)
        result = dilate(result);
    for (int i = 0; i < close_iter_; ++i)
        result = erode(result);

    return result;
}

}  // namespace imgpipe
