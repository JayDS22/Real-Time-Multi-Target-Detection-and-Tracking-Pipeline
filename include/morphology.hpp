// morphology.hpp
// Binary morphological ops for cleaning up the fg mask.
// Does the same thing as cv2.morphologyEx in the prototype,
// but we roll our own circular kernel here.

#ifndef IMGPIPELINE_MORPHOLOGY_HPP
#define IMGPIPELINE_MORPHOLOGY_HPP

#include "types.hpp"
#include <vector>

namespace imgpipe {

class MorphologyFilter {
public:
    MorphologyFilter(int kernel_size = 5, int open_iter = 2, int close_iter = 2);

    // open then close — removes noise, fills small holes
    GrayImage apply(const GrayImage& mask) const;

    // exposed individually for unit testing
    GrayImage erode(const GrayImage& img) const;
    GrayImage dilate(const GrayImage& img) const;

private:
    int radius_;
    int open_iter_;
    int close_iter_;

    // precomputed (row_off, col_off) pairs for the disk kernel
    std::vector<std::pair<int,int>> kernel_offsets_;

    void build_kernel(int kernel_size);
};

}  // namespace imgpipe

#endif
