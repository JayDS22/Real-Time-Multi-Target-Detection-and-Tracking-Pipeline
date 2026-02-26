// detection_extractor.hpp
// Blob detection via connected-component labeling.
// Replaces cv2.connectedComponentsWithStats from the prototype
// with a two-pass union-find approach.

#ifndef IMGPIPELINE_DETECTION_EXTRACTOR_HPP
#define IMGPIPELINE_DETECTION_EXTRACTOR_HPP

#include "types.hpp"
#include <vector>

namespace imgpipe {

class DetectionExtractor {
public:
    DetectionExtractor(int min_area = 100, int max_area = 50000);

    std::vector<Detection> extract(const GrayImage& mask) const;

private:
    int min_area_;
    int max_area_;

    // classic two-pass CCL with union-find
    std::vector<int> label_components(const GrayImage& mask, int& num_labels) const;

    static int uf_find(std::vector<int>& parent, int x);
    static void uf_union(std::vector<int>& parent, std::vector<int>& rank, int a, int b);
};

}  // namespace imgpipe

#endif
