// detection_extractor.cpp
// Two-pass connected component labeling + stats accumulation.
// This is the part that the python version gets "for free" from
// cv2.connectedComponentsWithStats — we have to do it ourselves here
// since the goal is zero external deps.

#include "detection_extractor.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>

namespace imgpipe {

DetectionExtractor::DetectionExtractor(int min_area, int max_area)
    : min_area_(min_area), max_area_(max_area)
{}

int DetectionExtractor::uf_find(std::vector<int>& parent, int x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];  // path halving
        x = parent[x];
    }
    return x;
}

void DetectionExtractor::uf_union(std::vector<int>& parent, std::vector<int>& rank,
                                   int a, int b) {
    int ra = uf_find(parent, a);
    int rb = uf_find(parent, b);
    if (ra == rb) return;
    if (rank[ra] < rank[rb]) std::swap(ra, rb);
    parent[rb] = ra;
    if (rank[ra] == rank[rb]) rank[ra]++;
}

std::vector<int> DetectionExtractor::label_components(const GrayImage& mask,
                                                       int& num_labels) const {
    const int W = mask.width;
    const int H = mask.height;
    std::vector<int> labels(W * H, 0);

    // over-allocate the UF arrays — we'll never have more labels than
    // half the pixels anyway
    int next_label = 1;
    std::vector<int> parent(W * H / 2 + 2);
    std::vector<int> uf_rank(parent.size(), 0);
    std::iota(parent.begin(), parent.end(), 0);

    // pass 1: assign provisional labels, record equivalences
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            if (mask.at(r, c) == 0) continue;

            int idx = r * W + c;
            int up   = (r > 0 && mask.at(r-1, c) != 0) ? labels[(r-1)*W + c] : 0;
            int left = (c > 0 && mask.at(r, c-1) != 0) ? labels[r*W + (c-1)] : 0;

            if (up == 0 && left == 0) {
                labels[idx] = next_label++;
            } else if (up != 0 && left == 0) {
                labels[idx] = up;
            } else if (up == 0 && left != 0) {
                labels[idx] = left;
            } else {
                labels[idx] = std::min(up, left);
                if (up != left)
                    uf_union(parent, uf_rank, up, left);
            }
        }
    }

    // pass 2: flatten through UF roots, remap to consecutive IDs
    std::vector<int> remap(next_label, 0);
    int final_label = 0;
    for (int i = 1; i < next_label; ++i) {
        int root = uf_find(parent, i);
        if (remap[root] == 0)
            remap[root] = ++final_label;
        remap[i] = remap[root];
    }

    for (auto& lbl : labels) {
        if (lbl > 0) lbl = remap[lbl];
    }

    num_labels = final_label + 1;  // +1 for the background
    return labels;
}

std::vector<Detection> DetectionExtractor::extract(const GrayImage& mask) const {
    int num_labels = 0;
    auto labels = label_components(mask, num_labels);

    if (num_labels <= 1) return {};

    const int W = mask.width;
    const int H = mask.height;

    struct ComponentStats {
        int area = 0;
        int min_x = std::numeric_limits<int>::max();
        int min_y = std::numeric_limits<int>::max();
        int max_x = 0;
        int max_y = 0;
        double sum_x = 0.0;
        double sum_y = 0.0;
    };

    std::vector<ComponentStats> stats(num_labels);

    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            int lbl = labels[r * W + c];
            if (lbl == 0) continue;

            auto& s = stats[lbl];
            s.area++;
            s.sum_x += c;
            s.sum_y += r;
            s.min_x = std::min(s.min_x, c);
            s.min_y = std::min(s.min_y, r);
            s.max_x = std::max(s.max_x, c);
            s.max_y = std::max(s.max_y, r);
        }
    }

    std::vector<Detection> detections;
    detections.reserve(num_labels);

    for (int i = 1; i < num_labels; ++i) {
        const auto& s = stats[i];
        if (s.area < min_area_ || s.area > max_area_) continue;

        Detection det;
        det.bbox.x = s.min_x;
        det.bbox.y = s.min_y;
        det.bbox.width  = s.max_x - s.min_x + 1;
        det.bbox.height = s.max_y - s.min_y + 1;
        det.centroid.x = s.sum_x / s.area;
        det.centroid.y = s.sum_y / s.area;
        det.area = static_cast<double>(s.area);

        // same confidence heuristic as the python version — area-based
        // with an aspect ratio penalty
        double aspect = static_cast<double>(det.bbox.width)
                        / std::max(det.bbox.height, 1);
        det.confidence = std::min(1.0, det.area / 5000.0)
                         * std::min(1.0, 1.0 / (std::abs(aspect - 1.0) + 0.5));

        detections.push_back(det);
    }

    return detections;
}

}  // namespace imgpipe
