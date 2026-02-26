// tracker.hpp
// Centroid-based multi-object tracker.
// Uses greedy nearest-neighbor assignment — works well enough for
// non-overlapping targets. For crowded scenes with lots of crossings
// you'd want to swap in a Hungarian solver (TODO).

#ifndef IMGPIPELINE_TRACKER_HPP
#define IMGPIPELINE_TRACKER_HPP

#include "types.hpp"
#include <unordered_map>
#include <vector>

namespace imgpipe {

class CentroidTracker {
public:
    CentroidTracker(int max_disappeared = 15, double max_distance = 80.0);

    const std::unordered_map<int, Track>& update(const std::vector<Detection>& detections);
    const std::unordered_map<int, Track>& tracks() const { return tracks_; }
    void reset();

private:
    int next_id_;
    int max_disappeared_;
    double max_distance_;
    std::unordered_map<int, Track> tracks_;

    void register_track(const Point2f& centroid);
    static double dist(const Point2f& a, const Point2f& b);
};

}  // namespace imgpipe

#endif
