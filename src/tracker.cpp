// tracker.cpp
// Greedy centroid tracker — pretty much a direct port of the python version.
// Sort all (track, detection) pairs by euclidean distance, greedily assign.

#include "tracker.hpp"
#include <cmath>
#include <algorithm>

namespace imgpipe {

CentroidTracker::CentroidTracker(int max_disappeared, double max_distance)
    : next_id_(0),
      max_disappeared_(max_disappeared),
      max_distance_(max_distance)
{}

double CentroidTracker::dist(const Point2f& a, const Point2f& b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

void CentroidTracker::register_track(const Point2f& centroid) {
    Track t;
    t.id = next_id_++;
    t.centroid = centroid;
    t.age = 0;
    t.frames_since_seen = 0;
    t.history.push_back(centroid);
    tracks_[t.id] = std::move(t);
}

const std::unordered_map<int, Track>&
CentroidTracker::update(const std::vector<Detection>& detections) {

    // nothing detected — bump the disappeared counter on everything
    if (detections.empty()) {
        std::vector<int> to_remove;
        for (auto& [tid, track] : tracks_) {
            track.frames_since_seen++;
            if (track.frames_since_seen > max_disappeared_)
                to_remove.push_back(tid);
        }
        for (int tid : to_remove)
            tracks_.erase(tid);
        return tracks_;
    }

    // no tracks yet — register everything we see
    if (tracks_.empty()) {
        for (const auto& det : detections)
            register_track(det.centroid);
        return tracks_;
    }

    // build flat arrays for the assignment
    std::vector<int> track_ids;
    std::vector<Point2f> track_pts;
    track_ids.reserve(tracks_.size());
    track_pts.reserve(tracks_.size());
    for (const auto& [tid, track] : tracks_) {
        track_ids.push_back(tid);
        track_pts.push_back(track.centroid);
    }

    const size_t n_tracks = track_ids.size();
    const size_t n_dets   = detections.size();

    // compute all pairwise distances, sort ascending
    struct DistEntry {
        size_t t_idx;
        size_t d_idx;
        double distance;
    };

    std::vector<DistEntry> entries;
    entries.reserve(n_tracks * n_dets);
    for (size_t ti = 0; ti < n_tracks; ++ti) {
        for (size_t di = 0; di < n_dets; ++di) {
            double d = dist(track_pts[ti], detections[di].centroid);
            entries.push_back({ti, di, d});
        }
    }

    std::sort(entries.begin(), entries.end(),
              [](const DistEntry& a, const DistEntry& b) {
                  return a.distance < b.distance;
              });

    // greedy matching
    std::vector<bool> assigned_t(n_tracks, false);
    std::vector<bool> assigned_d(n_dets, false);

    for (const auto& e : entries) {
        if (e.distance > max_distance_) break;
        if (assigned_t[e.t_idx] || assigned_d[e.d_idx]) continue;

        int tid = track_ids[e.t_idx];
        auto& trk = tracks_[tid];
        trk.centroid = detections[e.d_idx].centroid;
        trk.frames_since_seen = 0;
        trk.age++;
        trk.history.push_back(trk.centroid);

        assigned_t[e.t_idx] = true;
        assigned_d[e.d_idx] = true;
    }

    // drop tracks that disappeared too long ago
    std::vector<int> to_remove;
    for (size_t ti = 0; ti < n_tracks; ++ti) {
        if (!assigned_t[ti]) {
            int tid = track_ids[ti];
            tracks_[tid].frames_since_seen++;
            if (tracks_[tid].frames_since_seen > max_disappeared_)
                to_remove.push_back(tid);
        }
    }
    for (int tid : to_remove)
        tracks_.erase(tid);

    // anything we didn't match is a new track
    for (size_t di = 0; di < n_dets; ++di) {
        if (!assigned_d[di])
            register_track(detections[di].centroid);
    }

    return tracks_;
}

void CentroidTracker::reset() {
    tracks_.clear();
    next_id_ = 0;
}

}  // namespace imgpipe
