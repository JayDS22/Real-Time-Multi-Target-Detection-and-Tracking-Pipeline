// pipeline.hpp
// Chains the stages together: bg subtract -> morph -> detect -> track.
// Mirrors DetectionPipeline from the python prototype.

#ifndef IMGPIPELINE_PIPELINE_HPP
#define IMGPIPELINE_PIPELINE_HPP

#include "types.hpp"
#include "background_model.hpp"
#include "morphology.hpp"
#include "detection_extractor.hpp"
#include "tracker.hpp"
#include <vector>
#include <unordered_map>

namespace imgpipe {

struct FrameResult {
    std::vector<Detection> detections;
    std::unordered_map<int, Track> tracks;
    GrayImage foreground_mask;
};

class Pipeline {
public:
    explicit Pipeline(const PipelineConfig& cfg = PipelineConfig());

    FrameResult process_frame(const ColorImage& frame);
    void reset();
    const PipelineConfig& config() const { return cfg_; }

private:
    PipelineConfig cfg_;
    BackgroundModel bg_;
    MorphologyFilter morph_;
    DetectionExtractor extractor_;
    CentroidTracker tracker_;
};

}  // namespace imgpipe

#endif
