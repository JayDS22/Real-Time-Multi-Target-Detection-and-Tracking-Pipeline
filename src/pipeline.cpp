// pipeline.cpp

#include "pipeline.hpp"

namespace imgpipe {

Pipeline::Pipeline(const PipelineConfig& cfg)
    : cfg_(cfg),
      bg_(cfg.bg_alpha, cfg.fg_threshold),
      morph_(cfg.morph_kernel, cfg.morph_open_iter, cfg.morph_close_iter),
      extractor_(cfg.min_area, cfg.max_area),
      tracker_(cfg.max_disappeared, cfg.max_distance)
{}

FrameResult Pipeline::process_frame(const ColorImage& frame) {
    FrameResult result;

    GrayImage fg_mask = bg_.update(frame);
    result.foreground_mask = morph_.apply(fg_mask);
    result.detections = extractor_.extract(result.foreground_mask);

    const auto& trk_map = tracker_.update(result.detections);
    result.tracks = trk_map;

    return result;
}

void Pipeline::reset() {
    bg_.reset();
    tracker_.reset();
}

}  // namespace imgpipe
