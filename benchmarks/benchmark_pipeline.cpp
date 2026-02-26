// benchmark_pipeline.cpp
// Feeds synthetic moving-blob frames into the pipeline and measures
// per-frame timing. Compare the output against the python prototype's
// benchmark_pipeline() to see the speedup.
//
// usage: ./benchmark_pipeline [width] [height] [num_frames]

#include "pipeline.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

using Clock = std::chrono::high_resolution_clock;

// fast LCG for generating noise — don't need anything fancy here
class FastRNG {
public:
    explicit FastRNG(uint64_t seed) : state_(seed) {}
    uint32_t next() {
        state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<uint32_t>(state_ >> 32);
    }
    double uniform(double lo, double hi) {
        return lo + (next() / 4294967296.0) * (hi - lo);
    }
    int uniform_int(int lo, int hi) {
        return lo + static_cast<int>(next() % (hi - lo + 1));
    }
private:
    uint64_t state_;
};

static void draw_circle(imgpipe::ColorImage& img, int cx, int cy, int radius,
                         uint8_t b, uint8_t g, uint8_t r) {
    int r2 = radius * radius;
    int y0 = std::max(0, cy - radius);
    int y1 = std::min(img.height - 1, cy + radius);
    int x0 = std::max(0, cx - radius);
    int x1 = std::min(img.width - 1, cx + radius);
    for (int y = y0; y <= y1; ++y)
        for (int x = x0; x <= x1; ++x)
            if ((x-cx)*(x-cx) + (y-cy)*(y-cy) <= r2) {
                img.at(y, x, 0) = b;
                img.at(y, x, 1) = g;
                img.at(y, x, 2) = r;
            }
}

int main(int argc, char* argv[]) {
    int width      = (argc > 1) ? std::atoi(argv[1]) : 640;
    int height     = (argc > 2) ? std::atoi(argv[2]) : 480;
    int num_frames = (argc > 3) ? std::atoi(argv[3]) : 500;

    std::cout << "=== Image Pipeline Benchmark ===\n";
    std::cout << "Resolution: " << width << "x" << height << "\n";
    std::cout << "Frames: " << num_frames << "\n\n";

    imgpipe::PipelineConfig cfg;
    cfg.min_area = 50;
    imgpipe::Pipeline pipeline(cfg);

    FastRNG rng(42);

    // set up some bouncing blobs as targets
    const int n_targets = 5;
    struct Target { double x, y, vx, vy; int radius; };
    std::vector<Target> targets(n_targets);
    for (auto& t : targets) {
        t.x = rng.uniform(50, width - 50);
        t.y = rng.uniform(50, height - 50);
        t.vx = rng.uniform(-3.0, 3.0);
        t.vy = rng.uniform(-3.0, 3.0);
        t.radius = rng.uniform_int(10, 25);
    }

    // static background texture
    imgpipe::ColorImage bg_frame(width, height);
    for (size_t i = 0; i < bg_frame.data.size(); ++i)
        bg_frame.data[i] = static_cast<uint8_t>(rng.uniform_int(40, 80));

    std::vector<double> frame_times_ms;
    frame_times_ms.reserve(num_frames);
    int last_det_count = 0;
    int last_trk_count = 0;

    for (int f = 0; f < num_frames; ++f) {
        imgpipe::ColorImage frame = bg_frame;

        for (auto& t : targets) {
            t.x += t.vx;
            t.y += t.vy;
            if (t.x < t.radius || t.x > width - t.radius)  { t.vx = -t.vx; t.x += t.vx; }
            if (t.y < t.radius || t.y > height - t.radius) { t.vy = -t.vy; t.y += t.vy; }
            draw_circle(frame, (int)t.x, (int)t.y, t.radius, 200, 200, 200);
        }

        // gaussian-ish noise
        for (size_t i = 0; i < frame.data.size(); ++i) {
            int noise = rng.uniform_int(-10, 10);
            int val = static_cast<int>(frame.data[i]) + noise;
            frame.data[i] = static_cast<uint8_t>(std::clamp(val, 0, 255));
        }

        auto t0 = Clock::now();
        auto result = pipeline.process_frame(frame);
        auto t1 = Clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        frame_times_ms.push_back(ms);

        last_det_count = static_cast<int>(result.detections.size());
        last_trk_count = static_cast<int>(result.tracks.size());
    }

    // throw away the first few frames — bg model is still converging
    const int warmup = std::min(20, num_frames / 4);
    std::vector<double> steady(frame_times_ms.begin() + warmup, frame_times_ms.end());

    double sum    = std::accumulate(steady.begin(), steady.end(), 0.0);
    double avg_ms = sum / steady.size();
    double fps    = 1000.0 / avg_ms;

    std::sort(steady.begin(), steady.end());
    double p50 = steady[steady.size() / 2];
    double p95 = steady[static_cast<size_t>(steady.size() * 0.95)];
    double p99 = steady[static_cast<size_t>(steady.size() * 0.99)];

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Results (excluding " << warmup << " warmup frames):\n";
    std::cout << "  Average frame time:  " << avg_ms << " ms\n";
    std::cout << "  Throughput:          " << fps << " FPS\n";
    std::cout << "  Latency P50:         " << p50 << " ms\n";
    std::cout << "  Latency P95:         " << p95 << " ms\n";
    std::cout << "  Latency P99:         " << p99 << " ms\n";
    std::cout << "  Min / Max:           " << steady.front() << " / " << steady.back() << " ms\n";
    std::cout << "  Last frame dets:     " << last_det_count << "\n";
    std::cout << "  Last frame tracks:   " << last_trk_count << "\n";

    return 0;
}
