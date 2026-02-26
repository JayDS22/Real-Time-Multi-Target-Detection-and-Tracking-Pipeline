// main.cpp
// Runs the pipeline on synthetic moving-blob data and dumps the
// results as PPM/PGM frames. Mainly useful for eyeball-checking
// that things work before hooking up a real camera feed.
//
// usage: ./imgpipeline_demo [output_dir]

#include "pipeline.hpp"
#include "image_io.hpp"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <sys/stat.h>
#include <string>

static void draw_circle(imgpipe::ColorImage& img, int cx, int cy, int radius,
                         uint8_t b, uint8_t g, uint8_t r) {
    int r2 = radius * radius;
    for (int y = std::max(0, cy - radius); y <= std::min(img.height - 1, cy + radius); ++y) {
        for (int x = std::max(0, cx - radius); x <= std::min(img.width - 1, cx + radius); ++x) {
            int dx = x - cx, dy = y - cy;
            if (dx*dx + dy*dy <= r2) {
                img.at(y, x, 0) = b;
                img.at(y, x, 1) = g;
                img.at(y, x, 2) = r;
            }
        }
    }
}

static void draw_cross(imgpipe::ColorImage& img, int cx, int cy, int size,
                        uint8_t b, uint8_t g, uint8_t r) {
    for (int d = -size; d <= size; ++d) {
        int yy = cy + d, xx = cx + d;
        if (yy >= 0 && yy < img.height && cx >= 0 && cx < img.width)
            { img.at(yy, cx, 0) = b; img.at(yy, cx, 1) = g; img.at(yy, cx, 2) = r; }
        if (cy >= 0 && cy < img.height && xx >= 0 && xx < img.width)
            { img.at(cy, xx, 0) = b; img.at(cy, xx, 1) = g; img.at(cy, xx, 2) = r; }
    }
}

static void draw_rect_outline(imgpipe::ColorImage& img, int x, int y, int w, int h,
                               uint8_t b, uint8_t g, uint8_t r) {
    for (int c = x; c < x + w && c < img.width; ++c) {
        if (y >= 0 && y < img.height) { img.at(y, c, 0) = b; img.at(y, c, 1) = g; img.at(y, c, 2) = r; }
        int yb = y + h - 1;
        if (yb >= 0 && yb < img.height) { img.at(yb, c, 0) = b; img.at(yb, c, 1) = g; img.at(yb, c, 2) = r; }
    }
    for (int row = y; row < y + h && row < img.height; ++row) {
        if (x >= 0 && x < img.width) { img.at(row, x, 0) = b; img.at(row, x, 1) = g; img.at(row, x, 2) = r; }
        int xr = x + w - 1;
        if (xr >= 0 && xr < img.width) { img.at(row, xr, 0) = b; img.at(row, xr, 1) = g; img.at(row, xr, 2) = r; }
    }
}

static bool mkdir_p(const std::string& path) {
    return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
}

int main(int argc, char* argv[]) {
    std::string out_dir = (argc > 1) ? argv[1] : "output";
    mkdir_p(out_dir);
    mkdir_p(out_dir + "/frames");
    mkdir_p(out_dir + "/masks");
    mkdir_p(out_dir + "/annotated");

    const int W = 320, H = 240;
    const int num_frames = 60;

    std::cout << "Running detection pipeline demo...\n";
    std::cout << "  Resolution: " << W << "x" << H << "\n";
    std::cout << "  Frames: " << num_frames << "\n";
    std::cout << "  Output: " << out_dir << "/\n\n";

    imgpipe::PipelineConfig cfg;
    cfg.min_area = 30;
    cfg.morph_kernel = 3;
    cfg.morph_open_iter = 1;
    cfg.morph_close_iter = 1;
    imgpipe::Pipeline pipeline(cfg);

    // hand-rolled LCG so the output is deterministic across runs
    uint64_t rng_state = 12345;
    auto rng_next = [&]() -> uint32_t {
        rng_state = rng_state * 6364136223846793005ULL + 1;
        return static_cast<uint32_t>(rng_state >> 32);
    };

    // fill background with slightly noisy gray
    imgpipe::ColorImage bg_frame(W, H);
    for (size_t i = 0; i < bg_frame.data.size(); ++i)
        bg_frame.data[i] = 50 + (rng_next() % 20);

    struct Target { double x, y, vx, vy; int radius; };
    std::vector<Target> targets = {
        {80,  60,   2.0,  1.5, 15},
        {240, 180, -1.5,  2.0, 12},
        {160, 120,  1.0, -1.0, 18}
    };

    for (int f = 0; f < num_frames; ++f) {
        imgpipe::ColorImage frame = bg_frame;

        // per-frame noise
        for (size_t i = 0; i < frame.data.size(); ++i) {
            int noise = static_cast<int>(rng_next() % 11) - 5;
            int val = frame.data[i] + noise;
            frame.data[i] = static_cast<uint8_t>(std::clamp(val, 0, 255));
        }

        // move + draw targets
        for (auto& t : targets) {
            t.x += t.vx;
            t.y += t.vy;
            if (t.x < t.radius || t.x > W - t.radius) { t.vx = -t.vx; t.x += 2*t.vx; }
            if (t.y < t.radius || t.y > H - t.radius) { t.vy = -t.vy; t.y += 2*t.vy; }
            draw_circle(frame, (int)t.x, (int)t.y, t.radius, 190, 190, 190);
        }

        auto result = pipeline.process_frame(frame);

        // save everything
        std::ostringstream fname;
        fname << std::setw(4) << std::setfill('0') << f;

        imgpipe::io::save_ppm(out_dir + "/frames/frame_" + fname.str() + ".ppm", frame);
        imgpipe::io::save_pgm(out_dir + "/masks/mask_" + fname.str() + ".pgm", result.foreground_mask);

        // annotated version with bboxes + track markers
        imgpipe::ColorImage vis = frame;
        for (const auto& det : result.detections)
            draw_rect_outline(vis, det.bbox.x, det.bbox.y,
                              det.bbox.width, det.bbox.height, 0, 255, 0);
        for (const auto& [tid, track] : result.tracks) {
            int cx = static_cast<int>(track.centroid.x);
            int cy = static_cast<int>(track.centroid.y);
            draw_cross(vis, cx, cy, 5, 0, 0, 255);
        }
        imgpipe::io::save_ppm(out_dir + "/annotated/ann_" + fname.str() + ".ppm", vis);

        if ((f + 1) % 10 == 0 || f == 0) {
            std::cout << "  Frame " << std::setw(3) << f + 1 << "/" << num_frames
                      << " | detections: " << result.detections.size()
                      << " | tracks: " << result.tracks.size() << "\n";
        }
    }

    std::cout << "\nDone. Output in " << out_dir << "/\n";
    return 0;
}
