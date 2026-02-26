// test_pipeline.cpp
// Quick unit tests for each pipeline stage. No external test framework;
// just assertion macros and a pass/fail count at the end.

#include "types.hpp"
#include "background_model.hpp"
#include "morphology.hpp"
#include "detection_extractor.hpp"
#include "tracker.hpp"
#include "pipeline.hpp"
#include "image_io.hpp"

#include <iostream>
#include <cmath>
#include <cassert>
#include <cstring>

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "  FAIL: " << msg << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
            return 1; \
        } \
    } while(0)

// --- helpers ---

static imgpipe::ColorImage make_solid_frame(int w, int h, uint8_t b, uint8_t g, uint8_t r) {
    imgpipe::ColorImage img(w, h);
    for (int i = 0; i < w * h; ++i) {
        img.data[i*3 + 0] = b;
        img.data[i*3 + 1] = g;
        img.data[i*3 + 2] = r;
    }
    return img;
}

static void paint_rect(imgpipe::ColorImage& img, int x, int y, int w, int h,
                        uint8_t b, uint8_t g, uint8_t r) {
    for (int row = y; row < y + h && row < img.height; ++row)
        for (int col = x; col < x + w && col < img.width; ++col) {
            img.at(row, col, 0) = b;
            img.at(row, col, 1) = g;
            img.at(row, col, 2) = r;
        }
}

static int count_nonzero(const imgpipe::GrayImage& img) {
    int count = 0;
    for (auto px : img.data)
        if (px != 0) count++;
    return count;
}

// --- test cases ---

int test_gray_image_basics() {
    std::cout << "  test_gray_image_basics... ";
    imgpipe::GrayImage img(10, 5);
    TEST_ASSERT(img.width == 10, "width");
    TEST_ASSERT(img.height == 5, "height");
    TEST_ASSERT(img.pixel_count() == 50, "pixel_count");
    img.at(2, 3) = 128;
    TEST_ASSERT(img.at(2, 3) == 128, "pixel access");
    std::cout << "OK\n";
    return 0;
}

int test_color_image_basics() {
    std::cout << "  test_color_image_basics... ";
    imgpipe::ColorImage img(8, 6);
    TEST_ASSERT(img.data.size() == 8 * 6 * 3, "data size");
    img.at(1, 2, 0) = 100;
    img.at(1, 2, 2) = 200;
    TEST_ASSERT(img.at(1, 2, 0) == 100, "blue channel");
    TEST_ASSERT(img.at(1, 2, 2) == 200, "red channel");
    std::cout << "OK\n";
    return 0;
}

int test_background_model_init() {
    std::cout << "  test_background_model_init... ";
    imgpipe::BackgroundModel bg(0.05, 25);
    TEST_ASSERT(!bg.is_initialized(), "not initialized yet");

    auto frame = make_solid_frame(32, 32, 60, 60, 60);
    auto mask = bg.update(frame);
    TEST_ASSERT(bg.is_initialized(), "should be initialized now");
    TEST_ASSERT(count_nonzero(mask) == 0, "first frame -> empty mask");
    std::cout << "OK\n";
    return 0;
}

int test_background_model_detects_change() {
    std::cout << "  test_background_model_detects_change... ";
    imgpipe::BackgroundModel bg(0.01, 20);

    auto bg_frame = make_solid_frame(64, 64, 50, 50, 50);
    for (int i = 0; i < 10; ++i)
        bg.update(bg_frame);

    // put a bright rectangle in and see if it shows up
    auto frame_with_obj = bg_frame;
    paint_rect(frame_with_obj, 20, 20, 15, 15, 200, 200, 200);
    auto mask = bg.update(frame_with_obj);

    int fg_pixels = count_nonzero(mask);
    TEST_ASSERT(fg_pixels > 100, "should pick up the bright rect");
    TEST_ASSERT(fg_pixels < 64*64, "shouldn't flag everything");
    std::cout << "OK\n";
    return 0;
}

int test_morphology_removes_noise() {
    std::cout << "  test_morphology_removes_noise... ";
    imgpipe::MorphologyFilter morph(5, 2, 2);

    imgpipe::GrayImage mask(100, 100);
    // big 30x30 blob in the center
    for (int r = 35; r < 65; ++r)
        for (int c = 35; c < 65; ++c)
            mask.at(r, c) = 255;

    // a few isolated pixels that should get killed
    mask.at(5, 5) = 255;
    mask.at(10, 90) = 255;
    mask.at(90, 10) = 255;

    auto cleaned = morph.apply(mask);
    TEST_ASSERT(cleaned.at(50, 50) == 255, "blob center should survive");
    TEST_ASSERT(cleaned.at(5, 5) == 0, "noise pixel should be gone");
    TEST_ASSERT(cleaned.at(10, 90) == 0, "noise pixel should be gone");
    std::cout << "OK\n";
    return 0;
}

int test_erode_shrinks() {
    std::cout << "  test_erode_shrinks... ";
    imgpipe::MorphologyFilter morph(3, 1, 1);

    imgpipe::GrayImage mask(20, 20);
    for (int r = 5; r < 15; ++r)
        for (int c = 5; c < 15; ++c)
            mask.at(r, c) = 255;

    int before = count_nonzero(mask);
    int after = count_nonzero(morph.erode(mask));
    TEST_ASSERT(after < before, "erosion should shrink the blob");
    TEST_ASSERT(after > 0, "shouldn't wipe out a 10x10 block entirely");
    std::cout << "OK\n";
    return 0;
}

int test_dilate_expands() {
    std::cout << "  test_dilate_expands... ";
    imgpipe::MorphologyFilter morph(3, 1, 1);

    imgpipe::GrayImage mask(30, 30);
    for (int r = 13; r < 17; ++r)
        for (int c = 13; c < 17; ++c)
            mask.at(r, c) = 255;

    int before = count_nonzero(mask);
    int after = count_nonzero(morph.dilate(mask));
    TEST_ASSERT(after > before, "dilation should grow the blob");
    std::cout << "OK\n";
    return 0;
}

int test_detection_extractor_single_blob() {
    std::cout << "  test_detection_extractor_single_blob... ";
    imgpipe::DetectionExtractor ext(10, 50000);

    imgpipe::GrayImage mask(100, 100);
    for (int r = 40; r < 60; ++r)
        for (int c = 40; c < 60; ++c)
            mask.at(r, c) = 255;

    auto dets = ext.extract(mask);
    TEST_ASSERT(dets.size() == 1, "should find one blob");

    auto& d = dets[0];
    TEST_ASSERT(d.area >= 390 && d.area <= 410, "area ~400");
    TEST_ASSERT(std::abs(d.centroid.x - 49.5) < 1.0, "centroid x");
    TEST_ASSERT(std::abs(d.centroid.y - 49.5) < 1.0, "centroid y");
    TEST_ASSERT(d.confidence > 0.0 && d.confidence <= 1.0, "confidence range");
    std::cout << "OK\n";
    return 0;
}

int test_detection_extractor_multiple_blobs() {
    std::cout << "  test_detection_extractor_multiple_blobs... ";
    imgpipe::DetectionExtractor ext(10, 50000);

    imgpipe::GrayImage mask(200, 200);
    for (int r = 10; r < 30; ++r)
        for (int c = 10; c < 30; ++c)
            mask.at(r, c) = 255;
    for (int r = 150; r < 180; ++r)
        for (int c = 150; c < 180; ++c)
            mask.at(r, c) = 255;

    auto dets = ext.extract(mask);
    TEST_ASSERT(dets.size() == 2, "should find two separate blobs");
    std::cout << "OK\n";
    return 0;
}

int test_detection_extractor_area_filter() {
    std::cout << "  test_detection_extractor_area_filter... ";
    imgpipe::DetectionExtractor ext(500, 50000);

    imgpipe::GrayImage mask(100, 100);
    // tiny 3x3 blob, area=9, way below min_area=500
    for (int r = 48; r < 51; ++r)
        for (int c = 48; c < 51; ++c)
            mask.at(r, c) = 255;

    auto dets = ext.extract(mask);
    TEST_ASSERT(dets.empty(), "tiny blob should be filtered out");
    std::cout << "OK\n";
    return 0;
}

int test_tracker_registers_new() {
    std::cout << "  test_tracker_registers_new... ";
    imgpipe::CentroidTracker tracker(10, 50.0);

    std::vector<imgpipe::Detection> dets;
    imgpipe::Detection d1;
    d1.centroid = {100.0, 100.0}; d1.area = 200;
    dets.push_back(d1);

    auto& tracks = tracker.update(dets);
    TEST_ASSERT(tracks.size() == 1, "one detection -> one track");
    TEST_ASSERT(tracks.begin()->second.age == 0, "brand new track");
    std::cout << "OK\n";
    return 0;
}

int test_tracker_associates_correctly() {
    std::cout << "  test_tracker_associates_correctly... ";
    imgpipe::CentroidTracker tracker(10, 50.0);

    std::vector<imgpipe::Detection> dets1;
    imgpipe::Detection d;
    d.centroid = {100.0, 100.0}; d.area = 200;
    dets1.push_back(d);
    tracker.update(dets1);

    // same object moved slightly
    std::vector<imgpipe::Detection> dets2;
    d.centroid = {105.0, 103.0};
    dets2.push_back(d);
    auto& tracks = tracker.update(dets2);

    TEST_ASSERT(tracks.size() == 1, "still one track");
    auto& trk = tracks.begin()->second;
    TEST_ASSERT(trk.age == 1, "age bumped to 1");
    TEST_ASSERT(trk.history.size() == 2, "two history entries");
    std::cout << "OK\n";
    return 0;
}

int test_tracker_removes_disappeared() {
    std::cout << "  test_tracker_removes_disappeared... ";
    imgpipe::CentroidTracker tracker(3, 50.0);  // drop after 3 misses

    std::vector<imgpipe::Detection> dets;
    imgpipe::Detection d;
    d.centroid = {50.0, 50.0}; d.area = 100;
    dets.push_back(d);
    tracker.update(dets);

    // 4 frames of nothing
    std::vector<imgpipe::Detection> empty;
    for (int i = 0; i < 4; ++i)
        tracker.update(empty);

    TEST_ASSERT(tracker.tracks().empty(), "should have dropped the track by now");
    std::cout << "OK\n";
    return 0;
}

int test_tracker_handles_multiple() {
    std::cout << "  test_tracker_handles_multiple... ";
    imgpipe::CentroidTracker tracker(10, 30.0);

    std::vector<imgpipe::Detection> dets;
    imgpipe::Detection d1, d2;
    d1.centroid = {10.0, 10.0}; d1.area = 100;
    d2.centroid = {200.0, 200.0}; d2.area = 100;
    dets.push_back(d1);
    dets.push_back(d2);

    auto& tracks = tracker.update(dets);
    TEST_ASSERT(tracks.size() == 2, "two detections -> two tracks");

    dets[0].centroid = {12.0, 11.0};
    dets[1].centroid = {198.0, 202.0};
    tracker.update(dets);
    TEST_ASSERT(tracker.tracks().size() == 2, "still two after small movement");
    std::cout << "OK\n";
    return 0;
}

int test_full_pipeline() {
    std::cout << "  test_full_pipeline... ";
    imgpipe::PipelineConfig cfg;
    cfg.min_area = 20;
    cfg.morph_kernel = 3;
    cfg.morph_open_iter = 1;
    cfg.morph_close_iter = 1;
    imgpipe::Pipeline pipe(cfg);

    auto bg_frame = make_solid_frame(100, 100, 50, 50, 50);
    for (int i = 0; i < 5; ++i)
        pipe.process_frame(bg_frame);

    auto frame = bg_frame;
    paint_rect(frame, 40, 40, 20, 20, 200, 200, 200);
    auto result = pipe.process_frame(frame);

    TEST_ASSERT(!result.detections.empty(), "should detect the rectangle");
    TEST_ASSERT(!result.tracks.empty(), "should have at least one track");
    std::cout << "OK\n";
    return 0;
}

int test_ppm_roundtrip() {
    std::cout << "  test_ppm_roundtrip... ";
    imgpipe::ColorImage img(16, 8);
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 16; ++c) {
            img.at(r, c, 0) = static_cast<uint8_t>(r * 30);
            img.at(r, c, 1) = static_cast<uint8_t>(c * 15);
            img.at(r, c, 2) = 128;
        }

    const char* path = "/tmp/test_roundtrip.ppm";
    TEST_ASSERT(imgpipe::io::save_ppm(path, img), "save failed");

    imgpipe::ColorImage loaded;
    TEST_ASSERT(imgpipe::io::load_ppm(path, loaded), "load failed");
    TEST_ASSERT(loaded.width == 16 && loaded.height == 8, "dims mismatch");

    for (size_t i = 0; i < img.data.size(); ++i)
        TEST_ASSERT(img.data[i] == loaded.data[i], "pixel mismatch");

    std::cout << "OK\n";
    return 0;
}

int test_pgm_roundtrip() {
    std::cout << "  test_pgm_roundtrip... ";
    imgpipe::GrayImage img(20, 10);
    for (int r = 0; r < 10; ++r)
        for (int c = 0; c < 20; ++c)
            img.at(r, c) = static_cast<uint8_t>((r + c) * 10);

    const char* path = "/tmp/test_roundtrip.pgm";
    TEST_ASSERT(imgpipe::io::save_pgm(path, img), "save failed");

    imgpipe::GrayImage loaded;
    TEST_ASSERT(imgpipe::io::load_pgm(path, loaded), "load failed");
    TEST_ASSERT(loaded.width == img.width && loaded.height == img.height, "dims");
    for (size_t i = 0; i < img.data.size(); ++i)
        TEST_ASSERT(img.data[i] == loaded.data[i], "pixel mismatch");

    std::cout << "OK\n";
    return 0;
}

int main() {
    std::cout << "Running pipeline unit tests...\n\n";

    int failures = 0;
    failures += test_gray_image_basics();
    failures += test_color_image_basics();
    failures += test_background_model_init();
    failures += test_background_model_detects_change();
    failures += test_morphology_removes_noise();
    failures += test_erode_shrinks();
    failures += test_dilate_expands();
    failures += test_detection_extractor_single_blob();
    failures += test_detection_extractor_multiple_blobs();
    failures += test_detection_extractor_area_filter();
    failures += test_tracker_registers_new();
    failures += test_tracker_associates_correctly();
    failures += test_tracker_removes_disappeared();
    failures += test_tracker_handles_multiple();
    failures += test_full_pipeline();
    failures += test_ppm_roundtrip();
    failures += test_pgm_roundtrip();

    std::cout << "\n";
    if (failures == 0)
        std::cout << "All tests passed.\n";
    else
        std::cout << failures << " test(s) FAILED.\n";

    return failures;
}
