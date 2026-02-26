// image_io.cpp
// PPM (P6 binary RGB) and PGM (P5 binary gray) handling.
// These are maybe the simplest image formats around — a text header
// followed by raw bytes. Makes it easy to test without bringing in
// a full codec library.

#include "image_io.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

namespace imgpipe {
namespace io {

static void skip_comments(std::ifstream& f) {
    while (f.peek() == '#' || f.peek() == '\n' || f.peek() == '\r' || f.peek() == ' ') {
        if (f.peek() == '#') {
            std::string dummy;
            std::getline(f, dummy);
        } else {
            f.get();
        }
    }
}

bool load_ppm(const std::string& path, ColorImage& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    std::string magic;
    f >> magic;
    if (magic != "P6") {
        std::cerr << "load_ppm: expected P6, got " << magic << "\n";
        return false;
    }

    skip_comments(f);
    int w, h, maxval;
    f >> w >> h;
    skip_comments(f);
    f >> maxval;
    f.get();  // eat the whitespace after maxval

    out = ColorImage(w, h);

    // PPM stores RGB but we use BGR internally, so swap R and B
    std::vector<uint8_t> rgb(3 * w * h);
    f.read(reinterpret_cast<char*>(rgb.data()), rgb.size());

    for (size_t i = 0; i < static_cast<size_t>(w * h); ++i) {
        out.data[i * 3 + 0] = rgb[i * 3 + 2];  // B
        out.data[i * 3 + 1] = rgb[i * 3 + 1];  // G
        out.data[i * 3 + 2] = rgb[i * 3 + 0];  // R
    }
    return true;
}

bool save_ppm(const std::string& path, const ColorImage& img) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    f << "P6\n" << img.width << " " << img.height << "\n255\n";

    // BGR -> RGB for the file
    std::vector<uint8_t> rgb(img.data.size());
    for (size_t i = 0; i < static_cast<size_t>(img.width * img.height); ++i) {
        rgb[i * 3 + 0] = img.data[i * 3 + 2];
        rgb[i * 3 + 1] = img.data[i * 3 + 1];
        rgb[i * 3 + 2] = img.data[i * 3 + 0];
    }
    f.write(reinterpret_cast<const char*>(rgb.data()), rgb.size());
    return f.good();
}

bool save_pgm(const std::string& path, const GrayImage& img) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    f << "P5\n" << img.width << " " << img.height << "\n255\n";
    f.write(reinterpret_cast<const char*>(img.data.data()), img.data.size());
    return f.good();
}

bool load_pgm(const std::string& path, GrayImage& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    std::string magic;
    f >> magic;
    if (magic != "P5") return false;

    skip_comments(f);
    int w, h, maxval;
    f >> w >> h;
    skip_comments(f);
    f >> maxval;
    f.get();

    out = GrayImage(w, h);
    f.read(reinterpret_cast<char*>(out.data.data()), out.data.size());
    return true;
}

}  // namespace io
}  // namespace imgpipe
