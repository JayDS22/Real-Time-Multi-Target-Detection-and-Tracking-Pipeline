// image_io.hpp
// PPM/PGM reader & writer. We use these formats because they're trivially
// simple — no compression, no libraries. For production you'd swap in
// stb_image or libpng.

#ifndef IMGPIPELINE_IMAGE_IO_HPP
#define IMGPIPELINE_IMAGE_IO_HPP

#include "types.hpp"
#include <string>

namespace imgpipe {
namespace io {

bool load_ppm(const std::string& path, ColorImage& out);
bool save_ppm(const std::string& path, const ColorImage& img);
bool save_pgm(const std::string& path, const GrayImage& img);
bool load_pgm(const std::string& path, GrayImage& out);

}  // namespace io
}  // namespace imgpipe

#endif
