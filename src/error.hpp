#pragma once
#include <cstdint>
#include <cstdlib>

std::uint8_t *compute_error(std::uint8_t *img, std::uint8_t *gt, int height, int width);
