#include "error.hpp"

std::uint8_t *compute_error(std::uint8_t *img, std::uint8_t *gt, int height, int width) {

    std::uint8_t *error = (std::uint8_t *)malloc(height * width);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (gt[i * width + j] > img[i * width + j])
                error[i * width + j] = 255;
            else
                error[i * width + j] = img[i * width + j] - gt[i * width + j];
        }
    }
    return error;
}
