#include <png.h>
#include <cstddef>
#include <string>
#include <chrono>
#include <iostream>

#include "error.hpp"

int width, height;
png_bytep *row_pointers = NULL;
int patch_size = 32;

void write_png(png_bytep buffer, const char* filename) {
  png_structp png_ptr =
    png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);

  if (!png_ptr)
    return;

  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr)
  {
    png_destroy_write_struct(&png_ptr, nullptr);
    return;
  }

  FILE* fp = fopen(filename, "wb");
  png_init_io(png_ptr, fp);

  png_set_IHDR(png_ptr, info_ptr,
               width,
               height,
               8,
               PNG_COLOR_TYPE_GRAY,
               PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);

  png_write_info(png_ptr, info_ptr);
  for (int i = 0; i < height; ++i)
  {
    png_write_row(png_ptr, buffer);
    buffer += width;
  }

  png_write_end(png_ptr, info_ptr);
  png_destroy_write_struct(&png_ptr, &info_ptr);
  fclose(fp);
}


void read_png(const char *filename)
{
    FILE *fp = fopen(filename, "rb");

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if(!png) abort();

    png_infop info = png_create_info_struct(png);
    if(!info) abort();

    if(setjmp(png_jmpbuf(png))) abort();

    png_init_io(png, fp);

    png_read_info(png, info);

    width      = png_get_image_width(png, info);
    height     = png_get_image_height(png, info);
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth  = png_get_bit_depth(png, info);

    // Read any color_type into 8bit depth, RGBA format.
    // See http://www.libpng.org/pub/png/libpng-manual.txt

    if(bit_depth == 16)
        png_set_strip_16(png);

    if(color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png);

    // PNG_COLOR_TYPE_GRAY_ALPHA is always 8 or 16bit depth.
    if(color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png);

    if(png_get_valid(png, info, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png);

    // These color_type don't have an alpha channel then fill it with 0xff.
    if(color_type == PNG_COLOR_TYPE_RGB ||
            color_type == PNG_COLOR_TYPE_GRAY ||
            color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER);

    if(color_type == PNG_COLOR_TYPE_GRAY ||
            color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);

    png_read_update_info(png, info);

    row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    for (int y = 0; y < height; y++) {
        row_pointers[y] = (png_byte*)malloc(png_get_rowbytes(png,info));
    }

    png_read_image(png, row_pointers);

    fclose(fp);

    png_destroy_read_struct(&png, &info, NULL);
}

void free_row() {
    for (int i = 0; i < height; i++) {
        png_bytep row = row_pointers[i];
        free(row);
    }
    free(row_pointers);
}

std::uint8_t *img_to_grayscale(png_bytep *img) {
    std::uint8_t *gray_img = (std::uint8_t *)malloc(width * height * sizeof(std::uint8_t));
    for (int i = 0; i < height; i++) {
        png_bytep row = img[i];
        std::uint8_t *my_row = gray_img + i * width;
        for (int j = 0; j < width; j++) {
            png_bytep px = &(row[j * 4]);
            my_row[j] = 0.2126 * px[0] + 0.7152 * px[1] + 0.0722 * px[2];
        }
    }
    free_row();
    return gray_img;
}

void compute_sobel(std::uint8_t current[9], std::uint8_t *result_x, std::uint8_t *result_y) {
    int kernel_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    int kernel_y[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

    int x = 0;
    int y = 0;

    for (int i = 0; i < 9; i++) {
        x += current[i] * kernel_x[i];
        y += current[i] * kernel_y[i];
    }

    x = x > 255 ? 255 : x;
    x = x < 0 ? 0 : x;

    y = y > 255 ? 255 : y;
    y = y < 0 ? 0 : y;

    *result_x = x;
    *result_y = y;
}

void sobel_filter(std::uint8_t *img, std::uint8_t *sobel_x, std::uint8_t *sobel_y) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            std::uint8_t current[9];

            current[0] = i > 0 && j > 0 ? *(img + (i - 1) * width + j - 1) : 0;
            current[1] = i > 0 ? *(img + (i - 1) * width + j) : 0;
            current[2] = i > 0 && j < (width -1) ? *(img + (i - 1) * width + j + 1) : 0;

            current[3] = j > 0 ? *(img + i * width + j - 1) : 0;
            current[4] = *(img + i * width + j);
            current[5] = j < (width -1) ? *(img + i * width + j + 1) : 0;

            current[6] = i < (height - 1) && j > 0 ? *(img + (i + 1) * width + j - 1) : 0;
            current[7] = i < (height - 1) ? *(img + (i + 1) * width + j) : 0;
            current[8] = i < (height - 1) && j < (width -1) ? *(img + (i + 1) * width + j + 1) : 0;

            std::uint8_t result_x = 0;
            std::uint8_t result_y = 0;

            compute_sobel(current, &result_x, &result_y);

            *(sobel_x + i * width + j) = result_x;
            *(sobel_y + i * width + j) = result_y;
        }
    }
}

void compute_patch(std::uint8_t *sobel_x, std::uint8_t *sobel_y, int x, int y,
                   std::uint8_t *response) {
    int nb_pixel = patch_size * patch_size;

    std::uint8_t *patch_sobel_x = sobel_x + (x * patch_size * width) + y * patch_size;
    std::uint8_t *patch_sobel_y = sobel_y + (x * patch_size * width) + y * patch_size;

    int gradient_x = 0;
    int gradient_y = 0;

    for (int i = 0; i < patch_size; i++) {
        for (int j = 0; j < patch_size; j++) {
            gradient_x += *(patch_sobel_x + (i * width) + j);
            gradient_y += *(patch_sobel_y + (i * width) + j);
        }
    }

    gradient_x /= nb_pixel;
    gradient_y /= nb_pixel;

    int patch_width = width / patch_size;
    int delta = gradient_x - gradient_y < 0 ? 0 : gradient_x - gradient_y;
    *(response + x * patch_width + y) = delta;

    for (int i = 0; i < patch_size; i++) {
        for (int j = 0; j < patch_size; j++) {
            std::uint8_t *elt_x = patch_sobel_x + (i * width) + j;
            std::uint8_t *elt_y = patch_sobel_y + (i * width) + j;
            *elt_x = *elt_x + gradient_x > 255 ? 255 : *elt_x + gradient_x;
            *elt_y = *elt_y + gradient_y > 255 ? 255 : *elt_y + gradient_y;
        }
    }
}

std::uint8_t *compute_response(std::uint8_t *sobel_x, std::uint8_t *sobel_y) {
    int patch_height = height / patch_size;
    int patch_width = width / patch_size;
    std::uint8_t *response = (std::uint8_t *) malloc(patch_height * patch_width);

    for (int i = 0; i < patch_height; i++) {
        for (int j = 0; j < patch_width; j++) {
            compute_patch(sobel_x, sobel_y, i, j, response);
        }
    }

    return response;
}

std::uint8_t min(std::uint8_t *array, int length) {
    std::uint8_t min = 255;

    for (int i = 0; i < length; i++) {
        if (array[i] < min)
            min = array[i];
    }

    return min;
}

std::uint8_t max(std::uint8_t *array, int length) {
    std::uint8_t max = 0;

    for (int i = 0; i < length; i++) {
        if (array[i] > max)
            max = array[i];
    }

    return max;
}

std::uint8_t *patch_to_img(std::uint8_t *patches, const char *filename, bool print=true) {
    std::uint8_t *img = (std::uint8_t *) malloc(height * width);
    int patch_height = height / patch_size;
    int patch_width = width / patch_size;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (i / patch_size >= patch_height || j / patch_size >= patch_width)
                *(img + i * width + j) = 0;
            else
                *(img + i * width + j) = *(patches + (i / patch_size) * patch_width + (j / patch_size));
        }
    }

    if (!print)
        return img;

    write_png(img, filename);
    free(img);
    return nullptr;
}

void compute_kernel(std::uint8_t *response, std::uint8_t *kernel, int i, int j, uint8_t default_value) {
    int patch_height = height / patch_size;
    int patch_width = width / patch_size;

    kernel[0] = i > 0 && j > 1 ? *(response + (i - 1)  * patch_width + (j - 2)) : default_value;
    kernel[1] = i > 0 && j > 0 ? *(response + (i - 1) * patch_width + (j - 1)) : default_value;
    kernel[2] = i > 0 ? *(response + (i - 1) * patch_width + j) : default_value;
    kernel[3] = i > 0 && j < (patch_width - 1) ? *(response + (i - 1) * patch_width + (j + 1)) : default_value;
    kernel[4] = i > 0 && j < (patch_width - 2) ? *(response + (i - 1) * patch_width + (j + 2)) : default_value;

    kernel[5] = j > 1 ? *(response + i * patch_width + (j - 2)) : default_value;
    kernel[6] = j > 0 ? *(response + i * patch_width + (j - 1)) : default_value;
    kernel[7] = *(response + i * patch_width + j);
    kernel[8] = j < (patch_width - 1) ? *(response + i * patch_width + (j + 1)) : default_value;
    kernel[9] = j < (patch_width - 2) ? *(response + i * patch_width + (j + 2)) : default_value;

    kernel[10] = i < (patch_height - 1) && j > 1 ? *(response + (i + 1) * patch_width + (j - 2)) : default_value;
    kernel[11] = i < (patch_height - 1) && j > 0 ? *(response + (i + 1) * patch_width + (j - 1)) : default_value;
    kernel[12] = i < (patch_height - 1) ? *(response + (i + 1) * patch_width + j) : default_value;
    kernel[13] = i < (patch_height - 1) && j < (patch_width - 1) ? *(response + (i + 1) * patch_width + (j + 1)) : default_value;
    kernel[14] = i < (patch_height - 1) && j < (patch_width - 2) ? *(response + (i + 1) * patch_width + (j + 2)) : default_value;
}

std::uint8_t *dilation(std::uint8_t *response) {
    int patch_height = height / patch_size;
    int patch_width = width / patch_size;

    std::uint8_t *kernel = (std::uint8_t *) malloc(15);
    std::uint8_t *img = (std::uint8_t *) malloc(patch_height * patch_width);

    for (int i = 0; i < patch_height; i++) {
        for (int j = 0; j < patch_width; j++) {
            compute_kernel(response, kernel, i, j, 0);

            std::uint8_t m = max(kernel, 15);
            *(img + i * patch_width + j) = m;
        }
    }

    free(kernel);
    free(response);
    return img;
}

std::uint8_t *erosion(std::uint8_t *response) {
    int patch_height = height / patch_size;
    int patch_width = width / patch_size;

    std::uint8_t *kernel = (std::uint8_t *) malloc(15);
    std::uint8_t *img = (std::uint8_t *) malloc(patch_height * patch_width);

    for (int i = 0; i < patch_height; i++) {
        for (int j = 0; j < patch_width; j++) {
            compute_kernel(response, kernel, i, j, 255);

            std::uint8_t m = min(kernel, 15);
            *(img + i * patch_width + j) = m;
        }
    }

    free(kernel);
    free(response);
    return img;
}

void activation_map(std::uint8_t *response, std::uint8_t threshold) {
    int patch_height = height / patch_size;
    int patch_width = width / patch_size;

    for (int i = 0; i < patch_height; i++) {
        for (int j = 0; j < patch_width; j++) {
            *(response + i * patch_width + j) = *(response + i * patch_width + j) > threshold ? 255 : 0;
        }
    }
}

void connect_component(std::uint8_t *response) {
    int patch_height = height / patch_size;
    int patch_width = width / patch_size;

    for (int i = 1; i < patch_height - 1; i++) {
        for (int j = 1; j < patch_width - 1; j++) {
            std::uint8_t count = 0;
            count += *(response + (i - 1) * patch_width + j) > 0;
            count += *(response + i * patch_width + j - 1) > 0;
            count += *(response + i * patch_width + j + 1) > 0;
            count += *(response + (i + 1) * patch_width + j) > 0;

            if (count >= 2)
                *(response + i * patch_width + j) = 255;
        }
    }
}

void detect_barcode(const char* filename, const char *truth) {
    read_png(filename);

    int patch_height = height / patch_size;
    int patch_width = width / patch_size;

    std::uint8_t *gray_img = img_to_grayscale(row_pointers);
    write_png(gray_img, "gray.png");

    std::uint8_t *sobel_x = (std::uint8_t *)malloc(height * width);
    std::uint8_t *sobel_y = (std::uint8_t *)malloc(height * width);
    sobel_filter(gray_img, sobel_x, sobel_y);
    write_png(sobel_x, "sobel_x.png");
    write_png(sobel_y, "sobel_y.png");

    std::uint8_t *response = compute_response(sobel_x, sobel_y);
    patch_to_img(response, "response.png");
    response = dilation(response);
    patch_to_img(response, "dilation.png");
    response = erosion(response);
    patch_to_img(response, "erosion.png");

    std::uint8_t threshold = max(response, patch_height * patch_width) / 2;
    activation_map(response, threshold);
    patch_to_img(response, "barcode.png");
    connect_component(response);
    patch_to_img(response, "cc.png");

    if (truth != NULL) {
        std::uint8_t *realsize_img = patch_to_img(response, "", false);
        read_png(truth);
        std::uint8_t *gt = img_to_grayscale(row_pointers);
        std::uint8_t *error = compute_error(realsize_img, gt, height, width);

        write_png(error, "error.png");

        free(realsize_img);
        free(gt);
        free(error);
    }

    free(gray_img);
    free(sobel_x);
    free(sobel_y);
    free(response);
}

int main(int argc, char **argv) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    if (argc < 2) {
        printf("Missing input img\n");
        return 1;
    }
    const char *filename = argv[1];
    if (argc == 2) {
        detect_barcode(filename, NULL);
    } else {
        const char *truth = argv[2];
        detect_barcode(filename, truth);
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time elapsed : " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    std::cout << " nanoseconds" << std::endl;
    std::cout << "Time elapsed : " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout << " microseconds" << std::endl;

    return 0;
}

