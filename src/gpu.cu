#include <cstddef>
#include <string>
#include <iostream>
#include <algorithm>

#include <png.h>

void write_png(png_bytep buffer, const char* filename, int width, int height) {
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

void read_png(const char *filename, int* width, int* height, png_bytep** row_pointers)
{
    FILE *fp = fopen(filename, "rb");

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if(!png) abort();

    png_infop info = png_create_info_struct(png);
    if(!info) abort();

    if(setjmp(png_jmpbuf(png))) abort();

    png_init_io(png, fp);

    png_read_info(png, info);

    *width      = png_get_image_width(png, info);
    *height     = png_get_image_height(png, info);
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

    *row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * (*height));
    for (int y = 0; y < (*height); y++) {
        (*row_pointers)[y] = (png_byte*)malloc(png_get_rowbytes(png,info));
    }

    png_read_image(png, *row_pointers);

    fclose(fp);

    png_destroy_read_struct(&png, &info, NULL);
}

std::uint8_t *img_to_grayscale(png_bytep *img, int width, int height) {
    std::uint8_t *gray_img = (std::uint8_t *)malloc(width * height * sizeof(std::uint8_t));
    for (int i = 0; i < height; i++) {
        png_bytep row = img[i];
        std::uint8_t *my_row = gray_img + i * width;
        for (int j = 0; j < width; j++) {
            png_bytep px = &(row[j * 4]);
            my_row[j] = 0.2126 * px[0] + 0.7152 * px[1] + 0.0722 * px[2];
        }
    }
    return gray_img;
}

__device__ void mm_sobel(uint8_t* values, uint8_t* result_x, uint8_t* result_y) {
    int kernel_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    int kernel_y[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

    int x = 0;
    int y = 0;

    for (int i = 0; i < 9; i++) {
        x += values[i] * kernel_x[i];
        y += values[i] * kernel_y[i];
    }

    x = x > 255 ? 255 : x;
    x = x < 0 ? 0 : x;

    y = y > 255 ? 255 : y;
    y = y < 0 ? 0 : y;

    *result_x = x;
    *result_y = y;

}

__global__ void compute_sobel(uint8_t* buffer_in, uint8_t* sobel_x, uint8_t* sobel_y,
                                int width, int height,
                                size_t pitch_in, size_t pitch_x, size_t pitch_y)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    uint8_t values[9];

    values[0] = (y > 0 && x > 0) ? buffer_in[(y - 1) * pitch_in + (x - 1)] : 0;
    values[1] = y > 0 ? buffer_in[(y - 1) * pitch_in + x] : 0;
    values[2] = (y > 0 && x < width) ? buffer_in[(y - 1) * pitch_in + (x + 1)] : 0;

    values[3] = x > 0 ? buffer_in[y * pitch_in + (x - 1)] : 0;
    values[4] = buffer_in[y * pitch_in + x];
    values[5] = x < width ? buffer_in[y * pitch_in + (x + 1)] : 0;

    values[6] = (y < height && x > 0) ? buffer_in[(y + 1) * pitch_in + (x - 1)] : 0;
    values[7] = (y < height) ? buffer_in[(y + 1) * pitch_in + x] : 0;
    values[8] = (y < height && x < width) ? buffer_in[(y + 1) * pitch_in + (x + 1)] : 0;

    uint8_t result_x = 0;
    uint8_t result_y = 0;

    mm_sobel(values, &result_x, &result_y);

    sobel_x[y * pitch_x + x] = result_x;
    sobel_y[y * pitch_y + x] = result_y;
}

__global__ void compute_response(uint8_t* sobel_x, uint8_t* sobel_y, uint8_t* response,
                                int width, int height, int patch_size,
                                size_t pitch_x, size_t pitch_y, size_t pitch_out)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width / patch_size || y >= height / patch_size)
        return;

    int gradient_x = 0;
    int gradient_y = 0;

    for (int i = 0; i < patch_size; i++) {
        int pos_y = y * patch_size + i;
        for (int j = 0; j < patch_size; j++) {
            int pos_x = x * patch_size + j;
            gradient_x += sobel_x[pos_y * pitch_x + pos_x];
            gradient_y += sobel_y[pos_y * pitch_y + pos_x];
        }
    }

    gradient_x /= patch_size * patch_size;
    gradient_y /= patch_size * patch_size;

    int delta = gradient_x - gradient_y < 0 ? 0 : gradient_x - gradient_y;

    response[y * pitch_out + x] = delta;
}

__global__ void compute_dilation(uint8_t* buffer_in, uint8_t* buffer_out,
                                int width, int height,
                                size_t pitch_in, size_t pitch_out)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    uint8_t max = 0;

    for (int i = -2; i <= 2; i++) {
        for (int j = -1; j <= 1; j++) {
            if (x + i >= 0 && x + i < width && y + j >= 0 && y + j < height && buffer_in[(y + j) * pitch_in + (x + i)] > max)
                max = buffer_in[(y + j) * pitch_in + (x + i)];
        }
    }

    buffer_out[y * pitch_out + x] = max;
}

__global__ void compute_erosion(uint8_t* buffer_in, uint8_t* buffer_out,
                                int width, int height,
                                size_t pitch_in, size_t pitch_out)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    uint8_t min = 255;

    for (int i = -2; i <= 2; i++) {
        for (int j = -1; j <= 1; j++) {
            if (x + i >= 0 && x + i < width && y + j >= 0 && y + j < height && buffer_in[(y + j) * pitch_in + (x + i)] < min)
                min = buffer_in[(y + j) * pitch_in + (x + i)];
        }
    }

    buffer_out[y * pitch_out + x] = min;
}

uint8_t max(std::uint8_t *array, int length) {
    uint8_t max = 0;
    for (int i = 0; i < length; i++) {
        if (array[i] > max)
            max = array[i];
    }
    return max;
}

__global__ void binarize(uint8_t* image, int width, int height, uint8_t threshold, size_t pitch) {
    
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    image[y * pitch + x] = image[y * pitch + x] >= threshold ? 255 : 0;
}

__global__ void connect_components(uint8_t* buffer_in, uint8_t* buffer_out,
                                int width, int height,
                                size_t pitch_in, size_t pitch_out)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int count = 0;

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (x + i >= 0 && x + i < width && y + j >= 0 && y + j < height && buffer_in[(y + j) * pitch_in + (x + i)] > 0)
                count++;
        }
    }

    buffer_out[y * pitch_out + x] = count >= 2 ? 255 : buffer_in[y * pitch_in + x];
}

__global__ void upscale(uint8_t* patches, uint8_t* output, int width, int height, int patch_size, size_t pitch_in, size_t pitch_out) {
    
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int patch_x = min(x / patch_size, width / patch_size);
    int patch_y = min(y / patch_size, width / patch_size); 

    output[y * pitch_out + x] = patches[patch_y * pitch_in + patch_x];
}

int main(int argc, char **argv) {

    // Read Image
    int width, height;
    png_bytep* row_pointers;
    int patch_size = 32;

    const char *filename = argv[1];
    read_png(filename, &width, &height, &row_pointers);
    
    int patch_height = height / patch_size;
    int patch_width = width / patch_size;
    
    // To Grayscale
    uint8_t* host_gray_img = img_to_grayscale(row_pointers, width, height);
    uint8_t* dev_gray_img;
    size_t pitch;
    cudaError_t rc = cudaMallocPitch(&dev_gray_img, &pitch, width * sizeof(uint8_t), height);
    if (rc)
        std::cerr << cudaGetErrorString(rc);
    rc = cudaMemcpy2D(dev_gray_img, pitch, host_gray_img, width * sizeof(uint8_t), width * sizeof(uint8_t), height, cudaMemcpyHostToDevice);
    if (rc)
        std::cerr << cudaGetErrorString(rc) << std::endl;

    // Free original memory
    for (int y = 0; y < height; y++) {
        free(row_pointers[y]);
    }
    free(row_pointers);
    
    // Kernel execution Parameters
    int bsize = 32;
    dim3 dimBlock(bsize, bsize);
    dim3 dimGrid(width / bsize, height / bsize);

    // Sobel filters
    uint8_t* sobel_x;
    size_t sobelx_pitch;
    uint8_t* sobel_y;
    size_t sobely_pitch;

    rc = cudaMallocPitch(&sobel_x, &sobelx_pitch, width * sizeof(uint8_t), height);
    rc = cudaMallocPitch(&sobel_y, &sobely_pitch, width * sizeof(uint8_t), height);

    compute_sobel<<<dimGrid, dimBlock>>>(dev_gray_img,
                                            sobel_x, 
                                            sobel_y, 
                                            width, 
                                            height, 
                                            pitch, 
                                            sobelx_pitch, 
                                            sobely_pitch);

    if (cudaPeekAtLastError())
        std::cerr << "Computation Error";

    // Free grayscale
    free(host_gray_img);
    cudaFree(dev_gray_img);

    // Patch calculation
    uint8_t* buffer1;
    size_t buffer1_pitch;
    rc = cudaMallocPitch(&buffer1, &buffer1_pitch, patch_width * sizeof(uint8_t), patch_height);
    compute_response<<<dimGrid, dimBlock>>>(sobel_x, sobel_y, buffer1,
                                                width, height, patch_size,
                                                sobelx_pitch, sobely_pitch, buffer1_pitch);

    // Free Sobel Images
    cudaFree(sobel_x);
    cudaFree(sobel_y);

    // Dilation + Erosion
    uint8_t* buffer2;
    size_t buffer2_pitch;
    rc = cudaMallocPitch(&buffer2, &buffer2_pitch, patch_width * sizeof(uint8_t), patch_height);
    compute_dilation<<<dimGrid, dimBlock>>>(buffer1, buffer2, patch_width, patch_height, buffer1_pitch, buffer2_pitch);
    compute_erosion<<<dimGrid, dimBlock>>>(buffer2, buffer1, patch_width, patch_height, buffer2_pitch, buffer1_pitch);

    // Thresholding
    uint8_t* image_host = (uint8_t*) malloc(patch_height * patch_width * sizeof(uint8_t));
    rc = cudaMemcpy2D(image_host, patch_width * sizeof(uint8_t), buffer1, buffer1_pitch, patch_width * sizeof(uint8_t), patch_height, cudaMemcpyDeviceToHost);
    std::uint8_t threshold = max(image_host, patch_height * patch_width) / 2;
    free(image_host);
    binarize<<<dimGrid, dimBlock>>>(buffer1, patch_width, patch_height, threshold, buffer1_pitch);

    // Connect Components
    connect_components<<<dimGrid, dimBlock>>>(buffer1, buffer2, patch_width, patch_height,buffer1_pitch, buffer2_pitch);

    // Upscale Image to native res
    uint8_t* result_dev;
    size_t result_pitch;
    rc = cudaMallocPitch(&result_dev, &result_pitch, width * sizeof(uint8_t), height);
    upscale<<<dimGrid, dimBlock>>>(buffer2, result_dev, width, height, patch_size, buffer2_pitch, result_pitch);

    // Output Result
    uint8_t* result_image = (uint8_t*) malloc(width * height * sizeof(uint8_t));
    rc = cudaMemcpy2D(result_image, width * sizeof(uint8_t), result_dev, result_pitch, width * sizeof(uint8_t), height, cudaMemcpyDeviceToHost);
    if (rc)
        std::cerr << cudaGetErrorString(rc) << std::endl;
    write_png(result_image, "res/result.png", width, height);
    
    // Free Memory
    cudaFree(result_dev);
    cudaFree(buffer1);
    cudaFree(buffer2);
    free(result_image);
}