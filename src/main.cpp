#include <cstddef>
#include <string>

#include "cpu.hpp"


int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Missing input img\n");
        return 1;
    }
    const char *filename = argv[1];
    detect_barcode(filename);

    return 0;
}