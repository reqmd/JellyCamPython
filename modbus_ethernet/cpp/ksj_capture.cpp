#include <algorithm>
#include <cstddef>
#include <cstdint>

// KSJ_CaptureRgbData(int camera_index, unsigned char *destination), Linux ABI.
// Use the entry point from the SDK instance already initialized by the server.
using CaptureRgb = int (*)(int, unsigned char*);

extern "C" int jelly_capture_rgb(CaptureRgb capture, int index, unsigned char* dst) {
    return capture(index, dst);
}

extern "C" void jelly_flip_rows(unsigned char* data, std::size_t rows,
                                std::size_t row_bytes) {
    for (std::size_t i = 0; i < rows / 2; ++i) {
        auto* first = data + i * row_bytes;
        auto* last = data + (rows - 1 - i) * row_bytes;
        std::swap_ranges(first, first + row_bytes, last);
    }
}
