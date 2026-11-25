#include <algorithm>
#include <cstdint>
#include <cstdlib>

// Input is a full BGR block. The background is the detector's contiguous ROI.
// Scratch is private to the detector and holds every eighth pixel for one channel.
extern "C" int jelly_detect(const uint8_t* image, int rows, int stride,
                            int x0, int width, const int16_t* bg, int threshold,
                            int step, int touch, int* scratch, uint8_t* mask,
                            uint8_t* touched, uint8_t* dense) {
    const int samples = (width + 7) / 8;
    const int mid = samples / 2;
    bool occupied = false;
    for (int y = 0; y < rows; ++y) {
        const auto* row = image + static_cast<size_t>(y) * stride + 3 * x0;
        int median[3];
        for (int c = 0; c < 3; ++c) {
            for (int i = 0; i < samples; ++i) {
                const int offset = 24 * i + c;
                scratch[i] = row[offset] - bg[offset];
            }
            std::nth_element(scratch, scratch + mid, scratch + samples);
            median[c] = scratch[mid];
            if (samples % 2 == 0) {
                median[c] = (median[c] + *std::max_element(scratch, scratch + mid)) / 2;
            }
        }
        auto* dst = mask + static_cast<size_t>(y) * width;
        int hits = 0, sampled = 0, next_sample = 0;
        for (int x = 0; x < width; ++x) {
            const int k = 3 * x;
            dst[x] = std::abs(row[k] - bg[k] - median[0]) > threshold ||
                     std::abs(row[k+1] - bg[k+1] - median[1]) > threshold ||
                     std::abs(row[k+2] - bg[k+2] - median[2]) > threshold;
            hits += dst[x];
            if (x == next_sample) {
                sampled += dst[x];
                next_sample += step;
            }
        }
        occupied |= hits != 0;
        touched[y] = sampled * step >= touch;
        dense[y] = 4 * hits > 3 * width;
    }
    return occupied;
}
