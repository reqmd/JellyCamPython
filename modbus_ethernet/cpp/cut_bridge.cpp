#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
double millis(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b-a).count();
}
struct Crop {
    int x0, x1;
    int64_t row0, row1;
    int area, width, height;
    cv::Mat pixels;
};
struct Result {
    std::vector<Crop> crops;
    size_t output_bytes = 0;
    int at_edge = 0;
    int invalid = 0;
    double mask_ms = 0, morph_ms = 0, components_ms = 0, crop_ms = 0;
};
void err(char* out, size_t size, const char* message) {
    if (out && size) std::snprintf(out, size, "%s", message);
}
}  // namespace

extern "C" {
struct JellyCutItem {
    int x0, x1;
    int64_t row0, row1;
    int area, width, height;
    size_t offset, bytes;
};
struct JellyCutStats {
    int count, at_edge, invalid;
    double mask_ms, morph_ms, components_ms, crop_ms;
    size_t output_bytes;
};

int jelly_cut(const uint8_t* rows, int rows_h, int rows_w, size_t row_stride,
              int x_offset, const int64_t* runs, size_t n_runs,
              int64_t lo, int64_t hi, int left, int right, int halo,
              int roi_x0, int roi_x1, int edge, int min_area,
              int median_kernel, int open_kernel, int close_kernel,
              void** handle, JellyCutStats* stats, char* error, size_t error_size) {
    try {
        if (!handle || !stats || !rows || !runs || rows_h <= 0 || rows_w <= 0 ||
            rows_h != hi-lo || row_stride < static_cast<size_t>(rows_w)*3 ||
            left >= right || halo < 0 || open_kernel < 1 || close_kernel < 1 ||
            median_kernel < 1 || median_kernel % 2 == 0 ||
            open_kernel % 2 == 0 || close_kernel % 2 == 0)
            throw std::runtime_error("invalid snapshot geometry or kernels");
        *handle = nullptr;
        auto result = std::make_unique<Result>();
        const int height = rows_h + 2 * halo;
        const int width = right-left;
        if (height <= 0 || width <= 0 || width > 2048 || height > 4096 ||
            n_runs > 200000)
            throw std::runtime_error("snapshot exceeds safe limits");
        const auto t0 = Clock::now();
        cv::Mat mask = cv::Mat::zeros(height, width, CV_8UC1);
        const int64_t start = lo-halo;
        for (size_t i=0; i<n_runs; ++i) {
            const int64_t row = runs[i*3], a = runs[i*3+1], b = runs[i*3+2];
            if (row < start || row >= hi+halo || a < left || b > right || a >= b)
                throw std::runtime_error("run lies outside snapshot mask");
            std::memset(mask.ptr<uint8_t>(static_cast<int>(row-start)) + (a-left),
                        255, static_cast<size_t>(b-a));
        }
        const auto t1 = Clock::now();
        cv::Mat opened, closed;
        cv::Mat open_k = cv::Mat::ones(open_kernel, open_kernel, CV_8UC1);
        cv::Mat close_k = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                    {close_kernel,close_kernel});
        cv::morphologyEx(mask, opened, cv::MORPH_OPEN, open_k);
        cv::morphologyEx(opened, closed, cv::MORPH_CLOSE, close_k);
        const auto t2 = Clock::now();
        cv::Mat labels, comp_stats, centroids;
        const int n = cv::connectedComponentsWithStats(closed, labels, comp_stats,
                                                       centroids, 8);
        const auto t3 = Clock::now();
        cv::Mat rgb(rows_h, rows_w, CV_8UC3, const_cast<uint8_t*>(rows), row_stride);
        for (int i=1; i<n; ++i) {
            const int x=comp_stats.at<int>(i,cv::CC_STAT_LEFT);
            const int y=comp_stats.at<int>(i,cv::CC_STAT_TOP);
            const int w=comp_stats.at<int>(i,cv::CC_STAT_WIDTH);
            const int h=comp_stats.at<int>(i,cv::CC_STAT_HEIGHT);
            const int area=comp_stats.at<int>(i,cv::CC_STAT_AREA);
            const int a=left+x, b=a+w;
            const int64_t r0=start+y, r1=r0+h;
            if (area < min_area) continue;
            if (a < roi_x0+edge || b > roi_x1-edge) {
                ++result->at_edge;
                continue;
            }
            if (r0 < lo || r1 > hi) {
                ++result->invalid;
                continue;
            }
            if (a < x_offset || b > x_offset+rows_w)
                throw std::runtime_error("crop lies outside RGB snapshot");
            cv::Mat roi(rgb, cv::Rect(a-x_offset, static_cast<int>(r0-lo), w, h));
            cv::Mat filtered;
            if (median_kernel > 1) cv::medianBlur(roi, filtered, median_kernel);
            else filtered = roi;
            cv::Mat crop;
            if (area == w*h) {
                crop = filtered.clone();
            } else {
                cv::Mat component;
                const cv::Rect bounds(x,y,w,h);
                if (n==2) component = closed(bounds).clone();
                else cv::compare(labels(bounds), cv::Scalar(i), component, cv::CMP_EQ);
                std::vector<std::vector<cv::Point>> contours;
                cv::findContours(component, contours, cv::RETR_EXTERNAL,
                                 cv::CHAIN_APPROX_SIMPLE);
                cv::Mat filled = cv::Mat::zeros(h,w,CV_8UC1);
                cv::drawContours(filled,contours,-1,cv::Scalar(255),cv::FILLED);
                filtered.copyTo(crop,filled);
            }
            const size_t bytes = static_cast<size_t>(w)*h*3;
            if (!crop.isContinuous() || bytes > (64u << 20) - result->output_bytes)
                throw std::runtime_error("crop output is non-contiguous or exceeds 64 MiB");
            result->output_bytes += bytes;
            result->crops.push_back({a,b,r0,r1,area,w,h,std::move(crop)});
        }
        const auto t4 = Clock::now();
        result->mask_ms=millis(t0,t1);
        result->morph_ms=millis(t1,t2);
        result->components_ms=millis(t2,t3);
        result->crop_ms=millis(t3,t4);
        *stats={static_cast<int>(result->crops.size()),result->at_edge,result->invalid,
                result->mask_ms,result->morph_ms,result->components_ms,result->crop_ms,
                result->output_bytes};
        *handle=result.release();
        return 0;
    } catch (const std::exception& e) {
        err(error,error_size,e.what());
        return -1;
    } catch (...) {
        err(error,error_size,"unknown C++ error");
        return -1;
    }
}

int jelly_cut_export(void* handle, JellyCutItem* items, size_t item_capacity,
                     uint8_t* pixels, size_t pixel_capacity,
                     char* error, size_t error_size) {
    try {
        if (!handle || !items || !pixels)
            throw std::runtime_error("null export buffer");
        const auto& result=*static_cast<Result*>(handle);
        if (item_capacity < result.crops.size() ||
            pixel_capacity < result.output_bytes)
            throw std::runtime_error("export buffer too small");
        size_t offset=0;
        for (size_t i=0; i<result.crops.size(); ++i) {
            const auto& c=result.crops[i];
            const size_t bytes=static_cast<size_t>(c.width)*c.height*3;
            std::memcpy(pixels+offset,c.pixels.ptr<uint8_t>(0),bytes);
            items[i]={c.x0,c.x1,c.row0,c.row1,c.area,c.width,c.height,
                      offset,bytes};
            offset+=bytes;
        }
        return 0;
    } catch (const std::exception& e) {
        err(error,error_size,e.what());
        return -1;
    } catch (...) {
        err(error,error_size,"unknown C++ export error");
        return -1;
    }
}
void jelly_cut_destroy(void* handle) { delete static_cast<Result*>(handle); }
} // extern "C"
