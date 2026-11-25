#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>

extern "C" int jelly_prepare_rgb(const uint8_t* source,int width,int height,
                                  size_t stride,uint8_t* output,int out_width,
                                  int out_height,char* error,size_t error_cap) {
    try {
        if(!source || !output || width<1 || height<1 || out_width<1 || out_height<1 ||
           width>4096 || height>4096 || out_width>4096 || out_height>4096 ||
           stride<static_cast<size_t>(width)*3)
            throw std::runtime_error("invalid image dimensions");
        cv::Mat bgr(height,width,CV_8UC3,const_cast<uint8_t*>(source),stride);
        cv::Mat canvas(out_height,out_width,CV_8UC3,cv::Scalar(0,0,0));
        double scale=std::min(static_cast<double>(out_width)/width,
                              static_cast<double>(out_height)/height);
        // Python round() uses ties to even, matching nearbyint under default rounding mode.
        int w=std::min(out_width,std::max(1,static_cast<int>(std::nearbyint(width*scale))));
        int h=std::min(out_height,std::max(1,static_cast<int>(std::nearbyint(height*scale))));
        cv::Mat resized;
        cv::resize(bgr,resized,cv::Size(w,h),0,0,cv::INTER_LINEAR);
        resized.copyTo(canvas(cv::Rect((out_width-w)/2,(out_height-h)/2,w,h)));
        cv::Mat rgb(out_height,out_width,CV_8UC3,output);
        cv::cvtColor(canvas,rgb,cv::COLOR_BGR2RGB);
        return 0;
    } catch(const std::exception& ex) {
        if(error && error_cap) std::snprintf(error,error_cap,"%s",ex.what());
        return -1;
    }
}
