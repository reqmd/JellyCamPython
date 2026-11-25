CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -fPIC
OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4)
OPENCV_LIBS = $(shell pkg-config --libs opencv4)

.PHONY: all clean
all: libjelly_rknn.so libjelly_cut.so libjelly_tracker.so libjelly_prepare.so

libjelly_rknn.so: cpp_rknn_bridge.cpp rknn_api.h
	$(CXX) $(CXXFLAGS) -I. -shared $< /lib/librknnrt.so -o $@

libjelly_cut.so: cut_bridge.cpp
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -shared $< $(OPENCV_LIBS) -o $@

libjelly_tracker.so: tracker_bridge.cpp
	$(CXX) $(CXXFLAGS) -shared $< -o $@

libjelly_prepare.so: prepare_bridge.cpp
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -shared $< $(OPENCV_LIBS) -o $@

clean:
	rm -f libjelly_rknn.so libjelly_cut.so libjelly_tracker.so libjelly_prepare.so *.o
