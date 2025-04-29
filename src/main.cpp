#include "AttentionMonitor.hpp"
#include <iostream>

int main()
{
    std::string root = "../models/";
    AttentionMonitor monitor(root+"deploy.prototxt",
                             root+"res10_300x300_ssd_iter_140000.caffemodel",
                             root+"lbfmodel.yaml");

    cv::VideoCapture cam(0, cv::CAP_DSHOW);
    if (!cam.isOpened()) {
        std::cerr << "Camera not found\n";
        return -1;
    }

    while (true) {
        cv::Mat frame, annotated;
        cam >> frame;
        if (frame.empty()) break;

        monitor.update(frame, annotated);
        cv::imshow("Attention Monitor", annotated);
        if (cv::waitKey(1) == 27) break;   // Esc quits
    }
    return 0;
}
