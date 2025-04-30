#include "AttentionMonitor.hpp"
#include <iostream>
#ifdef _WIN32
#include <windows.h>
#endif

static void fireAlert(int level)
{
#ifdef _WIN32
    if(level==1){
        MessageBoxA(nullptr,
            "WARNING LEVEL 1:\nDriver distracted > 5 s",
            "Driver Attention Monitor", MB_ICONWARNING | MB_TOPMOST);
    }else if(level==2){
        MessageBoxA(nullptr,
            "WARNING LEVEL 2:\nDriver distracted > 20 s",
            "Driver Attention Monitor", MB_ICONERROR | MB_TOPMOST);
        Beep(500,1000);   // 500 Hz, 1 s
    }
#else
    std::cerr<<"WARNING LEVEL "<<level<<"\n";
#endif
}

int main()
{
    std::string modelsDir = "../../models/";   // adjust if needed
    AttentionMonitor mon(modelsDir+"deploy.prototxt",
                         modelsDir+"res10_300x300_ssd_iter_140000.caffemodel",
                         modelsDir+"lbfmodel.yaml");

    cv::VideoCapture cam(0, cv::CAP_DSHOW);
    if(!cam.isOpened()){ std::cerr<<"No camera found\n"; return -1; }

    while(true){
        cv::Mat f,a; cam>>f; if(f.empty()) break;
        auto [state,alert] = mon.update(f,a);
        if(alert.newEvent && alert.level>0) fireAlert(alert.level);

        cv::imshow("Driver Attention Monitor",a);
        if(cv::waitKey(1)==27) break;   // ESC
    }
    return 0;
}
