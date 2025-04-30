#include "AttentionMonitor.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <thread>
#ifdef _WIN32
  #include <windows.h>
#endif

#ifdef _WIN32
static void timedMessageBox(const char* txt,const char* ttl,
                            UINT flags,DWORD ms)
{
    std::thread([=]{
        Sleep(ms);
        if(HWND h=FindWindowA(nullptr,ttl))
            PostMessageA(h, WM_CLOSE, 0, 0);
    }).detach();
    MessageBoxA(nullptr, txt, ttl, flags | MB_TOPMOST);
}
#endif

static void fireAlert(AttentionMonitor &mon, int level)
{
#ifdef _WIN32
    std::thread([&mon,level]{
        if(level==1)
            timedMessageBox("WARNING LEVEL 1: Distracted >5s",
                            "Attention Warning", MB_ICONWARNING, 5000);
        else if(level==2) {
            timedMessageBox("WARNING LEVEL 2: Distracted >20s",
                            "Attention Warning", MB_ICONERROR, 5000);
            Beep(500,1000);
            mon.resetWarnings();
        }
    }).detach();
#else
    std::cerr<<"WARNING LEVEL "<<level<<"\n";
    if(level==2) mon.resetWarnings();
#endif
}

int main()
{
    AttentionMonitor mon(
        "./models/deploy.prototxt",
        "./models/res10_300x300_ssd_iter_140000.caffemodel",
        "./models/lbfmodel.yaml"
    );

    cv::VideoCapture cam(0, cv::CAP_DSHOW);
    if(!cam.isOpened()){
        std::cerr<<"Cannot open camera\n";
        return -1;
    }

    const std::string win="Driver Attention Monitor";
    cv::namedWindow(win, cv::WINDOW_AUTOSIZE | cv::WINDOW_GUI_EXPANDED);

    while(true){
        cv::Mat frame, out;
        cam >> frame;
        if(frame.empty()) break;

        auto [state,alert] = mon.update(frame, out);
        if(alert.newEvent && alert.level>0)
            fireAlert(mon, alert.level);

        cv::imshow(win, out);
        int k = cv::waitKey(1);
        if(k==27) break;
        if(cv::getWindowProperty(win, cv::WND_PROP_VISIBLE)<1) break;
    }

    cam.release();
    cv::destroyAllWindows();
    return 0;
}
