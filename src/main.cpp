#include "AttentionMonitor.hpp"
#include <iostream>
#ifdef _WIN32
  #include <windows.h>
  #include <thread>
#endif

// ╭──────────────────────────────────────────────────────╮
// │ Timed message box (auto-closes after ms)             │
// ╰──────────────────────────────────────────────────────╯
#ifdef _WIN32
static void timedMessageBox(const char* txt,const char* ttl,
                            UINT flags,DWORD ms)
{
    std::thread([=]{ Sleep(ms);
        if(HWND h=FindWindowA(nullptr,ttl)) PostMessageA(h,WM_CLOSE,0,0);
    }).detach();
    MessageBoxA(nullptr,txt,ttl,flags|MB_TOPMOST);
}
#endif

static void fireAlert(int level)
{
#ifdef _WIN32
    if(level==1){
        timedMessageBox("WARNING LEVEL 1:\nDriver distracted > 5 s",
                        "Driver Attention Monitor",MB_ICONWARNING,5000);
    }else if(level==2){
        timedMessageBox("WARNING LEVEL 2:\nDriver distracted > 20 s",
                        "Driver Attention Monitor",MB_ICONERROR,5000);
        Beep(500,1000);                       // 1-s low tone
    }
#else
    std::cerr<<"WARNING LEVEL "<<level<<"\n";
#endif
}

int main()
{
    std::string m="../../models/";   // adjust if needed
    AttentionMonitor mon(m+"deploy.prototxt",
                         m+"res10_300x300_ssd_iter_140000.caffemodel",
                         m+"lbfmodel.yaml");

    cv::VideoCapture cam(0,cv::CAP_DSHOW);
    if(!cam.isOpened()){ std::cerr<<"No camera\n"; return -1; }

#ifdef _WIN32          // compute 75 % of current screen & center
    int sw=GetSystemMetrics(SM_CXSCREEN);
    int sh=GetSystemMetrics(SM_CYSCREEN);
    int ww=int(sw*0.75), wh=int(sh*0.75);
    int px=(sw-ww)/2,     py=(sh-wh)/2;
#else
    int ww=1280, wh=720, px=100, py=80;
#endif

    const std::string win="Driver Attention Monitor";
    cv::namedWindow(win,cv::WINDOW_NORMAL|cv::WINDOW_GUI_EXPANDED);
    cv::resizeWindow(win,ww,wh);
    cv::moveWindow  (win,px,py);

    while(true){
        cv::Mat frm,ann; cam>>frm; if(frm.empty()) break;
        auto [state,al]=mon.update(frm,ann);
        if(al.newEvent && al.level>0) fireAlert(al.level);

        cv::imshow(win,ann);
        int key=cv::waitKey(1);
        if(key==27) break;                       // ESC

        // ➜ exit if user closed the window
        if(cv::getWindowProperty(win,cv::WND_PROP_VISIBLE)<1) break;
    }

    cam.release();
    cv::destroyAllWindows();
    return 0;
}
