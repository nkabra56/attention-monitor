#include "AttentionMonitor.hpp"
#include <iostream>
#include <thread>
#ifdef _WIN32
  #include <windows.h>
#endif

// Show a message box that closes itself after ms milliseconds.
// Runs the close logic on a separate thread.
#ifdef _WIN32
static void timedMessageBox(const char* txt,
                            const char* ttl,
                            UINT flags,
                            DWORD ms)
{
    std::thread([=]{
        Sleep(ms);
        if (HWND h = FindWindowA(nullptr, ttl))
            PostMessageA(h, WM_CLOSE, 0, 0);
    }).detach();

    MessageBoxA(nullptr, txt, ttl, flags | MB_TOPMOST);
}
#endif

// Fire an alert without blocking the main loop.
// For warning 2, reset warnings after the popup closes.
static void fireAlert(AttentionMonitor &mon, int level)
{
#ifdef _WIN32
    std::thread([&mon,level]{
        if (level==1) {
            timedMessageBox(
              "WARNING LEVEL 1:\nDriver distracted > 5 s",
              "Attention Warning",
              MB_ICONWARNING,
              5000
            );
        }
        else if (level==2) {
            timedMessageBox(
              "WARNING LEVEL 2:\nDriver distracted > 20 s",
              "Attention Warning",
              MB_ICONERROR,
              5000
            );
            Beep(500,1000);
            mon.resetWarnings();
        }
    }).detach();
#else
    std::cerr << "WARNING LEVEL " << level << "\n";
    if (level==2) mon.resetWarnings();
#endif
}

int main()
{
    // Path to the models folder (adjust if needed)
    std::string modelsDir = "../../models/";

    // Create the attention monitor
    AttentionMonitor mon(
        modelsDir + "deploy.prototxt",
        modelsDir + "res10_300x300_ssd_iter_140000.caffemodel",
        modelsDir + "lbfmodel.yaml"
    );

    // Open the default camera
    cv::VideoCapture cam(0, cv::CAP_DSHOW);
    if (!cam.isOpened()) {
        std::cerr << "No camera found\n";
        return -1;
    }

    // Compute window size (75% of screen) and center it
#ifdef _WIN32
    int sw = GetSystemMetrics(SM_CXSCREEN);
    int sh = GetSystemMetrics(SM_CYSCREEN);
    int ww = int(sw * 0.75), wh = int(sh * 0.75);
    int px = (sw - ww) / 2, py = (sh - wh) / 2;
#else
    int ww = 1280, wh = 720, px = 100, py = 80;
#endif

    const std::string winName = "Driver Attention Monitor";
    cv::namedWindow(winName,
                    cv::WINDOW_NORMAL | cv::WINDOW_GUI_EXPANDED);
    cv::resizeWindow(winName, ww, wh);
    cv::moveWindow(winName, px, py);

    // Main loop: grab frame, update monitor, show alerts and image
    while (true) {
        cv::Mat frame, annotated;
        cam >> frame;
        if (frame.empty()) break;

        auto [state, alert] = mon.update(frame, annotated);
        if (alert.newEvent && alert.level > 0) {
            fireAlert(mon, alert.level);
        }

        cv::imshow(winName, annotated);

        // Exit on ESC or if window is closed
        int key = cv::waitKey(1);
        if (key == 27) break;
        if (cv::getWindowProperty(winName, cv::WND_PROP_VISIBLE) < 1) break;
    }

    cam.release();
    cv::destroyAllWindows();
    return 0;
}
