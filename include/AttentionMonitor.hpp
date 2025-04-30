// File: include/AttentionMonitor.hpp

#ifndef ATTENTION_MONITOR_HPP
#define ATTENTION_MONITOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/dnn.hpp>
#include <chrono>
#include <deque>

// Driver attention states
enum class AttentionState { ATTENTIVE, DISTRACTED };

// Alert info: level (0/1/2) and whether a new popup should fire
struct MonitorAlert {
    int  level    = 0;
    bool newEvent = false;
};

using Clock   = std::chrono::steady_clock;
using Seconds = std::chrono::duration<double>;

class AttentionMonitor {
public:
    AttentionMonitor(const std::string& ssdProto,
                     const std::string& ssdModel,
                     const std::string& lbfModel);

    // Process one frame: fills 'annotated' and returns (state, alert)
    std::pair<AttentionState, MonitorAlert>
        update(const cv::Mat& frame, cv::Mat& annotated);

    // Reset warning level & timer (used after Warning2)
    void resetWarnings() {
        warningLevel = 0;
        stateStart   = Clock::now();
    }

    // After Warning1, keep DISTRACTED for this many seconds
    Clock::time_point warning1Time;
    static constexpr double warning1LockSec = 5.0;

private:
    // face detector + landmark fitter
    cv::dnn::Net faceNet;
    cv::Ptr<cv::face::Facemark> facemark;

    // frame counters for eyes closed, pose off, missing landmarks
    int eyesClosedFrames = 0;
    int poseOffFrames    = 0;
    int noLmFrames       = 0;

    // smoothing buffers for yaw
    double yawSmoothed        = 0.0;
    static constexpr double yawSmoothAlpha = 0.1;
    std::deque<double> yawHistory;
    static constexpr int   yawHistSize     = 5;

    // PnP camera intrinsics (initialized on first use)
    bool    camInit = false;
    cv::Mat cameraMatrix, distCoeffs;

    // thresholds for yaw hysteresis (degrees)
    static constexpr double yawThreshEnter = 18.0;
    static constexpr double yawThreshExit  = 10.0;

    // state machine & timers
    AttentionState    lastState   = AttentionState::ATTENTIVE;
    Clock::time_point stateStart;
    int               warningLevel = 0;

    // EAR threshold + required frame counts
    static constexpr double earThresh       = 0.20;
    static constexpr int    earFramesThresh = 10;
    static constexpr int    yawFramesThresh = 15;
    static constexpr int    noLmMaxFrames   = 3;

    // ── NEW: hybrid detect/track members ─────────────────────────
    int                         frameCounter   = 0;     // total frames seen
    static constexpr int        detectInterval = 7;     // detect every 7th frame
    cv::Mat                     prevGray;               // last gray frame
    std::vector<cv::Point2f>    prevPts;                // previous landmarks

    // helper routines
    bool findFace(const cv::Mat& frame, cv::Rect& faceBox);
    bool findLandmarks(const cv::Mat& gray,
                       const cv::Rect& faceBox,
                       std::vector<std::vector<cv::Point2f>>& lms);
    double eyeAspectRatio(const std::vector<cv::Point2f>& lm);
    double headYaw(const std::vector<cv::Point2f>& lm);
};

#endif // ATTENTION_MONITOR_HPP
