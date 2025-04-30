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
    // DNN face detector + LBF facemark
    cv::dnn::Net faceNet;
    cv::Ptr<cv::face::Facemark> facemark;

    // Counters for eye‐closure, pose off, missing landmarks
    int eyesClosedFrames = 0;
    int poseOffFrames    = 0;
    int noLmFrames       = 0;

    // Yaw smoothing
    double yawSmoothed        = 0.0;
    static constexpr double yawSmoothAlpha = 0.1;
    std::deque<double> yawHistory;
    static constexpr int   yawHistSize     = 5;

    // Camera intrinsics for PnP
    bool    camInit = false;
    cv::Mat cameraMatrix, distCoeffs;

    // Hysteresis thresholds for head yaw
    static constexpr double yawThreshEnter = 18.0;
    static constexpr double yawThreshExit  = 10.0;

    // State machine & timers
    AttentionState    lastState   = AttentionState::ATTENTIVE;
    Clock::time_point stateStart;
    int               warningLevel = 0;

    // EAR threshold + required frames for blink detection
    static constexpr double earThresh       = 0.20;
    static constexpr int    earFramesThresh = 3;   // << faster response
    static constexpr int    yawFramesThresh = 15;
    static constexpr int    noLmMaxFrames   = 3;

    // Hybrid detect/track members (optical flow)
    int                         frameCounter   = 0;
    static constexpr int        detectInterval = 7;
    cv::Mat                     prevGray;
    std::vector<cv::Point2f>    prevPts;

    // Helpers
    bool findFace(const cv::Mat& frame, cv::Rect& faceBox);
    bool findLandmarks(const cv::Mat& gray,
                       const cv::Rect& faceBox,
                       std::vector<std::vector<cv::Point2f>>& lms);

    // Compute EAR for one eye, starting at index i0 (6 points: i0..i0+5)
    double eyeAspectRatio(const std::vector<cv::Point2f>& lm, int i0);

    // Convenience wrappers for left/right eye
    inline double leftEyeEAR(const std::vector<cv::Point2f>& lm) {
        return eyeAspectRatio(lm, 36);
    }
    inline double rightEyeEAR(const std::vector<cv::Point2f>& lm) {
        return eyeAspectRatio(lm, 42);
    }

    // Not used: PnP is done inline
    double headYaw(const std::vector<cv::Point2f>& lm);
};

#endif // ATTENTION_MONITOR_HPP
