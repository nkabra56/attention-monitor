#ifndef ATTENTION_MONITOR_HPP
#define ATTENTION_MONITOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/video/tracking.hpp>
#include <chrono>
#include <deque>

// Possible attention states
enum class AttentionState { ATTENTIVE, DISTRACTED };

// Warning info: level (0,1,2) and new‐event flag
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

    // Process one frame: draw into 'annotated' and return (state, alert)
    std::pair<AttentionState,MonitorAlert>
    update(const cv::Mat& frame, cv::Mat& annotated);

    // Reset warnings (called after Warning2)
    void resetWarnings() {
        warningLevel = 0;
        stateStart   = Clock::now();
    }

    // Lockout start for Warning1
    Clock::time_point warning1Time;
    static constexpr double warning1LockSec = 5.0;

private:
    // ─── Models ────────────────────────────────────────────────────
    cv::dnn::Net                      faceNet;
    cv::Ptr<cv::face::Facemark>       facemark;

    // Camera intrinsics for solvePnP
    bool    camInit       = false;
    cv::Mat cameraMatrix, distCoeffs;

    // ─── Frame counters & FSM ─────────────────────────────────────
    int eyesClosedFrames = 0;
    int poseOffFrames    = 0;
    int noLmFrames       = 0;
    AttentionState    lastState   = AttentionState::ATTENTIVE;
    Clock::time_point stateStart;
    int               warningLevel = 0;

    // ─── Blink & head‐pose thresholds ─────────────────────────────
    static constexpr double earThresh       = 0.20;
    static constexpr int    earFramesThresh = 3;
    static constexpr double yawThreshEnter  = 18.0;
    static constexpr double yawThreshExit   = 10.0;
    static constexpr int    yawFramesThresh = 15;
    static constexpr double pitchThreshEnter = 20.0;
    static constexpr double pitchThreshExit  = 15.0;
    static constexpr int    pitchFramesThresh = 15;
    static constexpr int    noLmMaxFrames   = 3;

    // ─── Gaze & yawn thresholds ───────────────────────────────────
    static constexpr double gazeThresh     = 0.35;
    static constexpr double marThresh      = 0.6;
    static constexpr int    marFramesThresh = 5;

    // ─── Hybrid detect/track ───────────────────────────────────────
    static constexpr int    detectInterval = 5;
    int                     frameCounter   = 0;
    cv::Mat                 prevGray;
    std::vector<cv::Point2f> prevPts;

    // ─── Smoothing buffers ─────────────────────────────────────────
    double yawSmoothed        = 0.0;
    static constexpr double yawSmoothAlpha = 0.1;
    std::deque<double>       yawHistory;
    static constexpr int     yawHistSize     = 5;

    double pitchSmoothed      = 0.0;
    static constexpr double pitchSmoothAlpha = 0.07;
    std::deque<double>       pitchHistory;
    static constexpr int     pitchHistSize   = 7;

    // ─── Head‐nod & yawn counters ─────────────────────────────────
    int headNodFrames = 0;
    int marFrames     = 0;

    // ─── Kalman filters ───────────────────────────────────────────
    cv::KalmanFilter kfYaw{1,1,0}, kfPitch{1,1,0}, kfMar{1,1,0};
    cv::Mat          kfMeas, kfState;

    // ─── Helpers ───────────────────────────────────────────────────
    bool findFace(const cv::Mat& frame, cv::Rect& faceBox);
    bool findLandmarks(const cv::Mat& gray,
                       const cv::Rect& faceBox,
                       std::vector<std::vector<cv::Point2f>>& lms);

    // eye‐aspect ratio on 6‐point eye starting at lm[i0..i0+5]
    double eyeAspectRatio(const std::vector<cv::Point2f>& lm, int i0);

    // returns true if either pupil is >gazeThresh away from eye center
    bool isGazeDistracted(const cv::Mat& gray,
                          const std::vector<cv::Point2f>& lm);

    // unused stub
    double headYaw(const std::vector<cv::Point2f>& lm);
};

#endif // ATTENTION_MONITOR_HPP
