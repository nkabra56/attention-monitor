#ifndef ATTENTION_MONITOR_HPP
#define ATTENTION_MONITOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/dnn.hpp>
#include <chrono>
#include <deque>

// Possible states of driver attention
enum class AttentionState { ATTENTIVE, DISTRACTED };

// Holds the warning level (0 = none, 1 or 2) and flag for a new event
struct MonitorAlert {
    int  level    = 0;
    bool newEvent = false;
};

using Clock   = std::chrono::steady_clock;
using Seconds = std::chrono::duration<double>;

// This class does face detection, landmark fitting, head-pose and eye-blink checks,
// applies smoothing and hysteresis, and tracks timing to fire warnings.
class AttentionMonitor {
public:
    // Load the face detector (proto + model) and landmark model
    AttentionMonitor(const std::string& ssdProto,
                     const std::string& ssdModel,
                     const std::string& lbfModel);

    // Process one camera frame: draws on 'annotated' and returns state + any warning
    std::pair<AttentionState, MonitorAlert>
        update(const cv::Mat& frame, cv::Mat& annotated);

    // Clear the warning level and restart the timing
    void resetWarnings() {
        warningLevel = 0;
        stateStart   = Clock::now();
    }

    // After warning 1, keep DISTRACTED for this many seconds
    Clock::time_point warning1Time;
    static constexpr double warning1LockSec = 5.0;

private:
    // DNN face detector
    cv::dnn::Net faceNet;
    // Landmark detector
    cv::Ptr<cv::face::Facemark> facemark;

    // Counters for how many frames eyes closed, pose off, or no landmarks
    int eyesClosedFrames = 0;
    int poseOffFrames    = 0;
    int noLmFrames       = 0;

    // For smoothing the yaw angle
    double yawSmoothed        = 0.0;
    static constexpr double yawSmoothAlpha = 0.1;
    std::deque<double> yawHistory;
    static constexpr int   yawHistSize     = 5;

    // Camera parameters for solvePnP (initialized on first use)
    bool    camInit = false;
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;

    // Hysteresis thresholds for head yaw in degrees
    static constexpr double yawThreshEnter = 18.0;
    static constexpr double yawThreshExit  = 10.0;

    // State machine and timing
    AttentionState    lastState   = AttentionState::ATTENTIVE;
    Clock::time_point stateStart;
    int               warningLevel = 0;

    // Eye aspect ratio threshold and required frame counts
    static constexpr double earThresh       = 0.20;
    static constexpr int    earFramesThresh = 10;
    static constexpr int    yawFramesThresh = 15;
    static constexpr int    noLmMaxFrames   = 3;

    // Detect a face and return its rectangle
    bool findFace(const cv::Mat& frame, cv::Rect& faceBox);

    // Fit landmarks inside a face rectangle
    bool findLandmarks(const cv::Mat& gray,
                       const cv::Rect& faceBox,
                       std::vector<std::vector<cv::Point2f>>& lms);

    // Compute eye aspect ratio from landmarks
    double eyeAspectRatio(const std::vector<cv::Point2f>& lm);

    // Not used (PnP is done inline in update)
    double headYaw(const std::vector<cv::Point2f>& lm);
};

#endif // ATTENTION_MONITOR_HPP
