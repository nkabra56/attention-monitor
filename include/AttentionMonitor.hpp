#pragma once

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/face.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <deque>
#include <vector>
#include <utility>
#include <string>

enum class AttentionState { ATTENTIVE, DISTRACTED, DROWSY };

struct MonitorAlert {
    int level    = 0;
    bool newEvent = false;
};

class AttentionMonitor {
public:
    AttentionMonitor(const std::string& ssdProto,
                     const std::string& ssdModel,
                     const std::string& lbfModel);

    std::pair<AttentionState,MonitorAlert>
    update(const cv::Mat& frame, cv::Mat& out);

    // reset warning counts & timers
    void resetWarnings();

private:
    bool findFace(const cv::Mat& frame, cv::Rect& faceBox);
    bool findLandmarks(const cv::Mat& gray,
                       const cv::Rect& faceBox,
                       std::vector<cv::Point2f>& lm);
    double eyeAspectRatio(const std::vector<cv::Point2f>& lm);
    double headYaw(const std::vector<cv::Point2f>& lm);

    // core models
    cv::dnn::Net faceNet;
    cv::Ptr<cv::face::Facemark> facemark;

    // timing & thresholds
    using Clock   = std::chrono::steady_clock;
    using Seconds = std::chrono::duration<double>;
    Clock::time_point stateStart, warning1Time;
    int warningLevel = 0;

    static constexpr double warning1Lock   = 5.0;   // after W1, lock DISTRACTED for 5s
    static constexpr double yawThreshEnter = 25.0;  // degrees
    static constexpr double yawThreshExit  = 15.0;
    static constexpr double earThresh      = 0.20;  // eye aspect ratio
    static constexpr int    earFramesThresh= 10;    // ~1/3s @30fps
    static constexpr int    longCloseFrames= 150;   // 5s
    static constexpr int    yawFramesThresh= 15;    // 0.5s

    // state counters
    int eyesClosedFrames = 0, poseOffFrames = 0, longEyesClosed = 0;
    int noLmFrames = 0, noLmMaxFrames = 30;

    // PnP & filtering
    cv::Mat cameraMatrix, distCoeffs;
    bool camInit = false;
    std::deque<double> yawHistory;
    size_t yawHistSize     = 5;
    double yawSmoothAlpha  = 0.6, yawSmoothed = 0.0;

    AttentionState lastState = AttentionState::ATTENTIVE;
};
