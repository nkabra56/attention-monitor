#ifndef ATTENTION_MONITOR_HPP
#define ATTENTION_MONITOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/dnn.hpp>
#include <chrono>

enum class AttentionState { ATTENTIVE, DISTRACTED };

struct MonitorAlert {
    int  level = 0;      // 0-none, 1-WL1, 2-WL2
    bool newEvent = false;
};

using Clock   = std::chrono::steady_clock;
using Seconds = std::chrono::duration<double>;

class AttentionMonitor {
public:
    AttentionMonitor(const std::string& ssdProto,
                     const std::string& ssdModel,
                     const std::string& lbfModel);

    std::pair<AttentionState, MonitorAlert>
        update(const cv::Mat& frame, cv::Mat& annotated);

private:
    cv::dnn::Net                        faceNet;
    cv::Ptr<cv::face::Facemark>         facemark;

    int eyesClosedFrames = 0;
    int poseOffFrames    = 0;

    AttentionState lastState = AttentionState::ATTENTIVE;
    Clock::time_point stateStart;
    int warningLevel = 0;

    bool findFace(const cv::Mat& frame, cv::Rect& faceBox);
    bool findLandmarks(const cv::Mat& gray,
                       const cv::Rect& faceBox,
                       std::vector<std::vector<cv::Point2f>>& lms);
    double eyeAspectRatio(const std::vector<cv::Point2f>& lm);
    double headYaw(const std::vector<cv::Point2f>& lm);
};

#endif
