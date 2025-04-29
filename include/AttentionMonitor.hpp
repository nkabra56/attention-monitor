#ifndef ATTENTION_MONITOR_HPP
#define ATTENTION_MONITOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/dnn.hpp>

enum class AttentionState { ATTENTIVE, DISTRACTED, DROWSY };

class AttentionMonitor {
public:
    AttentionMonitor(const std::string& ssdProto,
                     const std::string& ssdModel,
                     const std::string& lbfModel);

    /** Process a new video frame.
     *  @param frame      BGR input image
     *  @param annotated  Output image with overlays (can be the same Mat)
     *  @return current attention state
     */
    AttentionState update(const cv::Mat& frame, cv::Mat& annotated);

private:
    cv::dnn::Net faceNet;                       // SSD face detector
    cv::Ptr<cv::face::Facemark> facemark;       // 68-landmark predictor

    int eyesClosedFrames  = 0;                  // consecutive frames
    int poseOffFrames     = 0;
    int longEyesClosed    = 0;                  // 5 s window

    // internal helpers
    bool findFace(const cv::Mat& frame, cv::Rect& faceBox);
    bool findLandmarks(const cv::Mat& gray,
                       const cv::Rect& faceBox,
                       std::vector<cv::Point2f>& landmarks);

    double eyeAspectRatio(const std::vector<cv::Point2f>& lm);
    double headYaw(const std::vector<cv::Point2f>& lm);
};

#endif
