#include "AttentionMonitor.hpp"

using namespace cv;

static constexpr double earThresh        = 0.20;   // eyes-closed limit
static constexpr int    earFramesThresh  = 10;     // 10 frames ≈ ⅓ s @30 FPS
static constexpr int    longCloseFrames  = 150;    // 5 s
static constexpr double yawThresh        = 25.0;   // degrees
static constexpr int    yawFramesThresh  = 15;     // 0.5 s

AttentionMonitor::AttentionMonitor(const std::string& ssdProto,
                                   const std::string& ssdModel,
                                   const std::string& lbfModel)
{
    faceNet = dnn::readNetFromCaffe(ssdProto, ssdModel);
    facemark = cv::face::FacemarkLBF::create();
    facemark->loadModel(lbfModel);
}

bool AttentionMonitor::findFace(const Mat& frame, Rect& faceBox)
{
    Mat blob = dnn::blobFromImage(frame, 1.0, {300,300},
                                  {104,177,123}, true, false);
    faceNet.setInput(blob);
    Mat detections = faceNet.forward();
    const float* data = detections.ptr<float>(0,0);
    float conf = data[2];
    if (conf < 0.5f) return false;

    int x1 = static_cast<int>(data[3] * frame.cols);
    int y1 = static_cast<int>(data[4] * frame.rows);
    int x2 = static_cast<int>(data[5] * frame.cols);
    int y2 = static_cast<int>(data[6] * frame.rows);
    faceBox = Rect(Point(x1,y1), Point(x2,y2)) & Rect(0,0,frame.cols,frame.rows);
    return true;
}

bool AttentionMonitor::findLandmarks(const Mat& gray,
                                     const Rect& faceBox,
                                     std::vector<Point2f>& landmarks)
{
    std::vector<Rect> faces{faceBox};
    return facemark->fit(gray, faces, landmarks);
}

double AttentionMonitor::eyeAspectRatio(const std::vector<Point2f>& lm)
{
    auto d = [&](int i,int j){ return norm(lm[i] - lm[j]); };
    // left eye 36-41 in 68-pt scheme
    double vert = d(37,41) + d(38,40);
    double horiz = 2.0 * d(36,39);
    return vert / horiz;
}

double AttentionMonitor::headYaw(const std::vector<Point2f>& lm)
{
    // 3-D model points (mm)
    std::vector<Point3f> model = {
        {0,0,0},          // nose tip 30
        {-225,170,-135},  // left eye corner 36
        {225,170,-135},   // right eye corner 45
        {-150,-150,-125}, // left mouth 48
        {150,-150,-125}   // right mouth 54
    };
    std::vector<Point2f> image = { lm[30], lm[36], lm[45], lm[48], lm[54] };

    Mat rvec, tvec;
    solvePnP(model, image, Mat::eye(3,3,CV_64F), noArray(), rvec, tvec, false,
             SOLVEPNP_ITERATIVE);

    Mat rot;
    Rodrigues(rvec, rot);
    double yaw = atan2(rot.at<double>(2,0), rot.at<double>(0,0)) * 180.0 / CV_PI;
    return yaw;
}

AttentionState AttentionMonitor::update(const Mat& frame, Mat& annotated)
{
    frame.copyTo(annotated);
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);

    Rect faceBox;
    if (!findFace(frame, faceBox)) {
        // no face resets counters
        eyesClosedFrames = poseOffFrames = longEyesClosed = 0;
        return AttentionState::DISTRACTED;
    }
    rectangle(annotated, faceBox, Scalar(0,255,0));

    std::vector<Point2f> lm;
    if (!findLandmarks(gray, faceBox, lm)) return AttentionState::DISTRACTED;

    double ear = eyeAspectRatio(lm);
    bool eyesClosed = ear < earThresh;
    eyesClosedFrames   = eyesClosed ? eyesClosedFrames + 1 : 0;
    longEyesClosed     = eyesClosed ? longEyesClosed + 1 : 0;

    double yaw = headYaw(lm);
    bool headAway  = std::abs(yaw) > yawThresh;
    poseOffFrames  = headAway ? poseOffFrames + 1 : 0;

    bool shortClose = eyesClosedFrames >= earFramesThresh;
    bool longClose  = longEyesClosed   >= longCloseFrames;
    bool poseOff    = poseOffFrames    >= yawFramesThresh;

    AttentionState state = AttentionState::ATTENTIVE;
    if (poseOff || shortClose) state = AttentionState::DISTRACTED;
    if (longClose)             state = AttentionState::DROWSY;

    Scalar col = state==AttentionState::ATTENTIVE ? Scalar(0,255,0) :
                 state==AttentionState::DISTRACTED ? Scalar(0,255,255) :
                 Scalar(0,0,255);
    putText(annotated,
            state==AttentionState::ATTENTIVE ? "ATTENTIVE" :
            state==AttentionState::DISTRACTED ? "DISTRACTED" : "DROWSY",
            Point(30,40), FONT_HERSHEY_SIMPLEX, 1.2, col, 2);
    return state;
}
