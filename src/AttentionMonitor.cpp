#include "AttentionMonitor.hpp"

using namespace cv;

static constexpr double earThresh       = 0.20;
static constexpr int    earFramesThresh = 10;   // ≈⅓ s @30 FPS
static constexpr int    yawFramesThresh = 15;   // ≈½ s
static constexpr double yawThresh       = 25.0; // deg

// ───────── constructor ─────────
AttentionMonitor::AttentionMonitor(const std::string& ssdProto,
                                   const std::string& ssdModel,
                                   const std::string& lbfModel)
{
    faceNet  = dnn::readNetFromCaffe(ssdProto, ssdModel);
    facemark = cv::face::FacemarkLBF::create();
    facemark->loadModel(lbfModel);
    stateStart = Clock::now();
}

// ───────── helpers ─────────
bool AttentionMonitor::findFace(const Mat& frame, Rect& faceBox)
{
    Mat blob = dnn::blobFromImage(frame, 1.0, {300,300},
                                  {104,177,123}, true, false);
    faceNet.setInput(blob);
    Mat dets = faceNet.forward();
    const float* d = dets.ptr<float>(0,0);
    if (d[2] < 0.5f) return false;

    faceBox = Rect(Point(d[3]*frame.cols, d[4]*frame.rows),
                   Point(d[5]*frame.cols, d[6]*frame.rows))
              & Rect(0,0,frame.cols,frame.rows);
    return true;
}

bool AttentionMonitor::findLandmarks(const Mat& gray,
                                     const Rect& faceBox,
                                     std::vector<std::vector<Point2f>>& lms)
{
    return facemark->fit(gray, std::vector<Rect>{faceBox}, lms);
}

double AttentionMonitor::eyeAspectRatio(const std::vector<Point2f>& lm)
{
    auto d=[&](int i,int j){return norm(lm[i]-lm[j]);};
    return (d(37,41)+d(38,40)) / (2.0*d(36,39));
}

double AttentionMonitor::headYaw(const std::vector<Point2f>& lm)
{
    /* 3-D reference (millimetres, roughly proportional)               *
     *  nose-tip, chin, left-eye-corner, right-eye-corner,             *
     *  left-mouth-corner, right-mouth-corner   →   6 points total     */
    std::vector<Point3f> model = {
        {   0,    0,    0},   // nose  30
        {   0, -330,  -65},   // chin   8
        {-225,  170, -135},   // left eye 36
        { 225,  170, -135},   // right eye 45
        {-150, -150, -125},   // left mouth 48
        { 150, -150, -125}    // right mouth 54
    };

    std::vector<Point2f> image = {
        lm[30], lm[ 8], lm[36], lm[45], lm[48], lm[54]
    };

    cv::Mat rvec, tvec, rot;
    bool ok = solvePnP(model, image, cv::Mat::eye(3,3,CV_64F),
                       cv::noArray(), rvec, tvec, false,
                       cv::SOLVEPNP_ITERATIVE);

    if(!ok) return 0.0;             // fallback: treat as looking straight

    Rodrigues(rvec, rot);
    double yaw = atan2(rot.at<double>(2,0), rot.at<double>(0,0))
                 * 180.0 / CV_PI;
    return yaw;
}

std::pair<AttentionState, MonitorAlert>
AttentionMonitor::update(const Mat& frame, Mat& annotated)
{
    frame.copyTo(annotated);
    Mat gray; cvtColor(frame,gray,COLOR_BGR2GRAY);

    Rect face;
    bool haveFace = findFace(frame,face);
    std::vector<std::vector<Point2f>> lms;
    bool haveLm = haveFace && findLandmarks(gray,face,lms);

    bool eyesClosed=false, headAway=false;
    if(haveLm && !lms.empty()){
        const auto& lm = lms[0];
        eyesClosed = eyeAspectRatio(lm) < earThresh;
        headAway   = std::abs(headYaw(lm)) > yawThresh;
        rectangle(annotated,face,Scalar(0,255,0));
    }

    eyesClosedFrames = eyesClosed ? eyesClosedFrames+1 : 0;
    poseOffFrames    = headAway   ? poseOffFrames+1   : 0;
    bool shortClose = eyesClosedFrames >= earFramesThresh;
    bool poseOff    = poseOffFrames    >= yawFramesThresh;

    AttentionState curr = (shortClose || poseOff)
                          ? AttentionState::DISTRACTED
                          : AttentionState::ATTENTIVE;

    MonitorAlert alert{warningLevel,false};
    auto   now     = Clock::now();
    double elapsed = Seconds(now - stateStart).count();

    if(curr == lastState){
        if(curr == AttentionState::DISTRACTED){
            if(warningLevel==0 && elapsed>=5.0){ warningLevel=1; alert={1,true}; }
            else if(warningLevel==1 && elapsed>=20.0){ warningLevel=2; alert={2,true}; }
        }else{
            if(warningLevel==1 && elapsed>=16.0) warningLevel=0;
            if(warningLevel==2 && elapsed>=25.0) warningLevel=0;
        }
    }else{
        stateStart = now;
        lastState  = curr;
    }

    Scalar col = curr==AttentionState::ATTENTIVE ? Scalar(0,255,0)
                                                 : Scalar(0,0,255);
    putText(annotated,
            curr==AttentionState::ATTENTIVE ? "ATTENTIVE" : "DISTRACTED",
            {30,40},FONT_HERSHEY_SIMPLEX,1.2,col,2);

    alert.level = warningLevel;   // ensure consistency
    return {curr,alert};
}
