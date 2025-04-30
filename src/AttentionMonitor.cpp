#include "AttentionMonitor.hpp"
#include <iomanip>
#include <sstream>

using namespace cv;

static constexpr double earThresh       = 0.20;
static constexpr int    earFramesThresh = 10;     // ~⅓ s @30 FPS
static constexpr int    yawFramesThresh = 15;     // ~½ s
static constexpr double yawThresh       = 15.0;   // tighter
static constexpr int    noLmMaxFrames   = 3;      // treat loss as distracted

// ───────────────── constructor ─────────────────
AttentionMonitor::AttentionMonitor(const std::string& ssdProto,
                                   const std::string& ssdModel,
                                   const std::string& lbfModel)
{
    faceNet  = dnn::readNetFromCaffe(ssdProto, ssdModel);
    facemark = cv::face::FacemarkLBF::create();
    facemark->loadModel(lbfModel);
    stateStart = Clock::now();
}

// ───────────────── helpers ─────────────────
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
    auto d=[&](int i,int j){ return norm(lm[i]-lm[j]); };
    return (d(37,41)+d(38,40)) / (2.0*d(36,39));
}

double AttentionMonitor::headYaw(const std::vector<Point2f>& lm)
{
    std::vector<Point3f> model = {
        {  0,   0,   0},   // nose 30
        {  0,-330, -65},   // chin 8  (6-point PnP needs ≥6)
        {-225, 170,-135},  // left eye 36
        { 225, 170,-135},  // right eye 45
        {-150,-150,-125},  // mouth 48
        { 150,-150,-125}   // mouth 54
    };
    std::vector<Point2f> img = { lm[30], lm[8], lm[36], lm[45], lm[48], lm[54] };

    Mat rvec,tvec,rot;
    bool ok = solvePnP(model,img,Mat::eye(3,3,CV_64F),noArray(),rvec,tvec,false,
                       SOLVEPNP_ITERATIVE);
    if(!ok) return 0.0;
    Rodrigues(rvec,rot);
    return atan2(rot.at<double>(2,0), rot.at<double>(0,0))*180.0/CV_PI;
}

// ───────────────── update ─────────────────
std::pair<AttentionState, MonitorAlert>
AttentionMonitor::update(const Mat& frame, Mat& out)
{
    frame.copyTo(out);
    Mat gray; cvtColor(frame,gray,COLOR_BGR2GRAY);

    // ── detect face & landmarks ──
    Rect face;
    bool haveFace = findFace(frame,face);

    std::vector<std::vector<Point2f>> lms;
    bool haveLm = haveFace && findLandmarks(gray,face,lms);

    bool eyesClosed=false, headAway=false;
    double earVal = 0.0, yawVal = 0.0;

    if(haveLm && !lms.empty()){
        noLmFrames = 0;
        const auto& lm = lms[0];
        earVal  = eyeAspectRatio(lm);
        yawVal  = std::abs(headYaw(lm));
        eyesClosed = earVal < earThresh;
        headAway   = yawVal > yawThresh;
        rectangle(out,face,Scalar(0,255,0));
    } else {
        noLmFrames++;
        headAway = noLmFrames >= noLmMaxFrames;    // lost tracking
    }

    // ── smooth per-frame flags ──
    eyesClosedFrames = eyesClosed ? eyesClosedFrames+1 : 0;
    poseOffFrames    = headAway   ? poseOffFrames+1   : 0;
    bool shortClose = eyesClosedFrames >= earFramesThresh;
    bool poseOff    = poseOffFrames    >= yawFramesThresh;

    AttentionState curr = (shortClose || poseOff)
                          ? AttentionState::DISTRACTED
                          : AttentionState::ATTENTIVE;

    // ── timing-based warnings ──
    MonitorAlert al{warningLevel,false};
    double elapsed = Seconds(Clock::now() - stateStart).count();

    if(curr == lastState){
        if(curr == AttentionState::DISTRACTED){
            if(warningLevel==0 && elapsed>=5.0){ warningLevel=1; al={1,true}; }
            else if(warningLevel==1 && elapsed>=20.0){ warningLevel=2; al={2,true}; }
        } else {
            if(warningLevel==1 && elapsed>=16.0) warningLevel=0;
            if(warningLevel==2 && elapsed>=25.0) warningLevel=0;
        }
    } else {
        stateStart = Clock::now();
        lastState  = curr;
    }
    al.level = warningLevel;

    // ── overlays ──
    putText(out, curr==AttentionState::ATTENTIVE ? "ATTENTIVE" : "DISTRACTED",
            {30,40},FONT_HERSHEY_SIMPLEX,1.2,
            curr==AttentionState::ATTENTIVE?Scalar(0,255,0):Scalar(0,0,255),2);

    std::ostringstream dbg;
    dbg << std::fixed << std::setprecision(2)
        << "EAR:" << earVal << "  yaw:" << yawVal;
    putText(out, dbg.str(), {30,70}, FONT_HERSHEY_PLAIN, 1.4,
            Scalar(255,255,0), 2);

    return {curr,al};
}
