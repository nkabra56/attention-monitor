#include "AttentionMonitor.hpp"
#include <opencv2/dnn.hpp>
#include <opencv2/face.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>

using namespace cv;

// Constructor: load DNN face detector & LBF landmark model
AttentionMonitor::AttentionMonitor(const std::string& ssdProto,
                                   const std::string& ssdModel,
                                   const std::string& lbfModel)
{
    faceNet  = dnn::readNetFromCaffe(ssdProto, ssdModel);
    facemark = cv::face::FacemarkLBF::create();
    facemark->loadModel(lbfModel);

    stateStart   = Clock::now();
    warning1Time = stateStart;
}

// Reset warnings & timers
void AttentionMonitor::resetWarnings() {
    warningLevel = 0;
    stateStart   = Clock::now();
    warning1Time = stateStart;
}

// Face detection via SSD
bool AttentionMonitor::findFace(const Mat& frame, Rect& faceBox) {
    Mat blob = dnn::blobFromImage(frame, 1.0, Size(300,300),
                                  Scalar(104,177,123), true, false);
    faceNet.setInput(blob);
    Mat dets = faceNet.forward();
    const float* d = dets.ptr<float>(0,0);
    if (d[2] < 0.5f) return false;
    faceBox = Rect(Point(d[3]*frame.cols, d[4]*frame.rows),
                   Point(d[5]*frame.cols, d[6]*frame.rows))
              & Rect(0,0,frame.cols,frame.rows);
    return true;
}

// Landmark fit
bool AttentionMonitor::findLandmarks(const Mat& gray,
                                    const Rect& faceBox,
                                    vector<Point2f>& lm) {
    return facemark->fit(gray, vector<Rect>{faceBox}, lm);
}

// Eye aspect ratio calculation
double AttentionMonitor::eyeAspectRatio(const vector<Point2f>& lm) {
    auto dist = [&](int i,int j){ return norm(lm[i]-lm[j]); };
    return (dist(37,41) + dist(38,40)) / (2.0*dist(36,39));
}

// Head yaw via solvePnP
double AttentionMonitor::headYaw(const vector<Point2f>& lm) {
    vector<Point3f> model = {
        {0,0,0}, {-225,170,-135}, {225,170,-135},
        {-150,-150,-125}, {150,-150,-125}
    };
    vector<Point2f> img = { lm[30], lm[36], lm[45], lm[48], lm[54] };
    Mat rvec,tvec,rot;
    if (!camInit) {
        double fx =  frame.cols, fy = frame.cols;
        double cx = frame.cols/2.0, cy = frame.rows/2.0;
        cameraMatrix = (Mat_<double>(3,3) << fx,0,cx, 0,fy,cy, 0,0,1);
        distCoeffs    = Mat::zeros(5,1,CV_64F);
        camInit       = true;
    }
    solvePnP(model, img, cameraMatrix, distCoeffs, rvec, tvec);
    Rodrigues(rvec, rot);
    return atan2(rot.at<double>(2,0), rot.at<double>(0,0)) * 180.0/CV_PI;
}

// Main update: copy frame, detect face+landmarks, track EAR & yaw, manage warnings
pair<AttentionState,MonitorAlert>
AttentionMonitor::update(const Mat& frame, Mat& out) {
    frame.copyTo(out);
    Mat gray; cvtColor(frame, gray, COLOR_BGR2GRAY);

    Rect face;
    bool haveFace = findFace(frame, face);

    vector<Point2f> lm;
    bool haveLm = haveFace && findLandmarks(gray, face, lm);

    bool eyesClosed=false, headAway=false;
    double earVal=0.0, rawYaw=0.0, yawVal=0.0;

    if (haveLm) {
        earVal = eyeAspectRatio(lm);
        rawYaw = headYaw(lm);
        rectangle(out, face, Scalar(0,255,0));

        // median filter on yaw
        yawHistory.push_back(rawYaw);
        if (yawHistory.size()>yawHistSize) yawHistory.pop_front();
        vector<double> tmp(yawHistory.begin(), yawHistory.end());
        sort(tmp.begin(), tmp.end());
        yawVal = tmp[tmp.size()/2];

        // exponential moving average
        yawSmoothed = yawSmoothAlpha*yawVal + (1.0-yawSmoothAlpha)*yawSmoothed;
        yawVal = yawSmoothed;

        eyesClosed = earVal < earThresh;
        headAway   = yawVal > yawThreshEnter;
    } else {
        noLmFrames++;
        headAway = noLmFrames >= noLmMaxFrames;
    }

    // frame counters
    eyesClosedFrames = eyesClosed ? eyesClosedFrames+1 : 0;
    poseOffFrames    = headAway   ? poseOffFrames+1   : 0;
    longEyesClosed   = eyesClosed ? longEyesClosed+1  : 0;

    bool shortClose = eyesClosedFrames >= earFramesThresh;
    bool longClose  = longEyesClosed   >= longCloseFrames;
    bool poseOff    = poseOffFrames    >= yawFramesThresh;

    AttentionState curr = AttentionState::ATTENTIVE;
    if (poseOff || shortClose) curr = AttentionState::DISTRACTED;
    if (longClose)            curr = AttentionState::DROWSY;

    // warning timing logic
    MonitorAlert al{warningLevel,false};
    auto now = Clock::now();
    double elapsed = Seconds(now - stateStart).count();

    if (curr == lastState) {
        if (curr == AttentionState::DISTRACTED) {
            if (warningLevel==0 && elapsed >= 5.0) {
                warningLevel = 1; al = {1,true};
                warning1Time = now;
            }
            else if (warningLevel==1 && elapsed >= 20.0) {
                warningLevel = 2; al = {2,true};
            }
        } else {
            if (warningLevel==1 && elapsed >= 16.0) warningLevel=0;
            if (warningLevel==2 && elapsed >= 25.0) warningLevel=0;
        }
    } else {
        stateStart = now;
        lastState  = curr;
    }
    al.level = warningLevel;

    // enforce 5s lock after W1
    if (warningLevel==1) {
        double sinceW1 = Seconds(now - warning1Time).count();
        if (sinceW1 < warning1Lock) curr = AttentionState::DISTRACTED;
    }

    // overlay state text
    Scalar c = (curr==AttentionState::ATTENTIVE)   ? Scalar(0,255,0)
              : (curr==AttentionState::DISTRACTED) ? Scalar(0,255,255)
                                                    : Scalar(0,0,255);
    putText(out,
            (curr==AttentionState::ATTENTIVE)   ? "ATTENTIVE" :
            (curr==AttentionState::DISTRACTED) ? "DISTRACTED" : "DROWSY",
            Point(30,60), FONT_HERSHEY_SIMPLEX, 2.0, c, 4);

    // debug overlay
    std::ostringstream dbg;
    dbg<<std::fixed<<std::setprecision(2)
       <<"EAR:"<<earVal<<" YAW:"<<yawVal;
    putText(out, dbg.str(), Point(30,120),
            FONT_HERSHEY_SIMPLEX, 1.2, Scalar(255,255,0),2);

    return {curr,al};
}
