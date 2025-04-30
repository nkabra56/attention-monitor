// File: src/AttentionMonitor.cpp

#include "AttentionMonitor.hpp"
#include <iomanip>
#include <sstream>

using namespace cv;

AttentionMonitor::AttentionMonitor(const std::string& ssdProto,
                                   const std::string& ssdModel,
                                   const std::string& lbfModel)
{
    // Load face detector and landmark model
    faceNet  = dnn::readNetFromCaffe(ssdProto, ssdModel);
    facemark = cv::face::FacemarkLBF::create();
    facemark->loadModel(lbfModel);

    stateStart   = Clock::now();
    warning1Time = stateStart;
}

bool AttentionMonitor::findFace(const Mat& frame, Rect& faceBox)
{
    Mat blob = dnn::blobFromImage(frame, 1.0, Size(300,300),
                                  Scalar(104,177,123), true, false);
    faceNet.setInput(blob);
    Mat dets = faceNet.forward();
    const float* d = dets.ptr<float>(0,0);
    if(d[2] < 0.5f) return false;
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
    auto d = [&](int i,int j){ return norm(lm[i]-lm[j]); };
    return (d(37,41) + d(38,40)) / (2.0 * d(36,39));
}

double AttentionMonitor::headYaw(const std::vector<Point2f>&) {
    return 0.0; // unused; PnP is inline in update()
}

std::pair<AttentionState, MonitorAlert>
AttentionMonitor::update(const Mat& frame, Mat& out)
{
    // copy frame for annotation
    frame.copyTo(out);

    // 1) Preprocess: grayscale + CLAHE for lighting normalization
    Mat gray; cvtColor(frame, gray, COLOR_BGR2GRAY);
    static auto clahe = createCLAHE(2.0, Size(8,8));
    Mat grayCLAHE; clahe->apply(gray, grayCLAHE);
    gray = grayCLAHE;

    // 2) Hybrid detect vs. track landmarks
    frameCounter++;
    std::vector<Point2f> lmPts;
    bool haveLm = false;

    if(frameCounter % detectInterval == 1 || prevPts.empty()) {
        // full detection on interval
        Rect face;
        bool okFace = findFace(frame, face);
        std::vector<std::vector<Point2f>> allLms;
        bool okLm = okFace && findLandmarks(gray, face, allLms);
        if(okLm && !allLms.empty()) {
            lmPts   = allLms[0];
            prevPts = lmPts;
            gray.copyTo(prevGray);
            rectangle(out, face, Scalar(0,255,0));
            haveLm  = true;
        }
    } else {
        // track via optical flow
        std::vector<uchar> status;
        std::vector<float> errs;
        std::vector<Point2f> curPts;
        calcOpticalFlowPyrLK(prevGray, gray, prevPts, curPts, status, errs);
        for(int i=0; i<(int)curPts.size(); ++i)
            if(status[i]) lmPts.push_back(curPts[i]);
        if((int)lmPts.size() >= 6) {
            haveLm   = true;
            prevPts  = lmPts;
            gray.copyTo(prevGray);
        }
    }

    // Prepare flags and metrics
    bool eyesClosed = false, headAway = false;
    double earVal = 0.0, rawYaw = 0.0, yawVal = 0.0;

    if(haveLm) {
        noLmFrames = 0;
        earVal = eyeAspectRatio(lmPts);

        // 3) Solve PnP with RANSAC for head yaw
        if(!camInit) {
            double fx = frame.cols, fy = frame.cols;
            double cx = frame.cols*0.5, cy = frame.rows*0.5;
            cameraMatrix = (Mat_<double>(3,3)
                             << fx,0,cx, 0,fy,cy, 0,0,1);
            distCoeffs = Mat::zeros(5,1,CV_64F);
            camInit = true;
        }
        std::vector<Point3f> modelPts = {
            {  0,   0,    0},  // nose
            {  0, -330,  -65},  // chin
            {-225, 170, -135},  // left eye
            { 225, 170, -135},  // right eye
            {-150,-150, -125},  // left mouth
            { 150,-150, -125}   // right mouth
        };
        // map indices from lmPts (must be same ordering as detection)
        std::vector<int> idx = {30,8,36,45,48,54};
        std::vector<Point2f> imgPts;
        for(int j: idx) if(j < (int)lmPts.size()) imgPts.push_back(lmPts[j]);
        if(imgPts.size()==6) {
            Mat rvec,tvec,rot,inliers;
            bool ok = solvePnPRansac(
                modelPts, imgPts,
                cameraMatrix, distCoeffs,
                rvec, tvec, false,
                100, 8.0, 0.99,
                inliers, SOLVEPNP_ITERATIVE
            );
            if(ok) {
                Rodrigues(rvec, rot);
                rawYaw = fabs( atan2(rot.at<double>(2,0),
                                     rot.at<double>(0,0))
                              *180.0/CV_PI );
            }
        }

        // 4) Median + EMA smoothing
        yawHistory.push_back(rawYaw);
        if((int)yawHistory.size() > yawHistSize)
            yawHistory.pop_front();
        std::vector<double> tmp(yawHistory.begin(), yawHistory.end());
        sort(tmp.begin(), tmp.end());
        yawVal = tmp[tmp.size()/2];
        yawSmoothed = yawSmoothAlpha*yawVal + (1.0-yawSmoothAlpha)*yawSmoothed;
        yawVal = yawSmoothed;

        // 5) Hysteresis for pose
        bool enter = yawVal > yawThreshEnter;
        bool exit  = yawVal > yawThreshExit;
        headAway   = (lastState==AttentionState::ATTENTIVE ? enter : exit);
        eyesClosed = earVal < earThresh;
    } else {
        noLmFrames++;
        headAway = noLmFrames >= noLmMaxFrames;
    }

    // 6) Per-frame smoothing
    eyesClosedFrames = eyesClosed ? eyesClosedFrames+1 : 0;
    poseOffFrames    = headAway   ? poseOffFrames+1   : 0;
    bool shortClose  = eyesClosedFrames >= earFramesThresh;
    bool poseOff     = poseOffFrames    >= yawFramesThresh;
    AttentionState curr = (shortClose || poseOff)
                          ? AttentionState::DISTRACTED
                          : AttentionState::ATTENTIVE;

    // 7) Warning FSM & lockout
    MonitorAlert al{warningLevel,false};
    auto now     = Clock::now();
    double elapsed = Seconds(now - stateStart).count();
    if(curr == lastState) {
        if(curr == AttentionState::DISTRACTED) {
            if(warningLevel==0 && elapsed>=5.0) {
                warningLevel=1; al={1,true};
                warning1Time = now;
            }
            else if(warningLevel==1 && elapsed>=20.0) {
                warningLevel=2; al={2,true};
            }
        } else {
            if(warningLevel==1 && elapsed>=16.0) warningLevel=0;
            if(warningLevel==2 && elapsed>=25.0) warningLevel=0;
        }
    } else {
        stateStart = now;
        lastState  = curr;
    }
    al.level = warningLevel;
    if(warningLevel==1) {
        if(Seconds(now - warning1Time).count() < warning1LockSec)
            curr = AttentionState::DISTRACTED;
    }

    // 8) Draw overlays
    putText(out, curr==AttentionState::ATTENTIVE ? "ATTENTIVE" : "DISTRACTED",
            {30,60}, FONT_HERSHEY_SIMPLEX, 2.5,
            curr==AttentionState::ATTENTIVE ? Scalar(0,255,0) : Scalar(0,0,255),
            4);
    std::ostringstream dbg;
    dbg<<std::fixed<<std::setprecision(2)
       <<"EAR:"<<earVal<<" yaw:"<<yawVal;
    putText(out, dbg.str(), {30,110},
            FONT_HERSHEY_SIMPLEX, 1.8, Scalar(255,255,0), 2);

    return {curr, al};
}
