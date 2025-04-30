#include "AttentionMonitor.hpp"
#include <iomanip>
#include <sstream>
#include <algorithm>

using namespace cv;

AttentionMonitor::AttentionMonitor(const std::string& ssdProto,
                                   const std::string& ssdModel,
                                   const std::string& lbfModel)
{
    // load DNN + facemark
    faceNet  = dnn::readNetFromCaffe(ssdProto, ssdModel);
    facemark = face::FacemarkLBF::create();
    facemark->loadModel(lbfModel);

    // init FSM timers
    stateStart   = Clock::now();
    warning1Time = stateStart;

    // init Kalman filters
    for(auto kfPtr : { &kfYaw, &kfPitch, &kfMar }) {
        kfPtr->transitionMatrix    = Mat::eye(1,1,CV_32F);
        kfPtr->measurementMatrix   = Mat::eye(1,1,CV_32F);
        setIdentity(kfPtr->processNoiseCov,    Scalar::all(1e-2));
        setIdentity(kfPtr->measurementNoiseCov,Scalar::all(1e-1));
        kfPtr->statePost = Mat::zeros(1,1,CV_32F);
    }
    kfMeas  = Mat::zeros(1,1,CV_32F);
    kfState = Mat::zeros(1,1,CV_32F);
}

bool AttentionMonitor::findFace(const Mat& frame, Rect& faceBox)
{
    Mat blob = dnn::blobFromImage(frame, 1.0, Size(300,300),
                                  Scalar(104,177,123), true, false);
    faceNet.setInput(blob);
    Mat dets = faceNet.forward();
    auto d = dets.ptr<float>(0,0);
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

double AttentionMonitor::eyeAspectRatio(const std::vector<Point2f>& lm, int i0)
{
    auto d = [&](int i,int j){ return norm(lm[i0+i] - lm[i0+j]); };
    return (d(1,5) + d(2,4)) / (2.0 * d(0,3));
}

bool AttentionMonitor::isGazeDistracted(const Mat& gray,
                                        const std::vector<Point2f>& lm)
{
    for(int eyeStart : {36, 42}) {
        std::vector<Point2f> eyePts(
            lm.begin()+eyeStart, lm.begin()+eyeStart+6);
        Rect r = boundingRect(eyePts);
        r.x = max(r.x-2,0); r.y = max(r.y-2,0);
        r.width  = min(r.width+4, gray.cols - r.x);
        r.height = min(r.height+4, gray.rows - r.y);
        Mat roi = gray(r);

        Mat thr;
        threshold(roi, thr, 50, 255, THRESH_BINARY_INV);

        std::vector<std::vector<Point>> C;
        findContours(thr, C, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        if(C.empty()) continue;

        int idxMax=0; double areaMax=0;
        for(int i=0;i<(int)C.size();i++){
            double a = contourArea(C[i]);
            if(a>areaMax){ areaMax=a; idxMax=i; }
        }
        Moments m = moments(C[idxMax]);
        if(m.m00 <= 0) continue;

        Point2f center(m.m10/m.m00, m.m01/m.m00);
        double normX = (center.x - roi.cols/2.0) / (roi.cols/2.0);
        if(fabs(normX) > gazeThresh) return true;
    }
    return false;
}

double AttentionMonitor::headYaw(const std::vector<Point2f>&) {
    return 0.0;
}

std::pair<AttentionState,MonitorAlert>
AttentionMonitor::update(const Mat& frame, Mat& out)
{
    // copy frame for annotations
    frame.copyTo(out);

    //── preprocess: grayscale + CLAHE ─────────────────────────────
    Mat gray; cvtColor(frame, gray, COLOR_BGR2GRAY);
    static auto clahe = createCLAHE(2.0, Size(8,8));
    Mat grayCLAHE; clahe->apply(gray, grayCLAHE);
    gray = grayCLAHE;

    //── hybrid detect/track landmarks ────────────────────────────
    frameCounter++;
    std::vector<Point2f> lmPts;
    bool haveLm = false;
    if(frameCounter % detectInterval == 1 || prevPts.empty()) {
        Rect face;
        if(findFace(frame, face)) {
            std::vector<std::vector<Point2f>> allLms;
            if(findLandmarks(gray, face, allLms) && !allLms.empty()) {
                lmPts   = allLms[0];
                prevPts = lmPts;
                gray.copyTo(prevGray);
                rectangle(out, face, Scalar(0,255,0));
                haveLm  = true;
            }
        }
    } else {
        std::vector<uchar> status;
        std::vector<float> errs;
        std::vector<Point2f> curPts;
        calcOpticalFlowPyrLK(prevGray, gray, prevPts, curPts, status, errs);
        for(int i=0;i<(int)curPts.size();i++)
            if(status[i]) lmPts.push_back(curPts[i]);
        if((int)lmPts.size() >= 6) {
            haveLm   = true;
            prevPts  = lmPts;
            gray.copyTo(prevGray);
        }
    }

    //── compute raw metrics ───────────────────────────────────────
    bool eyesClosed=false, headAway=false, gazeAway=false;
    double earVal=0, rawYaw=0, yawVal=0, rawPitch=0, pitchVal=0, rawMar=0, marVal=0;
    if(haveLm) {
        noLmFrames = 0;

        // blink
        double earL = eyeAspectRatio(lmPts, 36);
        double earR = eyeAspectRatio(lmPts, 42);
        earVal = (earL + earR) * 0.5;

        // gaze
        gazeAway = isGazeDistracted(gray, lmPts);

        // PnP for yaw & pitch
        if(!camInit) {
            double fx=frame.cols, fy=frame.cols;
            double cx=frame.cols*0.5, cy=frame.rows*0.5;
            cameraMatrix = (Mat_<double>(3,3)<<fx,0,cx, 0,fy,cy, 0,0,1);
            distCoeffs=Mat::zeros(5,1,CV_64F);
            camInit=true;
        }
        std::vector<Point3f> modelPts = {
            {0,0,0},{0,-330,-65},
            {-225,170,-135},{225,170,-135},
            {-150,-150,-125},{150,-150,-125}
        };
        std::vector<int> idx={30,8,36,45,48,54};
        std::vector<Point2f> imgPts;
        for(int j:idx) imgPts.push_back(lmPts[j]);

        Mat rvec,tvec,rot,inliers;
        if(solvePnPRansac(modelPts, imgPts,
              cameraMatrix, distCoeffs,
              rvec, tvec, false, 100, 8.0, 0.99, inliers,
              SOLVEPNP_ITERATIVE))
        {
            Rodrigues(rvec, rot);
            rawYaw = fabs(atan2(rot.at<double>(2,0),
                                rot.at<double>(0,0)) *180.0/CV_PI);
            rawPitch = atan2(
                -rot.at<double>(2,1),
                sqrt( rot.at<double>(2,0)*rot.at<double>(2,0)
                    +rot.at<double>(2,2)*rot.at<double>(2,2))
            ) * 180.0/CV_PI;
        }

        // yawn (MAR)
        rawMar = (
            norm(lmPts[62]-lmPts[66]) +
            norm(lmPts[63]-lmPts[65])
        ) / (2.0 * norm(lmPts[60]-lmPts[64]));
    } else {
        noLmFrames++;
        headAway = noLmFrames >= noLmMaxFrames;
    }

    //── smoothing pipelines ───────────────────────────────────────
    // yaw median + EMA
    yawHistory.push_back(rawYaw);
    if((int)yawHistory.size()>yawHistSize) yawHistory.pop_front();
    std::vector<double> tmp(yawHistory.begin(), yawHistory.end());
    std::sort(tmp.begin(), tmp.end());
    yawVal = tmp[tmp.size()/2];
    yawSmoothed = yawSmoothAlpha*yawVal + (1.0-yawSmoothAlpha)*yawSmoothed;

    // pitch median + EMA
    pitchHistory.push_back(rawPitch);
    if((int)pitchHistory.size()>pitchHistSize) pitchHistory.pop_front();
    tmp.assign(pitchHistory.begin(), pitchHistory.end());
    std::sort(tmp.begin(), tmp.end());
    pitchVal = tmp[tmp.size()/2];
    pitchSmoothed = pitchSmoothAlpha*pitchVal + (1.0-pitchSmoothAlpha)*pitchSmoothed;

    // Kalman filter wrapper
    auto applyKF = [&](KalmanFilter& kf, double v){
        kfMeas.at<float>(0) = float(v);
        kf.correct(kfMeas);
        kfState = kf.predict();
        return double(kfState.at<float>(0));
    };

    double yawKF   = applyKF(kfYaw,   yawSmoothed);
    double pitchKF = applyKF(kfPitch, pitchSmoothed);
    double marKF   = applyKF(kfMar,   rawMar);

    //── frame‐based thresholds ────────────────────────────────────
    eyesClosed = earVal < earThresh;
    eyesClosedFrames = eyesClosed? eyesClosedFrames+1 : 0;
    poseOffFrames    = (yawKF > yawThreshEnter)? poseOffFrames+1 : 0;
    headNodFrames    = (pitchKF > pitchThreshEnter)? headNodFrames+1 : 0;
    marFrames        = (marKF > marThresh)? marFrames+1 : 0;

    bool blinkOff    = eyesClosedFrames >= earFramesThresh;
    bool poseOffYaw  = poseOffFrames    >= yawFramesThresh;
    bool poseOffPitch= headNodFrames    >= pitchFramesThresh;
    bool mouthOff    = marFrames        >= marFramesThresh;

    // combine all distraction signals
    AttentionState curr = (blinkOff || poseOffYaw || poseOffPitch
                           || gazeAway || mouthOff)
                          ? AttentionState::DISTRACTED
                          : AttentionState::ATTENTIVE;

    //── warning FSM & lockout ────────────────────────────────────
    MonitorAlert al{warningLevel, false};
    auto now = Clock::now();
    double elapsed = Seconds(now - stateStart).count();

    if(curr == lastState) {
        if(curr == AttentionState::DISTRACTED) {
            if(warningLevel == 0 && elapsed >= 5.0) {
                warningLevel = 1; al = {1,true}; warning1Time = now;
            }
            else if(warningLevel == 1 && elapsed >= 20.0) {
                warningLevel = 2; al = {2,true};
            }
        } else {
            if(warningLevel == 1 && elapsed >= 16.0) warningLevel = 0;
            if(warningLevel == 2 && elapsed >= 25.0) warningLevel = 0;
        }
    } else {
        stateStart = now; lastState = curr;
    }
    al.level = warningLevel;
    if(warningLevel == 1 &&
       Seconds(now - warning1Time).count() < warning1LockSec)
    {
        curr = AttentionState::DISTRACTED;
    }

    //── draw overlays ────────────────────────────────────────────
    putText(out,
        curr==AttentionState::ATTENTIVE ? "ATTENTIVE" : "DISTRACTED",
        {30,60}, FONT_HERSHEY_SIMPLEX, 2.5,
        curr==AttentionState::ATTENTIVE ? Scalar(0,255,0) : Scalar(0,0,255),
        4);

    std::ostringstream dbg;
    dbg << std::fixed << std::setprecision(2)
        << "EAR:" << earVal
        << " yaw:" << yawKF
        << " pitch:" << pitchKF
        << " gaze:" << (gazeAway ? "OFF" : "OK")
        << " yawn:" << marKF;
    putText(out, dbg.str(), {30,110},
            FONT_HERSHEY_SIMPLEX, 1.8, Scalar(255,255,0), 2);

    return {curr, al};
}
