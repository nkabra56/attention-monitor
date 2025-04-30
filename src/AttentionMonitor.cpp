#include "AttentionMonitor.hpp"
#include <iomanip>
#include <sstream>

using namespace cv;

// Constructor: load the face and landmark models, init timers
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

// Run the SSD face detector on the frame
bool AttentionMonitor::findFace(const Mat& frame, Rect& faceBox)
{
    Mat blob = dnn::blobFromImage(frame,1.0,Size(300,300),
                                  Scalar(104,177,123),true,false);
    faceNet.setInput(blob);
    Mat dets = faceNet.forward();
    const float* d = dets.ptr<float>(0,0);
    if (d[2] < 0.5f) return false;

    faceBox = Rect(Point(d[3]*frame.cols, d[4]*frame.rows),
                   Point(d[5]*frame.cols, d[6]*frame.rows))
              & Rect(0,0,frame.cols,frame.rows);
    return true;
}

// Fit 68 facial landmarks with LBF
bool AttentionMonitor::findLandmarks(const Mat& gray,
                                     const Rect& faceBox,
                                     std::vector<std::vector<Point2f>>& lms)
{
    return facemark->fit(gray, std::vector<Rect>{faceBox}, lms);
}

// Compute Eye Aspect Ratio for blink detection
double AttentionMonitor::eyeAspectRatio(const std::vector<Point2f>& lm)
{
    auto dist = [&](int i,int j){ return norm(lm[i]-lm[j]); };
    return (dist(37,41) + dist(38,40)) / (2.0 * dist(36,39));
}

// Stub, not used
double AttentionMonitor::headYaw(const std::vector<Point2f>&) {
    return 0.0;
}

// Process one frame: detect face, landmarks, pose, eye-blink, FSM, warnings, overlays
std::pair<AttentionState, MonitorAlert>
AttentionMonitor::update(const Mat& frame, Mat& out)
{
    frame.copyTo(out);
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);

    // Detect face
    Rect face;
    bool haveFace = findFace(frame, face);

    // Detect landmarks if face found
    std::vector<std::vector<Point2f>> lms;
    bool haveLm = haveFace && findLandmarks(gray, face, lms);

    bool eyesClosed = false, headAway = false;
    double earVal = 0.0, rawYaw = 0.0, yawVal = 0.0;

    if (haveLm && !lms.empty()) {
        noLmFrames = 0;
        const auto& lm = lms[0];

        // Calculate EAR
        earVal = eyeAspectRatio(lm);
        rectangle(out, face, Scalar(0,255,0));

        // Initialize camera intrinsics once
        if (!camInit) {
            double fx = frame.cols, fy = frame.cols;
            double cx = frame.cols*0.5, cy = frame.rows*0.5;
            cameraMatrix = (Mat_<double>(3,3)
                << fx,0,cx, 0,fy,cy, 0,0,1);
            distCoeffs = Mat::zeros(5,1,CV_64F);
            camInit = true;
        }

        // Prepare 3D-2D correspondences
        std::vector<Point3f> model = {
            {0,0,0},{0,-330,-65},
            {-225,170,-135},{225,170,-135},
            {-150,-150,-125},{150,-150,-125}
        };
        std::vector<Point2f> imgPts = {
            lm[30], lm[8], lm[36],
            lm[45], lm[48], lm[54]
        };

        // Solve PnP with RANSAC to get yaw angle
        Mat rvec,tvec,rot,inliers;
        bool ok = solvePnPRansac(
            model, imgPts,
            cameraMatrix, distCoeffs,
            rvec,tvec,false,
            100,   // iterations
            8.0,   // reprojection error
            0.99,  // confidence
            inliers,
            SOLVEPNP_ITERATIVE
        );
        if (ok) {
            Rodrigues(rvec, rot);
            rawYaw = std::abs(
                atan2(rot.at<double>(2,0),
                      rot.at<double>(0,0))
                *180.0/CV_PI
            );
        }

        // Median filter on rawYaw
        yawHistory.push_back(rawYaw);
        if ((int)yawHistory.size() > yawHistSize)
            yawHistory.pop_front();
        std::vector<double> tmp(yawHistory.begin(), yawHistory.end());
        std::sort(tmp.begin(), tmp.end());
        yawVal = tmp[tmp.size()/2];

        // Exponential smoothing
        yawSmoothed = yawSmoothAlpha*yawVal +
                      (1.0-yawSmoothAlpha)*yawSmoothed;
        yawVal = yawSmoothed;

        // Hysteresis for entering/exiting distracted by head-turn
        bool enter = yawVal > yawThreshEnter;
        bool exit  = yawVal > yawThreshExit;
        headAway = (lastState == AttentionState::ATTENTIVE)
                    ? enter : exit;

        // Blink detection
        eyesClosed = earVal < earThresh;
    }
    else {
        noLmFrames++;
        headAway = noLmFrames >= noLmMaxFrames;
    }

    // Require several consecutive frames for each flag
    eyesClosedFrames = eyesClosed ? eyesClosedFrames+1 : 0;
    poseOffFrames    = headAway   ? poseOffFrames+1   : 0;
    bool blink = eyesClosedFrames >= earFramesThresh;
    bool pose  = poseOffFrames    >= yawFramesThresh;

    AttentionState curr = (blink||pose)
                          ? AttentionState::DISTRACTED
                          : AttentionState::ATTENTIVE;

    // Warning state machine
    MonitorAlert al{warningLevel,false};
    auto now = Clock::now();
    double elapsed = Seconds(now - stateStart).count();

    if (curr == lastState) {
        if (curr == AttentionState::DISTRACTED) {
            if (warningLevel==0 && elapsed>=5.0) {
                warningLevel=1;
                al = {1,true};
                warning1Time = now;  // start lockout
            }
            else if (warningLevel==1 && elapsed>=20.0) {
                warningLevel=2;
                al = {2,true};
            }
        }
        else {
            if (warningLevel==1 && elapsed>=16.0) warningLevel=0;
            if (warningLevel==2 && elapsed>=25.0) warningLevel=0;
        }
    }
    else {
        // reset on state change
        stateStart = now;
        lastState  = curr;
    }
    al.level = warningLevel;

    // Keep distracted for warning1LockSec after warning 1
    if (warningLevel==1) {
        double since1 = Seconds(now - warning1Time).count();
        if (since1 < warning1LockSec)
            curr = AttentionState::DISTRACTED;
    }

    // Draw ATTENTIVE/DISTRACTED
    putText(out,
            curr==AttentionState::ATTENTIVE ? "ATTENTIVE" : "DISTRACTED",
            {30,60}, FONT_HERSHEY_SIMPLEX,2.5,
            curr==AttentionState::ATTENTIVE ? Scalar(0,255,0)
                                            : Scalar(0,0,255),
            4);

    // Draw EAR and yaw for debugging
    std::ostringstream dbg;
    dbg << std::fixed << std::setprecision(2)
        << "EAR:" << earVal << " yaw:" << yawVal;
    putText(out, dbg.str(), {30,110},
            FONT_HERSHEY_SIMPLEX,1.8, Scalar(255,255,0),2);

    return {curr,al};
}
