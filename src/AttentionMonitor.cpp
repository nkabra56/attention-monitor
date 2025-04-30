#include "AttentionMonitor.hpp"
#include <iomanip>
#include <sstream>
#include <algorithm>

using namespace cv;

AttentionMonitor::AttentionMonitor(const std::string& ssdProto,
                                   const std::string& ssdModel,
                                   const std::string& lbfModel)
{
    faceNet  = dnn::readNetFromCaffe(ssdProto, ssdModel);
    facemark = cv::face::FacemarkLBF::create();
    facemark->loadModel(lbfModel);

    stateStart   = Clock::now();
    warning1Time = stateStart;

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
    for(int eyeStart : {36,42}) {
        std::vector<Point2f> pts(lm.begin()+eyeStart,
                                 lm.begin()+eyeStart+6);
        Rect r = boundingRect(pts);
        r.x = max(r.x-2,0); r.y = max(r.y-2,0);
        r.width  = min(r.width+4, gray.cols - r.x);
        r.height = min(r.height+4, gray.rows - r.y);
        Mat roi = gray(r);

        Mat thr;
        threshold(roi, thr, 50, 255, THRESH_BINARY_INV);
        std::vector<std::vector<Point>> C;
        findContours(thr, C, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        if(C.empty()) continue;

        int best=0; double A=0;
        for(int i=0;i<(int)C.size();i++){
            double a = contourArea(C[i]);
            if(a>A){ A=a; best=i; }
        }
        Moments m = moments(C[best]);
        if(m.m00<=0) continue;
        Point2f c(m.m10/m.m00, m.m01/m.m00);
        double nx = (c.x - roi.cols/2.0)/(roi.cols/2.0);
        if(fabs(nx)>gazeThresh) return true;
    }
    return false;
}

double AttentionMonitor::headYaw(const std::vector<Point2f>&) { return 0.0; }

std::pair<AttentionState,MonitorAlert>
AttentionMonitor::update(const Mat& frame, Mat& out)
{
    frame.copyTo(out);

    // 1) Preprocess
    Mat gray; cvtColor(frame, gray, COLOR_BGR2GRAY);
    static auto clahe = createCLAHE(2.0, Size(8,8));
    Mat g2; clahe->apply(gray, g2); gray = g2;

    // 2) Detect vs Track
    frameCounter++;
    std::vector<Point2f> lmPts;
    bool haveLm=false;
    if(frameCounter%detectInterval==1 || prevPts.empty()){
        Rect face;
        if(findFace(frame, face)){
            std::vector<std::vector<Point2f>> all;
            if(findLandmarks(gray, face, all) && !all.empty()){
                lmPts   = all[0];
                prevPts = lmPts;
                gray.copyTo(prevGray);
                rectangle(out, face, Scalar(0,255,0),2);
                haveLm=true;
            }
        }
    } else {
        std::vector<uchar> st; std::vector<float> er;
        std::vector<Point2f> cur;
        calcOpticalFlowPyrLK(prevGray, gray, prevPts, cur, st, er);
        for(int i=0;i<(int)cur.size();i++)
            if(st[i]) lmPts.push_back(cur[i]);
        if((int)lmPts.size()>=6){
            haveLm=true;
            prevPts=lmPts;
            gray.copyTo(prevGray);
        }
    }

    // 3) Draw eye‐contours
    if(haveLm){
        std::vector<std::vector<Point>> eyes;
        { std::vector<Point> e; for(int i=36;i<=41;++i) e.emplace_back(lmPts[i]); eyes.push_back(e);}
        { std::vector<Point> e; for(int i=42;i<=47;++i) e.emplace_back(lmPts[i]); eyes.push_back(e);}
        polylines(out, eyes, true, Scalar(255,0,0),1);
    }

    // 4) Compute metrics
    bool gazeAway=false;
    double earVal=0, rawYaw=0, rawPitch=0, rawMar=0;
    if(haveLm){
        double L=eyeAspectRatio(lmPts,36), R=eyeAspectRatio(lmPts,42);
        earVal=(L+R)*0.5;
        gazeAway = isGazeDistracted(gray, lmPts);
        if(!camInit){
            double fx=frame.cols, fy=fx, cx=frame.cols*0.5, cy=frame.rows*0.5;
            cameraMatrix=(Mat_<double>(3,3)<<fx,0,cx,0,fy,cy,0,0,1);
            distCoeffs=Mat::zeros(5,1,CV_64F);
            camInit=true;
        }
        std::vector<Point3f> model={
            {0,0,0},{0,-330,-65},
            {-225,170,-135},{225,170,-135},
            {-150,-150,-125},{150,-150,-125}
        };
        std::vector<int> idx={30,8,36,45,48,54};
        std::vector<Point2f> img;
        for(int j:idx) img.push_back(lmPts[j]);
        Mat r,t,rot,inn;
        if(solvePnPRansac(model,img,cameraMatrix,distCoeffs,
                          r,t,false,100,8.0,0.99,inn,SOLVEPNP_ITERATIVE)){
            Rodrigues(r,rot);
            rawYaw   = fabs(atan2(rot.at<double>(2,0),
                                  rot.at<double>(0,0))*180.0/CV_PI);
            rawPitch= atan2(-rot.at<double>(2,1),
                           std::sqrt(rot.at<double>(2,0)*rot.at<double>(2,0)
                                   +rot.at<double>(2,2)*rot.at<double>(2,2)))
                        *180.0/CV_PI;
        }
        rawMar = ( norm(lmPts[62]-lmPts[66])+norm(lmPts[63]-lmPts[65]) )
                 /(2.0*norm(lmPts[60]-lmPts[64]));
    } else {
        noLmFrames++;
    }

    // 5) Smooth & KF
    yawHistory.push_back(rawYaw);
    if(yawHistory.size()>yawHistSize) yawHistory.pop_front();
    std::vector<double> t1(yawHistory.begin(),yawHistory.end());
    std::sort(t1.begin(),t1.end());
    double medYaw=t1[t1.size()/2];
    yawSmoothed = yawSmoothAlpha*medYaw + (1-yawSmoothAlpha)*yawSmoothed;
    kfMeas.at<float>(0)=float(yawSmoothed);
    kfYaw.correct(kfMeas);
    kfState=kfYaw.predict(); double yawKF=kfState.at<float>(0);

    pitchHistory.push_back(rawPitch);
    if(pitchHistory.size()>pitchHistSize) pitchHistory.pop_front();
    t1.assign(pitchHistory.begin(),pitchHistory.end());
    std::sort(t1.begin(),t1.end());
    double medPit=t1[t1.size()/2];
    pitchSmoothed = pitchSmoothAlpha*medPit + (1-pitchSmoothAlpha)*pitchSmoothed;
    kfMeas.at<float>(0)=float(pitchSmoothed);
    kfPitch.correct(kfMeas);
    kfState=kfPitch.predict(); double pitchKF=kfState.at<float>(0);

    kfMeas.at<float>(0)=float(rawMar);
    kfMar.correct(kfMeas);
    kfState=kfMar.predict(); double marKF=kfState.at<float>(0);

    // 6) Thresholds
    static int eCnt=0,yCnt=0,pCnt=0,mCnt=0;
    bool blinkOff   = (++eCnt>=earFramesThresh   && earVal < earThresh);
    bool poseOffYaw = (++yCnt>=yawFramesThresh   && yawKF   > yawThreshEnter);
    bool poseOffPit = (++pCnt>=pitchFramesThresh && pitchKF > pitchThreshEnter);
    bool mouthOff   = (++mCnt>=marFramesThresh   && marKF   > marThresh);

    AttentionState curr = (blinkOff||poseOffYaw||poseOffPit||gazeAway||mouthOff)
                          ? AttentionState::DISTRACTED
                          : AttentionState::ATTENTIVE;

    // 7) FSM & warnings
    MonitorAlert al{warningLevel,false};
    auto now=Clock::now();
    double el=Seconds(now-stateStart).count();
    if(curr==lastState){
        if(curr==AttentionState::DISTRACTED){
            if(warningLevel==0&&el>=5.0){warningLevel=1;al={1,true};warning1Time=now;}
            else if(warningLevel==1&&el>=20.0){warningLevel=2;al={2,true};}
        } else {
            if(warningLevel==1&&el>=16.0) warningLevel=0;
            if(warningLevel==2&&el>=25.0) warningLevel=0;
        }
    } else {
        stateStart=now; lastState=curr;
    }
    al.level=warningLevel;
    if(warningLevel==1&&Seconds(now-warning1Time).count()<warning1LockSec)
        curr=AttentionState::DISTRACTED;

    // 8) Draw smaller fonts
    putText(out,
        curr==AttentionState::ATTENTIVE?"ATTENTIVE":"DISTRACTED",
        Point(30,40),
        FONT_HERSHEY_SIMPLEX,
        1.2,
        curr==AttentionState::ATTENTIVE?Scalar(0,255,0):Scalar(0,0,255),
        2);

    std::ostringstream dbg;
    dbg<<std::fixed<<std::setprecision(2)
       <<"EAR:"<<earVal
       <<" yaw:"<<yawKF
       <<" pitch:"<<pitchKF
       <<" gaze:"<<(gazeAway?"OFF":"OK")
       <<" yawn:"<<marKF;
    putText(out,
        dbg.str(),
        Point(30,80),
        FONT_HERSHEY_SIMPLEX,
        1.0,
        Scalar(255,255,0),
        1);

    return {curr,al};
}
