#include <iostream>
#include <fstream>
#include <Windows.h>
#include <GL\GL.h>
#include <GL\GLU.h>


#include "include\GL\freeglut.h"
#include "opencv2\opencv.hpp"

using namespace cv;
using namespace std;

#pragma comment(lib, "opencv_world310.lib")

#define	FRAME_WIDTH		640
#define	FRAME_HEIGHT	480

///< unit: cm
#define	MARKER_HEIGHT	20 //마커의 크기 20CM로 지정
#define MARKER_WIDTH	20


#define	zNear	1.0
#define	zFar	1000000
int HAND_FLAG;
Mat img, gRGB, gYCrCb, grayImg2, grayAddImg, skinMask, hand, aMask, bMask, cMask;
vector <Mat> YCrCb;

Rect roiRect = Rect(390, 230, 250, 250);
Mat gMarkerImg[2];	///< 마커 이미지
Mat gSceneImg;	///< 카메라로 캡쳐한 이미지
Mat	gOpenGLImg;	///< OpenGL로 렌더링할 이미지
VideoCapture gVideoCapture;	///< 카메라 캡쳐 객체

cv::Ptr<cv::ORB> detector[2];	///< ORB 특징점 추출기
cv::Ptr<cv::DescriptorMatcher> matcher[2];	///< ORB 특징정보 매칭 객체

//< 마커 및 카메라로 캡쳐한 이미지의 ORB 특징정보(keypoints)
std::vector<cv::KeyPoint> gvMarkerKeypoints[2], gvSceneKeypoints[2];

///< 마커 및 카메라로 캡쳐한 이미지의 ORB 특징정보(descriprtors)
cv::Mat gMarkerDescriptors[2], gSceneDescriptors[2];


cv::Mat E[2];	///< 마커 좌표계에서 카메라 좌표계로의 변환 행렬
cv::Mat K;	///< 카메라 내부 파라메터

vector <double> mydata,mydata2;
int DAY = 0, TIME = 0;
int gawicount = 20, moogcount = 0, gawitoggle = 1, batoggle = 1;
double moveX = 0.0;

void init(void)
{
	///< 마커에서 카메라로의 변환 행렬을 초기화 한다.
	E[0] = cv::Mat::eye(4, 4, CV_64FC1);
	E[1] = cv::Mat::eye(4, 4, CV_64FC1);
	K = cv::Mat::eye(3, 3, CV_64FC1);

	///< 카메라 내부 파라메터 초기화
	K.at<double>(0, 0) = 864.258645;
	K.at<double>(1, 1) = 867.822578;
	K.at<double>(0, 2) = 377.648867;
	K.at<double>(1, 2) = 316.777493;
	

	///< 마커 이미지를 읽는다.
	gMarkerImg[0] = cv::imread("allstar.png", 0);
	gMarkerImg[1] = cv::imread("angels.jpg", 0);
	
	
	///< 카메라를 초기화
	if (!gMarkerImg[0].data || !gMarkerImg[1].data || !gVideoCapture.open(0)) {
		std::cerr << "초기화를 수행할 수 없습니다." << std::endl;
		exit(-1);
	}

	///< 특징정보 추출기와 매칭 객체 초기화
	for (int lp = 0; lp < 2; lp++) {
		detector[lp] = cv::ORB::create();
		matcher[lp] = cv::DescriptorMatcher::create("BruteForce-Hamming");
		///< 마커 영상의 특징정보 추출
		detector[lp]->detect(gMarkerImg[lp], gvMarkerKeypoints[lp]);
		detector[lp] ->compute(gMarkerImg[lp], gvMarkerKeypoints[lp], gMarkerDescriptors[lp]);
		///< 마커 영상의 실제 크기 측정
		for (int i = 0; i < (int)gvMarkerKeypoints[lp].size(); i++) {
			gvMarkerKeypoints[lp][i].pt.x /= gMarkerImg[lp].cols;
			gvMarkerKeypoints[lp][i].pt.y /= gMarkerImg[lp].rows;

			gvMarkerKeypoints[lp][i].pt.x -= 0.5;
			gvMarkerKeypoints[lp][i].pt.y -= 0.5;

			gvMarkerKeypoints[lp][i].pt.x *= MARKER_WIDTH;
			gvMarkerKeypoints[lp][i].pt.y *= MARKER_HEIGHT;
		}
	}
	///< OpenGL 초기화
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
}

///< 카메라 내부 파마메터에서 OpenGL 내부 파라메터 변환
void convertFromCaemraToOpenGLProjection(double* mGL)
{
	cv::Mat P = cv::Mat::zeros(4, 4, CV_64FC1);

	P.at<double>(0, 0) = 2 * K.at<double>(0, 0) / FRAME_WIDTH;
	P.at<double>(1, 0) = 0;
	P.at<double>(2, 0) = 0;
	P.at<double>(3, 0) = 0;

	P.at<double>(0, 1) = 0;
	P.at<double>(1, 1) = 2 * K.at<double>(1, 1) / FRAME_HEIGHT;
	P.at<double>(2, 1) = 0;
	P.at<double>(3, 1) = 0;

	P.at<double>(0, 2) = 1 - 2 * K.at<double>(0, 2) / FRAME_WIDTH;
	P.at<double>(1, 2) = -1 + (2 * K.at<double>(1, 2) + 2) / FRAME_HEIGHT;
	P.at<double>(2, 2) = (zNear + zFar) / (zNear - zFar);
	P.at<double>(3, 2) = -1;

	P.at<double>(0, 3) = 0;
	P.at<double>(1, 3) = 0;
	P.at<double>(2, 3) = 2 * zNear*zFar / (zNear - zFar);
	P.at<double>(3, 3) = 0;

	for (int ix = 0; ix < 4; ix++)
	{
		for (int iy = 0; iy < 4; iy++)
		{
			mGL[ix * 4 + iy] = P.at<double>(iy, ix);
		}
	}
}

///< 호모그래피로부터 마커에서 카메라로의 변환 행렬 추출
bool	calculatePoseFromH(const cv::Mat& H, cv::Mat& R, cv::Mat& T)
{
	cv::Mat InvK = K.inv();
	cv::Mat InvH = InvK * H;
	cv::Mat h1 = H.col(0);
	cv::Mat h2 = H.col(1);
	cv::Mat h3 = H.col(2);

	double dbNormV1 = cv::norm(InvH.col(0));

	if (dbNormV1 != 0) {
		InvK /= dbNormV1;

		cv::Mat r1 = InvK * h1;
		cv::Mat r2 = InvK * h2;
		cv::Mat r3 = r1.cross(r2);

		T = InvK * h3;

		cv::Mat R1 = cv::Mat::zeros(3, 3, CV_64FC1);

		r1.copyTo(R1.rowRange(cv::Range::all()).col(0));
		r2.copyTo(R1.rowRange(cv::Range::all()).col(1));
		r3.copyTo(R1.rowRange(cv::Range::all()).col(2));

		cv::SVD svd(R1);

		R = svd.u * svd.vt;

		return true;
	}
	else
		return false;
}

int getWhiteCount(Mat &grayImg2) {
	int whitecount = 0; //white pixel의 개수를 0으로 초기화 합니다.
	for (int i = 0; i<grayImg2.cols; i++) {
		for (int j = 0; j<grayImg2.rows; j++) {
			int value = grayImg2.at<uchar>(i, j);
			if (value == 255) {//해당 pixel이 흰색이면 white pixel의 개수를 1추가합니다. 
				whitecount++;
			}
		}
	}
	//white pixel의 개수를 반환합니다.
	return whitecount;
}
///< 카메라로부터 영상을 읽고, 특징정보 추출한 후, 마커 영상과의 매칭 및 호모그래피를 추정
///< 추정한 호모그래피로부터 마커에서 카메라로의 변환 행렬 추정
void hand_interact() {
	/*묵찌빠 테스트*/
	
	cv::rectangle(gSceneImg, roiRect, Scalar(0, 0, 255), 2);

	Mat roi = gSceneImg(roiRect);

	//cv::cvtColor(roi, gRGB, CV_BGRA2BGR);
	cv::cvtColor(roi, gYCrCb, CV_BGR2YCrCb);

	cv::split(gYCrCb, YCrCb);

	inRange(YCrCb[2], Scalar(77), Scalar(150), YCrCb[2]);
	inRange(YCrCb[1], Scalar(133), Scalar(173), YCrCb[1]);

	cv::bitwise_and(YCrCb[1], YCrCb[2], grayImg2);

	cv::cvtColor(grayImg2, skinMask, CV_GRAY2BGR);

	cv::bitwise_and(skinMask, roi, hand);

	//범위에서 pixel이 흰색인 개수를 가지고 옵니다.
	int whitecount = getWhiteCount(grayImg2);
	//printf("%d\n", whitecount);
	
	//흰색 pixel의 수가 10000개~15000개면 바위로 판별합니다.
	if (whitecount > 15000 && whitecount <=25000) {
		HAND_FLAG = 1;
 	}//흰색 pixel의 수가 15000개~20000개면 가위로 판별합니다.
	else if (whitecount > 25000 && whitecount <32000) {
		HAND_FLAG = 2;
	}//흰색 pixel의 수가 20000개~30000개면 보자기로 판별합니다.
	else if (whitecount >= 32000 && whitecount <45000) {
		HAND_FLAG = 3;
	}else {
		HAND_FLAG = 4;
	}
	
	
	imshow("roi", roi);
	imshow("gray", grayImg2);
	imshow("hand", hand);
	
	//------------------묵찌빠 테스트
}

void processVideoCapture(void)
{
	cv::Mat grayImg;

	///< 카메라로부터 영상획득
	gVideoCapture >> gSceneImg;
	//hand_interact();
	
	///< 특징정보 추출을 위하여 흑백영상으로 변환
	cv::cvtColor(gSceneImg, grayImg, CV_BGR2GRAY);
	vector < vector< vector<DMatch> >> matches(2);
	vector <vector<DMatch>> good_matches(2);
	
	///< 카메라로부터 획득한 영상의 특징정보 추출
	for (int lp = 0; lp < 2; lp++) {

		detector[lp]->detect(grayImg, gvSceneKeypoints[lp]);
		detector[lp]->compute(grayImg, gvSceneKeypoints[lp], gSceneDescriptors[lp]);
	
		///< 마커 특징정보와 매칭 수행
		if (gvMarkerKeypoints[lp].size() >= 4 && gvSceneKeypoints[lp].size() >= 4)
			matcher[lp]->knnMatch(gMarkerDescriptors[lp], gSceneDescriptors[lp], matches[lp], 2);
				
		for (int i = 0; i < (int)matches[lp].size(); i++) {
			if (matches[lp][i][0].distance < 0.9 * matches[lp][i][1].distance) {
				good_matches[lp].push_back(matches[lp][i][0]);
			}
		}

		///< 마커 특징정보와 충분한 대응점이 있는 경우에....
		if (good_matches[lp].size() > 100) {
			std::vector<cv::Point2f> vMarkerPts;
			std::vector<cv::Point2f> vScenePts;
			///< 호모그래피 추정
			for (int i = 0; i < (int)good_matches[lp].size(); i++) {
				vMarkerPts.push_back(gvMarkerKeypoints[lp][matches[lp][i][0].queryIdx].pt);
				vScenePts.push_back(gvSceneKeypoints[lp][matches[lp][i][0].trainIdx].pt);
			}

			cv::Mat H = cv::findHomography(vMarkerPts, vScenePts, CV_RANSAC);

			std::vector<cv::Point2f> obj_corners(4);
			obj_corners[0] = cv::Point(-MARKER_WIDTH / 2, -MARKER_HEIGHT / 2);
			obj_corners[1] = cv::Point(MARKER_WIDTH / 2, -MARKER_HEIGHT / 2);
			obj_corners[2] = cv::Point(MARKER_WIDTH / 2, MARKER_HEIGHT / 2);
			obj_corners[3] = cv::Point(-MARKER_WIDTH / 2, MARKER_HEIGHT / 2);

			std::vector<cv::Point2f> scene_corners(4);

			cv::perspectiveTransform(obj_corners, scene_corners, H);

			cv::line(gSceneImg, scene_corners[0], scene_corners[1], cv::Scalar(0, 255, 0), 2);
			cv::line(gSceneImg, scene_corners[1], scene_corners[2], cv::Scalar(0, 255, 0), 2);
			cv::line(gSceneImg, scene_corners[2], scene_corners[3], cv::Scalar(0, 255, 0), 2);
			cv::line(gSceneImg, scene_corners[3], scene_corners[0], cv::Scalar(0, 255, 0), 2);
		
			cv::Mat R, T;

			///< 호모그래피로부터 마커에서 카메라 좌표로의 변환 행렬 추정
			if (calculatePoseFromH(H, R, T)) {
				R.copyTo(E[lp].rowRange(0, 3).colRange(0, 3));
				T.copyTo(E[lp].rowRange(0, 3).col(3));

				static double changeCoordArray[4][4] = { { 1, 0, 0, 0 },{ 0, -1, 0, 0 },{ 0, 0, -1, 0 },{ 0, 0, 0, 1 } };
				static cv::Mat changeCoord(4, 4, CV_64FC1, changeCoordArray);

				E[lp] = changeCoord * E[lp];
			}
		}
		if (gSceneImg.data) {
			cv::flip(gSceneImg, gOpenGLImg, 0);
		}
	}
		
	glutPostRedisplay();
}
void cal(vector<double> a) {
	int n = mydata.size();
	double sum = 0, mean, var;
	for (int i = 0; i<n; i = i + 1)
		sum += a[i];
	mean = sum / n;
	sum = 0;
	for (int i = 0; i<n; i = i + 1)
		sum += (a[i] - mean)*(a[i] - mean);
	var = sum / (n - 1);
	printf("평균 = %8.3fcm\n", mean);
	printf("분산 = %8.3fcm\n", var);
	printf("표준편차 = %8.3fcm\n", sqrt(var));
}

void display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	///< 배경 영상 렌더링
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glDrawPixels(gOpenGLImg.cols, gOpenGLImg.rows, GL_BGR_EXT, GL_UNSIGNED_BYTE, (void *)gOpenGLImg.data);

	
	//< 마커로부터 카메라로의 변환행렬을 통해 마커좌표계로의 변환
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	cv::Mat Ea = E[0].t();
	cv::Mat Eb = E[1].t();
	double etoeDistance = sqrt(pow(E[0].at<double>(0, 3) - E[1].at<double>(0, 3), 2) 
						     + pow(E[0].at<double>(1, 3) - E[1].at<double>(1, 3), 2)); 
								//postion의 x,y좌표를 통해 길이 측정
	double normDistance = norm(E[0], E[1], NORM_L2); //OpenCV Norm 함수를 이용하여 길이 측정
	
	mydata.push_back(etoeDistance);
	mydata2.push_back(normDistance);
	cout << "Dinstance(My)" << etoeDistance << "CM" << endl;
	cal(mydata);
	cout << "Distance(Norm)" << normDistance << "CM" << endl;
	cal(mydata2);
	//cout << "E2" << (E[0].) << endl;
	glMultMatrixd((double *)Ea.data);
	
	glLineWidth(2.0f);
	glBegin(GL_LINES);
	glColor3f(1.0f, 0.0f, 0.0f); glVertex3d(0.0, 0.0, 0.0); glVertex3d(10.0, 0.0, 0.0);
	glColor3f(0.0f, 1.0f, 0.0f); glVertex3d(0.0, 0.0, 0.0); glVertex3d(0.0, 10.0, 0.0);
	glColor3f(0.0f, 0.0f, 1.0f); glVertex3d(0.0, 0.0, 0.0); glVertex3d(0.0, 0.0, 10.0);
	glEnd();

		
	bool changeWire = false; 

	if (HAND_FLAG == 0) {
		glRotated(-90.0, 1.0, 0.0, 0.0);
		glLineWidth(1.0f);
	}
	else {
		if (HAND_FLAG == 1) {//'묵'일때
				DAY += 5 % 360; //공전
				TIME += 10 % 360; //자전
				moveX = MARKER_WIDTH / 2; //x축 이동					
		}
		else if (HAND_FLAG == 2) { //"찌'일때
			gawicount++; //가위 카운트에 따라 색 변경
			if (gawicount > 20) { //일정 시간을 두고 색 변경
				gawicount = 0;
				gawitoggle++;
			}
		}
		else if (HAND_FLAG == 3) {//"빠"일때
			changeWire = true; 
		}
		glRotated(-(GLdouble)DAY, 0.0, 0.0, 1.0);
		glTranslated((GLdouble)moveX, 0.0, 0.0);
		glRotated((GLdouble)TIME, 0.0, 0.0, 1.0);
		glLineWidth(1.0f);
	}
	
	
	gawitoggle %= 3; //0,1,2 - 0,1,2 반복
	
	if (gawitoggle == 0) {
		glColor3f(1.0f, 0.0, 0.0);
	}
	else if (gawitoggle == 1) {
		glColor3f(0.0, 1.0f, 0.0);
	}
	else if (gawitoggle == 2) {
		glColor3f(0.0, 0.0, 1.0f);
	}
	

	if (changeWire) { //빠일때
		glutWireTeapot(2);
	}
	else { //빠가 아닐때
		glutSolidTeapot(2);
	}
	
	
	glPopMatrix();
	glLoadIdentity();

	

	//new
	//glMatrixMode(GL_MODELVIEW);
	//glLoadIdentity();
		
	///< 마커로부터 카메라로의 변환행렬을 통해 마커좌표계로의 변환
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glMultMatrixd((double *)Eb.data);
	
	glLineWidth(2.0f);
	glBegin(GL_LINES);
	glColor3f(1.0f, 0.0f, 0.0f); glVertex3d(0.0, 0.0, 0.0); glVertex3d(10.0, 0.0, 0.0);
	glColor3f(0.0f, 1.0f, 0.0f); glVertex3d(0.0, 0.0, 0.0); glVertex3d(0.0, 10.0, 0.0);
	glColor3f(0.0f, 0.0f, 1.0f); glVertex3d(0.0, 0.0, 0.0); glVertex3d(0.0, 0.0, 10.0);
	glEnd();

	/*
	glRotated(-90.0, 1.0, 0.0, 0.0);
	glLineWidth(1.0f);
	glColor3f(1.0f, 1.0f, 0.0);
	*/
	
	
	if (HAND_FLAG == 0) {
		glRotated(-90.0, 1.0, 0.0, 0.0);
		glLineWidth(1.0f);

	}
	else {
		glRotated((GLdouble)DAY, 0.0, 0.0, 1.0);
		glTranslated((GLdouble)moveX, 0.0, 0.0);
		glRotated(-(GLdouble)TIME, 0.0, 0.0, 1.0);
		glLineWidth(1.0f);
	}
		
	if (gawitoggle == 0) {
		glColor3f(0.0, 1.0f, 0.0);
	}
	else if (gawitoggle == 1) {
		glColor3f(0.0, 0.0, 1.0f);
	}
	else if (gawitoggle == 2) {
		glColor3f(1.0f, 0.0, 0.0);
	}


	if (changeWire) {
		glutWireCube(2);
	}
	else {
		glutSolidCube(2);
	}
	
	//< 마커 좌표계의 중심에서 객체 렌더링
	
	
	//여기삽입
	glPushMatrix();
	
	//glPopMatrix();
	//glLoadIdentity();


	glutSwapBuffers();
}

void idle(void)
{
	processVideoCapture();

	glutPostRedisplay();
}

void reshape(int w, int h)
{
	double P[16] = { 0 };

	convertFromCaemraToOpenGLProjection(P);

	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadMatrixd(P);
	glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char** argv)
{
	
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(640, 480);
	glutCreateWindow("AR Homework");

	init();
		
	glEndList();
	glutDisplayFunc(display);
	glutIdleFunc(idle);
	glutReshapeFunc(reshape);
	glutMainLoop();
	
	return 0;
}


