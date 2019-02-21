#include <iostream>
#include <dlib/opencv.h>
#include <opencv2/face.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <fstream>
#include <unordered_map>

using namespace dlib;
using namespace std;

#define FACE_DOWNSAMPLE_RATIO 4
#define SKIP_FRAMES 2

std::ostringstream outtext;

void getRandomColors(std::vector<cv::Scalar> &colors, int numColors)
{
    cv::RNG rng(0);
    for(int i=0; i < numColors; i++)
        colors.push_back(cv::Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255)));
}

static cv::Rect dlibRectangleToOpenCV( dlib::rectangle r)
{
    return cv::Rect(cv::Point2i(r.left(), r.top()), cv::Point2i(r.right() + 1, r.bottom() + 1));
}

static dlib::rectangle openCVRectToDlib(cv::Rect r)
{
    return dlib::rectangle((long)r.tl().x, (long)r.tl().y, (long)r.br().x - 1, (long)r.br().y - 1);
}

std::vector<string> trackerTypes = {"BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN", "MOSSE", "CSRT"};

cv::Ptr<cv::Tracker> createTrackerByName(string trackerType)
{
    cv::Ptr<cv::Tracker> tracker;
    if (trackerType ==  trackerTypes[0])
        tracker = cv::TrackerBoosting::create();
    else if (trackerType == trackerTypes[1])
        tracker = cv::TrackerMIL::create();
    else if (trackerType == trackerTypes[2])
        tracker = cv::TrackerKCF::create();
    else if (trackerType == trackerTypes[3])
        tracker = cv::TrackerTLD::create();
    else if (trackerType == trackerTypes[4])
        tracker = cv::TrackerMedianFlow::create();
    else if (trackerType == trackerTypes[5])
        tracker = cv::TrackerGOTURN::create();
    else if (trackerType == trackerTypes[6])
        tracker = cv::TrackerMOSSE::create();
    else if (trackerType == trackerTypes[7])
        tracker = cv::TrackerCSRT::create();
    else {
        cout << "Incorrect tracker name" << endl;
        cout << "Available trackers are: " << endl;
        for (std::vector<string>::iterator it = trackerTypes.begin() ; it != trackerTypes.end(); ++it)
            std::cout << " " << *it << endl;
    }
    return tracker;
}

cv::Mat get_camera_matrix(float focal_length, cv::Point2d center)
{
    cv::Mat camera_matrix = (cv::Mat_<double>(3,3) << focal_length, 0, center.x, 0 , focal_length, center.y, 0, 0, 1);
    return camera_matrix;
}

std::vector<cv::Point3d> get_3d_model_points()
{
    std::vector<cv::Point3d> modelPoints;

    modelPoints.push_back(cv::Point3d(0.0f, 0.0f, 0.0f));
    modelPoints.push_back(cv::Point3d(0.0f, -330.0f, -65.0f));
    modelPoints.push_back(cv::Point3d(-225.0f, 170.0f, -135.0f));
    modelPoints.push_back(cv::Point3d(225.0f, 170.0f, -135.0f));
    modelPoints.push_back(cv::Point3d(-150.0f, -150.0f, -125.0f));
    modelPoints.push_back(cv::Point3d(150.0f, -150.0f, -125.0f));

    return modelPoints;
}
std::vector<cv::Point2d> get_2d_image_points(full_object_detection &d)
{
    std::vector<cv::Point2d> image_points;
    image_points.push_back( cv::Point2d( d.part(30).x(), d.part(30).y() ) );    // Nose tip
    image_points.push_back( cv::Point2d( d.part(8).x(), d.part(8).y() ) );      // Chin
    image_points.push_back( cv::Point2d( d.part(36).x(), d.part(36).y() ) );    // Left eye left corner
    image_points.push_back( cv::Point2d( d.part(45).x(), d.part(45).y() ) );    // Right eye right corner
    image_points.push_back( cv::Point2d( d.part(48).x(), d.part(48).y() ) );    // Left Mouth corner
    image_points.push_back( cv::Point2d( d.part(54).x(), d.part(54).y() ) );    // Right mouth corner

    return image_points;
}

struct membuf : std::streambuf {
    membuf(char const* base, size_t size) {
        char* p(const_cast<char*>(base));
        this->setg(p, p, p + size);
    }
};
struct imemstream : virtual membuf, std::istream {
    imemstream(char const* base, size_t size)
            : membuf(base, size)
            , std::istream(static_cast<std::streambuf*>(this)) {
    }
};

void deserialize_shape_predictor(dlib::shape_predictor &predictor){

    const char* file_name = "shape_predictor_68_face_landmarks.dat";
    //const char* file_name = "shape_predictor_5_face_landmarks.dat";
    ifstream fs(file_name, ios::binary | ios::ate);
    streamsize size = fs.tellg();
    fs.seekg(0, ios::beg);
    std::vector<char> buffer(size);
    if (fs.read(buffer.data(), size))
    {
        cout << "Successfully read " << size << " bytes from " << file_name << " into buffer" << endl;
        imemstream stream(&buffer.front(), size);
        dlib::deserialize(predictor,stream);
        cout << "Deserialized shape_predictor" << endl;
    }
}

bool isOverlap ( dlib::rectangle &face, cv::Rect tracker_rect ){

    cv::Rect face_rect = dlibRectangleToOpenCV(face);
    bool intersects;
    intersects = ((face_rect & tracker_rect).area() > 0);
    return intersects;
}

bool is_Tracked( dlib::rectangle &face, cv::Ptr<cv::MultiTracker> &multiTracker ){

    for(unsigned j=0; j<multiTracker->getObjects().size(); j++){
        if (isOverlap( face, multiTracker->getObjects()[j])) return true;
    }
    return false;
}

class Face {

public:
    cv::Rect cvRect;
    dlib::rectangle dlibRect;

    double x;
    double y;

    double t;
    const double theta1 = (CV_PI)/8;

    double roll;
    double pitch;
    double yaw;

    double rollRadians;
    double pitchRadians;
    double yawRadians;

    cv::Mat rotation_vec;                //3 x 3 R
    cv::Mat translation_vec;             //3 x 1 T

    int face_id;

    cv::Mat rotation_mat;

    //temp buf for decomposeProjectionMatrix()
    cv::Mat out_intrinsics = cv::Mat(3, 3, CV_64FC1);
    cv::Mat out_rotation = cv::Mat(3, 3, CV_64FC1);
    cv::Mat out_translation = cv::Mat(3, 1, CV_64FC1);

    cv::Mat pose_mat = cv::Mat(3, 4, CV_64FC1);     //3 x 4 R | T
    cv::Mat euler_angle = cv::Mat(3, 1, CV_64FC1);
    std::vector<cv::Point3d> object_pts;

    Face(dlib::rectangle r) {
        this->dlibRect = r;
        this->cvRect = dlibRectangleToOpenCV(r);
        this->object_pts = get_3d_model_points();
    }


    std::vector<cv::Point2d> calculateFeaturePoints(cv::Mat frame, dlib::shape_predictor predictor) {
        dlib::cv_image<dlib::bgr_pixel> cimg(frame);
        full_object_detection shape = predictor(cimg, this->dlibRect);

        std::vector<cv::Point2d> img_points = get_2d_image_points(shape);
        this->x = img_points[0].x;
        this->y = img_points[0].y;

        //draw features
        for (unsigned int i = 0; i < 68; ++i) {
            cv::circle(frame, cv::Point(shape.part(i).x(), shape.part(i).y()), 2, cv::Scalar(255, 0, 255), -1);
        }

        return img_points;
    }

    void calculatePose(cv::Mat im, dlib::shape_predictor predictor) {

        double focal_length = im.cols;
        std::vector<cv::Point2d> image_pts = this->calculateFeaturePoints(im, predictor);

        cv::Mat cam_matrix = get_camera_matrix(focal_length, cv::Point2d(im.cols / 2, im.rows / 2));
        cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type);

        //cv::solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs, rotation_vec, translation_vec); //calc pose
        cv::solvePnPRansac(object_pts, image_pts, cam_matrix, dist_coeffs, rotation_vec, translation_vec, false, 1000, 8.0, 0.99, cv::noArray() , cv::SOLVEPNP_UPNP); //calc pose

        std::vector<cv::Point3d> nose_end_point3D;
        std::vector<cv::Point2d> nose_end_point2D;
        nose_end_point3D.push_back(cv::Point3d(0.0, 0.0, 1000.0));

        cv::projectPoints(nose_end_point3D, rotation_vec, translation_vec, cam_matrix, dist_coeffs, nose_end_point2D);
        cv::line(im, image_pts[0], nose_end_point2D[0], cv::Scalar(255, 0, 0), 2);

        cv::Rodrigues(rotation_vec, rotation_mat);
        cv::hconcat(rotation_mat, translation_vec, pose_mat);
        cv::decomposeProjectionMatrix(pose_mat, out_intrinsics, out_rotation, out_translation, cv::noArray(), cv::noArray(), cv::noArray(), euler_angle);
        
        this->roll = euler_angle.at<double>(0);
        this->pitch = euler_angle.at<double>(1);
        this->yaw = euler_angle.at<double>(2);

        this->rollRadians = roll * (CV_PI)/180;
        this->pitchRadians = pitch * (CV_PI)/180;
        this->yawRadians = yaw * (CV_PI)/180;
    }

    void updateTracker(std::unordered_map<int, double> map, cv::Ptr<cv::MultiTracker> multiTracker) {
        for(int j = 0; j < multiTracker->getObjects().size(); j++  ){
            if ( isOverlap(this->dlibRect, multiTracker->getObjects()[j]) ){
                map[j] = this->pitchRadians;
                this->face_id = j;
            }
        }
    }

    void printFace(){
        cout<<"Face_ID: "<< this->face_id <<endl;
        cout<<"X: " << this->x << ", " << "Y: " << this->y <<endl;
        cout<<"Roll: " << this->roll << endl;
        cout<<"Pitch: " << this->pitch << endl;
        cout<<"Yaw: " << this->yaw << endl;
    }


};


int main(){

    int frameCount = 0;
    cv::VideoCapture cap("/Users/sravanchennuri/Desktop/Research_Code/tom.mp4");

    cv::Mat first_frame;


    if (!cap.isOpened())
    {
        std::cout << "Unable to open" << std::endl;
        return EXIT_FAILURE;
    }

    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor predictor;
    deserialize_shape_predictor(predictor);

    std::unordered_map <int,  double> map;
    std::vector<std::unordered_map <int, double>> map_list;

    cap>>first_frame;
    cv::resize(first_frame, first_frame, cv::Size(), 1.0/2, 1.0/2);

    dlib::cv_image<dlib::bgr_pixel> fimg(first_frame);
    std::vector<dlib::rectangle> first_frame_faces = detector(fimg);

    cv::Ptr<cv::MultiTracker> multiTracker =  cv::MultiTracker::create();
    for (int i = 0; i < first_frame_faces.size(); ++i) {
        multiTracker->add (createTrackerByName("CSRT"), first_frame, cv::Rect2d( dlibRectangleToOpenCV(first_frame_faces[i])) );
    }

    double t1;
    double t2;
    double thetaConst = (CV_PI)/8;

    while (frameCount<83)
    {
        std::vector<Face> faces;
        cv::Mat frame;
        cap >> frame;
        cv::resize(frame, frame, cv::Size(), 1.0/2, 1.0/2);

        dlib::cv_image<dlib::bgr_pixel> cimg(frame);
        std::vector<dlib::rectangle> dlibfaces = detector(cimg); // Detect faces

        for(dlib::rectangle f : dlibfaces) {
            faces.push_back(Face(f));
        }
        cout<<"FrameNumber: "<<frameCount<<endl;

        for(unsigned long i=0; i < faces.size(); ++i){

            if ( !is_Tracked(faces[i].dlibRect, multiTracker) ){
                multiTracker->add (createTrackerByName("CSRT"), frame, cv::Rect2d( faces[i].cvRect ));
            }
            //track features
            cv::Rect rect_in_cv = faces[i].cvRect;
            cv::rectangle(frame, rect_in_cv, cv::Scalar(0,0,255), 2);

            faces[i].calculatePose(frame, predictor); //getPose

            outtext << "Number of detected faces " << std::setprecision(3) << faces.size();
            cv::putText(frame, outtext.str(), cv::Point(50, 20), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0));
            outtext.str("");

            for(int j = 0; j < multiTracker->getObjects().size(); j++  ){
                if ( isOverlap(faces[i].dlibRect, multiTracker->getObjects()[j]) ){
                    map[j] = faces[i].pitchRadians;
                }
            }
            faces[i].updateTracker(map, multiTracker);
            faces[i].printFace();

            if(faces[i].face_id == 0) {
                //t1 = - (faces[i].x) / cos(faces[i].pitchRadians - thetaConst);
                t1 = - (faces[i].y) / sin(faces[i].pitchRadians + thetaConst);
            }
            else if(faces[i].face_id == 1){
                //t2 = - (faces[i].x) / cos(faces[i].pitchRadians - thetaConst);
                t2 =  (faces[i].y) / sin(faces[i].pitchRadians + thetaConst);
            }

        }
        cout<<"t1: "<< t1 << endl;
        cout<<"t2: "<< t2 << endl;

        cout << endl;cout << endl;

        multiTracker->update(frame); //update tracker

        if(faces.size()>1){
            map_list.push_back(map);
        }

        double cosine;
        for( auto k: map_list ){
            //cout <<"map_list: "<< k.at(0) << "-> " << k.at(1) << endl;
            //cout << k.first << " -> " << k.second << endl;
            cosine =  cos( k.at(0) - k.at(1) );
            if( faces.size() > 1 && cosine < 0 ){
                outtext << "Interacting " << std::setprecision(3);
                cv::putText(frame, outtext.str(), cv::Point(50, 40), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0));
                outtext.str("");
            }
        }

        frameCount++;
        outtext << "FrameCount:  " << frameCount << std::setprecision(3);
        cv::putText(frame, outtext.str(), cv::Point(50, 60), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0));
        outtext.str("");

        cv::imshow("demo", frame);
        unsigned char key = cv::waitKey(1);
        if (key == 27)
        {
            break;
        }

    }
    return 0;
}

