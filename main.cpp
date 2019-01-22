#include <iostream>
#include <dlib/opencv.h>
#include <opencv2/face.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>

#include <fstream>


using namespace dlib;
using namespace std;

#define FACE_DOWNSAMPLE_RATIO 4
#define SKIP_FRAMES 2

cv::Mat get_camera_matrix(float focal_length, cv::Point2d center)
{
    cv::Mat camera_matrix = (cv::Mat_<double>(3,3) << focal_length, 0, center.x, 0 , focal_length, center.y, 0, 0, 1);
    return camera_matrix;
}


std::vector<cv::Point3d> get_3d_model_points()
{
    std::vector<cv::Point3d> modelPoints;

    modelPoints.push_back(cv::Point3d(0.0f, 0.0f, 0.0f)); //The first must be (0,0,0) while using POSIT
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

int main()
{
    //open cam
    cv::VideoCapture cap("/Users/sravanchennuri/Desktop/Research_Code/example1.mp4");

    cv::Mat im;


    cap >> im;
    cv::Mat im_small, im_display;
    cv::resize(im, im_small, cv::Size(), 1.0/4, 1.0/4);
    cv::resize(im, im_display, cv::Size(), 0.5, 0.5);
    cv::Size size = im.size();

    if (!cap.isOpened())
    {
        std::cout << "Unable to open" << std::endl;
        return EXIT_FAILURE;
    }

    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor predictor;
    dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> predictor;

    int count = 0;
    std::vector<dlib::rectangle> faces;

    std::vector<cv::Point3d> object_pts = get_3d_model_points();

    //2D ref points(image coordinates), referenced from detected facial feature
    std::vector<cv::Point2d> image_pts;

    //result
    cv::Mat rotation_vec;                           //3 x 1
    cv::Mat rotation_mat;                           //3 x 3 R
    cv::Mat translation_vec;                        //3 x 1 T
    cv::Mat pose_mat = cv::Mat(3, 4, CV_64FC1);     //3 x 4 R | T
    cv::Mat euler_angle = cv::Mat(3, 1, CV_64FC1);


    //temp buf for decomposeProjectionMatrix()
    cv::Mat out_intrinsics = cv::Mat(3, 3, CV_64FC1);
    cv::Mat out_rotation = cv::Mat(3, 3, CV_64FC1);
    cv::Mat out_translation = cv::Mat(3, 1, CV_64FC1);
    //text on screen
    std::ostringstream outtext;

    //main loop
    while (1)
    {
        // Grab a frame
        cv::Mat im;
        cap >> im;

        dlib::cv_image<dlib::bgr_pixel> cimg(im);

        // Detect faces
        std::vector<dlib::rectangle> faces = detector(cimg);
        std::vector<full_object_detection> shapes;

        for(unsigned long i=0; i < faces.size(); ++i)
        {

            //track features
            full_object_detection shape = predictor(cimg, faces[i]);
            shapes.push_back(shape);

            //draw features
            for (unsigned int i = 0; i < 68; ++i) {
                cv::circle(im, cv::Point(shape.part(i).x(), shape.part(i).y()), 2, cv::Scalar(255, 0, 255), -1);
            }

            image_pts = get_2d_image_points(shape);


            double focal_length = im.cols;
            cv::Mat cam_matrix = get_camera_matrix(focal_length, cv::Point2d(im.cols / 2, im.rows / 2));

            cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type);

            //calc pose
            cv::solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs, rotation_vec, translation_vec);


            std::vector<cv::Point3d> nose_end_point3D;
            std::vector<cv::Point2d> nose_end_point2D;
            nose_end_point3D.push_back(cv::Point3d(0.0, 0.0, 1000.0));

            cv::projectPoints(nose_end_point3D, rotation_vec, translation_vec, cam_matrix, dist_coeffs,
                              nose_end_point2D);
            cv::line(im, image_pts[0], nose_end_point2D[0], cv::Scalar(255, 0, 0), 2);


            //reproject
            //cv::projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix, dist_coeffs, reprojectdst);


            cv::Rodrigues(rotation_vec, rotation_mat);
            cv::hconcat(rotation_mat, translation_vec, pose_mat);
            cv::decomposeProjectionMatrix(pose_mat, out_intrinsics, out_rotation, out_translation, cv::noArray(),
                                          cv::noArray(), cv::noArray(), euler_angle);

            //show angle result
            outtext << "Z: YAW" << std::setprecision(3) << euler_angle.at<double>(2);
            cv::putText(im, outtext.str(), cv::Point(50, 80), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0));
            outtext.str("");

            outtext << "Y: PITCH" << std::setprecision(3) << euler_angle.at<double>(1);
            cv::putText(im, outtext.str(), cv::Point(50, 60), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0));
            outtext.str("");

            outtext << "X: Roll" << std::setprecision(3) << euler_angle.at<double>(0);
            cv::putText(im, outtext.str(), cv::Point(50, 40), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0));
            outtext.str("");


            //image_pts.clear();

        }


        //press esc to end
        cv::imshow("demo", im);
        unsigned char key = cv::waitKey(1);
        if (key == 27)
        {
            break;
        }
    }

    return 0;
}