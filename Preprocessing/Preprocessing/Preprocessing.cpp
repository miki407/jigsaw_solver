#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <utility>
#include <opencv2/core.hpp>

//#include <cmath>

const float pi = 3.14159265358979323846f;
int puzzle_x = 7;
int puzzle_y = 10;
int number_of_pieces = puzzle_x * puzzle_y;

void status(int& steps, std::string task) {
    steps++;
    std::cout << steps << " \t \t" << task << "\t \t" << "\r";  

}
int main(){
    int step = 0;
    status(step, "Initilized code");
    for (int n = 1; n <= number_of_pieces; n++) {
        std::string imagePath = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(n) + ".jpg";
        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) {
            std::cout << "ERROR IMAGE" << imagePath << " NOT FOUND";
        }
        status(step, "Loaded image:" + imagePath);

        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);   //convert image to grayscale
        
        int threshold = 89;  //Treshold value
        cv::threshold(image, image, threshold, 255, cv::THRESH_BINARY); //treshold the image
        status(step, "Converted " + imagePath + " to binary bitmap"); 
        cv::Mat image_r;
        cv::resize(image, image_r, cv::Size((194 * 3), (259 * 3))); // resize
        //imshow( "Image", image_r);
        //cv::waitKey(0); // wait for user
        //status(step, "Displayed image:" + imagePath);
        cv::Canny(image, image, 80, 120, 3, false);
        //cv::Sobel(image, image, CV_8U, 1, 1, 3, 1, 0); // run derivative
        status(step, "Sobel function ran on:" + imagePath);
        //cv::resize(image, image_r, cv::Size((194 * 3), (259 * 3)));
        //imshow( "Image", image_r);
        //cv::waitKey(0);
        //status(step, "Displayed image:" + imagePath);
        cv::Mat mask;
        int blockSize = 30;
        int maxCorners = 4; 
        double qualityLevel = 0.01;
        double minDistance = 580.;
        std::vector< cv::Point2f > corners;
        cv::goodFeaturesToTrack(image, corners, maxCorners, qualityLevel, minDistance, mask, blockSize,3, true, 0.04 );
        //cv::cornerHarris(image, image, 20, 0, 0.2);
        status(step, "Ran cornerHarris() on image");
        cv::Mat img_with_points = image.clone();

        // draw each point as a red circle
        for (int i = 0; i < corners.size(); i++) {
            cv::circle(img_with_points, corners[i], 10, cv::Scalar(255, 255, 255), -1);
        }
        cv::Mat resized_img;
        cv::resize(img_with_points, resized_img, cv::Size((194 * 3), (259 * 3)));

        // display the image with points in a window
        cv::imshow("Points", resized_img);
        cv::waitKey(0); 
        status(step, "Displayed corners");
        
    }
    return 0;
}