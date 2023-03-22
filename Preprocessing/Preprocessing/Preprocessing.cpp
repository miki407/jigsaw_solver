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
    std::cout << steps << " \t \t" << task  << "\n";  

}

int  x_sum;
int  y_sum;
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
        //cv::resize(image, image_r, cv::Size((194 * 3), (259 * 3))); // resize
        //imshow( "Image", image_r);
        //cv::waitKey(0); // wait for user
        //status(step, "Displayed image:" + imagePath);
        //cv::Canny(image, image, 80, 120, 3, false);
        //cv::Sobel(image, image, CV_8U, 1, 1, 3, 1, 0); // run derivative
        cv::Mat grad_x, grad_y;
        cv::Mat abs_grad_x, abs_grad_y;
        int scale = 1;
        int delta = 0;
        int ddepth = CV_16S;
        Sobel(image, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_REPLICATE);
        Sobel(image, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_REPLICATE);
        convertScaleAbs(grad_x, abs_grad_x);
        convertScaleAbs(grad_y, abs_grad_y);
        addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, image);
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
        cv::goodFeaturesToTrack(image, corners, maxCorners, qualityLevel, minDistance, mask, blockSize, 3, true, 0.04);
        //cv::cornerHarris(image, image, 20, 0, 0.2);
        status(step, "Ran cornerHarris() on image");
        cv::Mat img_with_points = image.clone();

        //draw each point as a red circle
        for (int i = 0; i < corners.size(); i++) {
            //cv::circle(img_with_points, corners[i], 10, cv::Scalar(255, 255, 255), -1);
            x_sum = x_sum + corners[i].x;
            y_sum = y_sum + corners[i].y;
        }

        //cv::Mat resized_img;
        //cv::resize(img_with_points, resized_img, cv::Size((194 * 3), (259 * 3)));
        int avrage_x = x_sum / 4;
        int avrage_y = y_sum / 4;
        x_sum = 0;
        y_sum = 0;
        cv::Point2f middle(avrage_x, avrage_y);
        //cv::imshow("Points", resized_img);
        //cv::waitKey(0);
        //display the image with points in a window

        status(step, "Displayed corners");
        for (int o = 0; o < corners.size(); o++) {
            corners[o].x = corners[o].x - middle.x;
            corners[o].y = corners[o].y - middle.y;
        }

        int smallest1[2] = {INT_MAX,0};
        int smallest2[2] = {INT_MAX,0};
        for (int i = 0; i < corners.size(); i++) {//find the smallest .x points to get 2 ajacant points
            if (smallest1[0] > corners[i].x) {
                smallest1[0] = corners[i].x;
                smallest1[1] = i;
            }
        }

        for (int i = 0; i < corners.size(); i++) { //find the smallest .x points to get 2 ajacant points
            if (smallest2[0] > corners[i].x && i != smallest1[1]) {
                smallest2[0] = corners[i].x;
                smallest2[1] = i; 
            }
        }
       
        int referance_x = corners[smallest1[1]].x - corners[smallest2[1]].x;//find the angle to align the side between points smallest1[1] and smallest2[1] from corners
        int referance_y = corners[smallest1[1]].y - corners[smallest2[1]].y;
        double angle = atan2(referance_y, referance_x);
        angle = CV_PI / 2 - angle;
        for (cv::Point2f& point : corners) {//rotate the vector by angle
            double x = point.x;
            double y = point.y;
            point.x = x * cos(angle) - y * sin(angle);
            point.y = x * sin(angle) + y * cos(angle);
        }

        cv::Mat img_with_points2 = image.clone();
        // draw each point as a red circle
       

        std::vector<cv::Point2f> white_pixels;
        for (int y = 0; y < image.rows; y++) {//create a vector for the outline
            for (int x = 0; x < image.cols; x++) {//map the points
                if (image.at<uchar>(y, x) == 255) {
                    white_pixels.push_back(cv::Point(x, y));
                }
            }
        }

        for (int o = 0; o < white_pixels.size(); o++) {//offset the points
            white_pixels[o].x = white_pixels[o].x - middle.x;
            white_pixels[o].y = white_pixels[o].y - middle.y;
        }

        for (cv::Point2f& point : white_pixels) {//rotate the vector by the angle
            double x = point.x;
            double y = point.y;
            point.x = x * cos(angle) - y * sin(angle);
            point.y = x * sin(angle) + y * cos(angle);
        }

        for (int i = 0; i < white_pixels.size(); i++) {
            cv::Point2f zero(0, 0);
            cv::circle(img_with_points2, zero, 1, cv::Scalar(255, 255, 255), -1);
            cv::circle(img_with_points2, white_pixels[i], 1, cv::Scalar(255, 255, 255), -1);
        }
        //for (int i = 0; i < corners.size(); i++) {//desplay corner points
            
          //  std::cout << corners[i].y << "\n" << corners[i].x << "\n";
        //}
    
        std::vector<cv::Point2f> reformed_corners(4);//sorted corners
        for (int i = 0; i < corners.size(); i++) {//sort the corner points
            if (corners[i].x > 0 && corners[i].y > 0) {
                reformed_corners[0].y = corners[i].y;
                reformed_corners[0].x = corners[i].x;
            }
            if (corners[i].x < 0 && corners[i].y > 0) {
                reformed_corners[1].y = corners[i].y;
                reformed_corners[1].x = corners[i].x;
            }
            if (corners[i].x < 0 && corners[i].y < 0) {
                reformed_corners[2].y = corners[i].y;
                reformed_corners[2].x = corners[i].x;
            }
            if (corners[i].x > 0 && corners[i].y < 0) {
                reformed_corners[3].y = corners[i].y;
                reformed_corners[3].x = corners[i].x;
            }
        }
        int points[4];
        for (int i = 0; i < reformed_corners.size(); i++) {//move points for the clossest outline points
            int closest = 0;
            float closestDist = cv::norm(reformed_corners[i] - white_pixels[0]);
            for (int j = 1; j < white_pixels.size(); j++) {
                float dist = cv::norm(reformed_corners[i] - white_pixels[j]);
                if (dist < closestDist) {
                    closestDist = dist;
                    closest = j;
                }
            }
            reformed_corners[i] = white_pixels[closest];
            points[i] = closest;
        }
        for (int i = 0; i < reformed_corners.size(); i++) {//desplay corner points
            cv::Point2f zero(0, 0);
            cv::circle(img_with_points2, zero, 10, cv::Scalar(255, 255, 255), -1);
            cv::circle(img_with_points2, reformed_corners[i], 10, cv::Scalar(255, 255, 255), -1);
          //  std::cout << reformed_corners[i].y << "\n" << reformed_corners[i].x << "\n";
        } 
        for (int i = 0; i < white_pixels.size(); i++) {
            for (int j = 0; j < white_pixels.size(); j++) {

            }
        }
        cv::Mat resized_img;
        cv::resize(img_with_points2, resized_img, cv::Size((194 * 3), (259 * 3)));
        cv::imshow("Points", resized_img);
        cv::waitKey(0);
    }
    return 0;
}