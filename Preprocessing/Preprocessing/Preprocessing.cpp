#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <utility>
#include <opencv2/core.hpp>
#include <chrono>
#include <cstdint>
#include < iomanip >
//#include <cmath>
using std::cout;
using std::setw;
const float pi = 3.14159265358979323846f;
int puzzle_x = 7;
int puzzle_y = 10;
int number_of_pieces = puzzle_x * puzzle_y;
std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

void status(int& steps, std::string task) {
    steps++;
    std::cout << steps << " \t \t" << task << "\t" << (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()) - ms).count() << "ms" << "\n";  
    ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
}

int  x_sum;
int  y_sum;
int main() {
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

        int smallest1[2] = { INT_MAX,0 };
        int smallest2[2] = { INT_MAX,0 };
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


        std::vector<cv::Point2f> white_pixels;//create a vector for the outline
        for (int y = 0; y < image.rows; y++) {//map the points
            for (int x = 0; x < image.cols; x++) {
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
            cv::circle(img_with_points2, zero, 10, cv::Scalar(255, 255, 255), -1);
            cv::circle(img_with_points2, white_pixels[i], 1, cv::Scalar(255, 255, 255), -1);
        }

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
            cv::circle(img_with_points2, reformed_corners[i], 10, cv::Scalar(255, 255, 255), -1);
        }

        int ref[5] = { 0, 1, 2, 3, 0 };
        double angle_min[4];
        double angle_max[4];
        for (int i = 0; i < 4; i++) {
            int p1 = points[ref[i]];
            int p2 = points[ref[i + 1]];
            double angle1 = atan2(white_pixels[p1].y, white_pixels[p1].x);
            double angle2 = atan2(white_pixels[p2].y, white_pixels[p2].x);
            if (angle1 < angle2) {
                angle_min[i] = angle1;
                angle_max[i] = angle2;
            }
            else {
                angle_min[i] = angle2;
                angle_max[i] = angle1;

            }

        }

        status(step, "Bounding angles calculated");


        std::ofstream output_file;
        output_file.open("C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\1.txt");
        for (auto const& point : white_pixels) {
            output_file << point.x << " " << point.y << std::endl;
        }
        status(step, "Wrote vectors to file");
        std::vector<std::vector<cv::Point2f>> sides(4);

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < white_pixels.size(); j++) {
                double angle1 = atan2(white_pixels[j].y, white_pixels[j].x);

                if ((angle1 > angle_min[i] && angle1 < angle_max[i] && i != 1) || (i == 1 && (angle1 < angle_min[i] || angle1 > angle_max[i]))) {

                    sides[i].push_back(white_pixels[j]);

                }
            }

        }
        status(step, "Order by angle");
        for (int k = 0; k < 4; k++) {
            std::ofstream output_file0;
            output_file0.open("C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\1_" + std::to_string(k) + ".txt");
            for (auto const& point : sides[k]) {
                output_file0 << point.x << " " << point.y << std::endl;
            }

        }
        status(step, "Wrote vectors to file");


        int treshold = 50;
        int treshold2 = 100;
        std::vector <std::vector<cv::Point2f>> sides_2(4);

        for (int j = 0; j < 4; j++) {
            int p = 0;
            int k = 1;

            cv::Point2f  ref;
            cv::Point2f  ref1;

            while (k != 0) {
                k = 0;

                ref = ref1;
                if (p == 0) {
                    ref = reformed_corners[j];
                }

                double smallest = INT_MAX;

                for (int i = 0; i < sides[j].size(); i++) {

                    double d1 = cv::norm(sides[j][i] - ref);


                    if (d1 < treshold && i > 0) {
                        sides_2[j].push_back(sides[j][i]);
                        sides[j].erase(sides[j].begin() + i);
                        i = 0;
                    }
                    if (d1 < smallest && d1 < treshold2 && i > 0) {

                        ref1 = sides[j][i];
                        smallest = d1;
                        k = 1;
                        p = 1;
                    }


                }

            }
        }
        status(step, "Isolate clusters");

        for (int k = 0; k < 4; k++) {
            std::ofstream output_file0;
            output_file0.open("C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\1_" + std::to_string(k) + "1.txt");

            for (auto const& point : sides_2[k]) {
                output_file0 << point.x << " " << point.y << std::endl;
            }
        }

        status(step, "Wrote vectors to file");
        int size;
        int comb[4];

        for (int j = 0; j < 4; j++) {


            if (sides[j].size() > 1) {


                for (int i = 0; i < 4; i++) {
                    double smallest = INT_MAX;
                    if (j != i) {
                        for (int k = 1; k < sides_2[i].size(); k++) {


                            double d1 = cv::norm(sides_2[i][k] - sides[j][1]);
                            if (d1 < smallest) {
                                smallest = d1;
                                comb[j] = i;
                            }


                        }
                    }
                }
            }
            else {
                comb[j] = 5;
            }
        }
        for (int j = 0; j < 4; j++) {
            std::cout << comb[j] << std::endl;
        }
        for (int j = 0; j < 4; j++) {
            for (int i = 0; i < 4; i++) {
                if (comb[j] == i) {
                    for (int k = 1; k < sides[j].size(); k++) {
                        sides_2[i].push_back(sides[j][k]);
                    }
                }
            }
        }
        /*  int treshold3 = 20;
           for (int j = 0; j < 1; j++) {
               int p = 1;
               int counter1 = 0;

               cv::Point2f ref3;
               cv::Point2f ref4;
               int n{};
               ref3 = reformed_corners[j];
               double sumx1{};
               double sumy1{};

               while (p != 0) {
                   p = 0;
                   cv::Point2f avrage(sumx1 / counter1, sumy1 / counter1);
                   std::cout << avrage.x << "----" << avrage.y << "\n";
                   sides_2[j].push_back(avrage);
                   double smallest = INT_MAX;
                   for (int i = 0; i < (sides_2[j].size() -n); i++) {
                       double d = cv::norm(sides_2[j][i] - ref3);
                       if (d < treshold3) {
                           sumx1 = sumx1 + sides_2[j][i].x;
                           sumy1 = sumy1 + sides_2[j][i].y;
                           sides_2[j].erase(sides_2[j].begin() + i);
                           i = 0;
                           counter1++;
                       }
                       std::cout << i << "\n";
                       if (d < smallest && i > 0 && d > (treshold3 + 10)) {

                           ref4 = sides_2[j][i];
                           smallest = d;


                           p = 1;
                       }
                   }
                   n++;
                   ref3 = ref4;
                   std::cout << ref4 << "\n";
               }
           }*/
        status(step, "Matched clusters");

        for (int k = 0; k < 4; k++) {
            std::ofstream output_file02;
            output_file02.open("C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\1_" + std::to_string(k) + "1.txt");
            for (auto const& point : sides_2[k]) {
                output_file02 << point.x << " " << point.y << std::endl;
            }
        }

        status(step, "wrote vectors to file");

        std::vector <std::vector<float>> normal_diviation(4);
        for (int k = 0; k < 4; k++) {
            std::ofstream output_file32;
            output_file32.open("C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\1_normalDiviation_" + std::to_string(k) + ".txt");
            for (int i = 0; i < sides_2[k].size(); i++) {
                float ac = cv::norm(reformed_corners[k] - sides_2[k][i]);
                float bc = cv::norm(reformed_corners[ref[k + 1]] - sides_2[k][i]);
                float ab = cv::norm(reformed_corners[k] - reformed_corners[ref[k + 1]]);

                float S = (ab + ac + bc) / 2;
                float A = sqrt(S * (S - ab) * (S - ac) * (S - bc));
               normal_diviation[k].push_back(2 * A / ab);
                /*if (k != 3) {
                    float angle = atan2(reformed_corners[k].y - sides_2[k][i].y, reformed_corners[k].x - sides_2[k][i].x) - atan2(reformed_corners[k].y - reformed_corners[k + 1].y, reformed_corners[k].x - reformed_corners[k + 1].x);
                }
                else {
                    float angle = atan2(reformed_corners[k].y - sides_2[k][i].y, reformed_corners[k].x - sides_2[k][i].x) - atan2(reformed_corners[k].y - reformed_corners[0].y, reformed_corners[k].x - reformed_corners[0].x);
                }
                normal_diviation[k].push_back(point_distance * sin(angle));
                */
                output_file32 << 2 * A / ab << std::endl;
            }
            output_file32.close();
        }
        status(step, "Claculated normal diviation");
        /*       for (int k = 0; k < 4; k++) {
                   std::ofstream output_file32;
                   output_file32.open("C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\1_normalDiviation_" + std::to_string(k) + ".txt");
                   for (auto const& point : normal_diviation[k]) {
                       output_file32 << point << std::endl;
                   }
                   output_file32.close();
               }
               status(step, "wrote normal diviation to file");
       */



        cv::Mat resized_img;
        cv::resize(img_with_points2, resized_img, cv::Size((194 * 3), (259 * 3)));
        cv::imshow("Points", resized_img);
        cv::waitKey(0);
        status(step, "displayed vector sides_2");
    }
    return 0;
}
