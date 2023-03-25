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
        std::vector<int> vertices(white_pixels.size());
        for (int i = 0; i < white_pixels.size(); i++) {
            float smallest = INT_MAX;
            for (int m = i; m < white_pixels.size(); m++) {
                if (smallest > cv::norm(white_pixels[i] - white_pixels[m]) && i != m) {
                    smallest = cv::norm(white_pixels[i] - white_pixels[m]);
                    vertices[i] = m;
                }
            }
        }


        /*
                // Initialize the reordered points vector with the first point
                std::vector<cv::Point2f> reorderedPoints;
                reorderedPoints.push_back(white_pixels[0]);

                // Remove the first point from the unprocessed points vector
                std::vector<cv::Point2f> unprocessedPoints(white_pixels.begin() + 1, white_pixels.end());

                // Iteratively connect the closest unconnected point to the last processed point
                while (!unprocessedPoints.empty()) {
                    cv::Point2f lastPoint = reorderedPoints.back();
                    cv::Point2f closestPoint = unprocessedPoints[0];
                    float closestDistance = cv::norm(closestPoint - lastPoint);

                    // Find the closest unconnected point to the last processed point
                    for (int i = 1; i < unprocessedPoints.size(); i++) {
                        cv::Point2f point = unprocessedPoints[i];
                        float distance = cv::norm(point - lastPoint);
                        if (distance < closestDistance) {
                            closestPoint = point;
                            closestDistance = distance;
                        }
                    }

                    // Add the closest point to the reordered points vector and remove it from the unprocessed points vector
                    reorderedPoints.push_back(closestPoint);
                    unprocessedPoints.erase(std::remove(unprocessedPoints.begin(), unprocessedPoints.end(), closestPoint), unprocessedPoints.end());
                } */





        status(step, "Calculated verticies table");






        for (int i = 0; i < reformed_corners.size(); i++) {//desplay corner points
            cv::circle(img_with_points2, reformed_corners[i], 10, cv::Scalar(255, 255, 255), -1);
            //  std::cout << reformed_corners[i].y << "\n" << reformed_corners[i].x << "\n";
        }
        cv::Mat img(512, 512, CV_8U);
        for (int i = 0; i < white_pixels.size() - 1; i++) {
            cv::line(img, white_pixels[i], white_pixels[vertices[i]], cv::Scalar(255, 0, 0), 1);
            // cv::line(img, reorderedPoints[i], reorderedPoints[i+1], cv::Scalar(255, 0, 0), 1);
        }
        cv::imshow("bullshit", img);
        cv::waitKey(0);

        int ref[5] = { 0, 1, 2, 3, 0 };
        double angle_min[4];
        double angle_max[4];
        for (int i = 0; i < 4; i++) {
            int p1 = points[ref[i]];
            int p2 = points[ref[i+1]];
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

        for (int k = 0; k < white_pixels.size(); k++) {

       //     cout << white_pixels[k].x << std::endl;

        }
        std::cout << "-------------------------------------------------------------------------------------------------" << std::endl;

        for (int k = 0; k < white_pixels.size(); k++) {

         //   cout << white_pixels[k].y << std::endl;

        }
        std::ofstream output_file;
        output_file.open("C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\1.txt");
        for (auto const& point : white_pixels) {
            output_file << point.x << " " << point.y << std::endl;
        }
       
        std::vector<cv::Point2f> side0;
        std::vector<cv::Point2f> side1;
        std::vector<cv::Point2f> side2;
        std::vector<cv::Point2f> side3;


        for (int i = 0; i < 4; i++) {
    
            for (int j=0; j < white_pixels.size(); j++) {
                double angle1 = atan2(white_pixels[j].y, white_pixels[j].x); 
              
               
                if ((angle1 > angle_min[i] && angle1 < angle_max[i] && i!=1) || (i==1 && (angle1 < angle_min[i] || angle1 > angle_max[i]))) {
                  
                    if (i == 0) {
                        side0.push_back(white_pixels[j]);
                    }
                    if (i == 1) {
                        side1.push_back(white_pixels[j]);
                    }
                    if (i == 2) {
                        side2.push_back(white_pixels[j]);
                    }
                    if (i == 3) {
                        side3.push_back(white_pixels[j]);
                    } 
                   // std::cout << white_pixels[j] << "\n";
                 
                
                }
            }
        }
        std::ofstream output_file0;
        output_file0.open("C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\1_0.txt");
        for (auto const& point : side0) {
            output_file0 << point.x << " " << point.y << std::endl;
        }
        std::ofstream output_file1;
        output_file1.open("C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\1_1.txt");
        for (auto const& point : side1) {
            output_file1 << point.x << " " << point.y << std::endl;
        }
        std::ofstream output_file2;
        output_file2.open("C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\1_2.txt");
        for (auto const& point : side2) {
            output_file2 << point.x << " " << point.y << std::endl;
        }
        std::ofstream output_file3;
        output_file3.open("C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\1_3.txt");
        for (auto const& point : side3) {
            output_file3 << point.x << " " << point.y << std::endl;
        }
  
            status(step, "Order by angle");
   

        int treshold=50;
        int treshold2=100;
      
        std::vector<cv::Point2f> sides_20;
        std::vector<cv::Point2f> sides_21;
        std::vector<cv::Point2f> sides_22;
        std::vector<cv::Point2f> sides_23;
        for (int j = 0; j < 4; j++) {
            int p = 0;
            int k = 1; 
           // std::cout << side0.size() << std::endl;
          
            std::vector<cv::Point2f> side;
            if (j == 0) {
                for (int k = 0; k < side0.size(); k++) {
                    side.push_back(side0[k]);
                }
               // std::cout << 0 << "\n";
            }     
            if (j == 1) {
                for (int k = 0; k < side1.size(); k++) {
                    side.push_back(side1[k]);
                  //  std::cout << side1[k] << "\n";
                }
              //  std::cout << 1 << "\n";
            }     
            if (j == 2) {
                for (int k = 0; k < side2.size(); k++) {
                    side.push_back(side2[k]);
                   // std::cout << side2[k] << "\n";
                }
              //  std::cout << 2 << "\n";
            }     
            if (j == 3) {
                for (int k = 0; k < side3.size(); k++) {
                    side.push_back(side3[k]);
                   // std::cout << side3[k] << "\n";
                }
              //  std::cout << 3 << "\n";
            }
            cv::Point2f  ref;
            cv::Point2f  ref1;
           
            while (k != 0) {
                k = 0;
       
                ref = ref1;
                if (p == 0) {
                    ref = reformed_corners[j];
                }
              
                double smallest = INT_MAX;

                for (int i = 0; i < side.size(); i++) {
                  //  std::cout << side[i].x << side[i].y << std::endl;
                    double d1 = cv::norm(side[i] - ref);
                  
                 
                    if (d1 < treshold && i > 0) {
                       
                        if (j == 0 ) {
                            sides_20.push_back(side[i]);
                            side.erase(side.begin() + i);
                            side0.erase(side0.begin() + i);
                           // std::cout << side0.size() << std::endl;
                        }
                        if (j == 1 ) {
                            sides_21.push_back(side[i]);
                            side.erase(side.begin() + i);
                            side1.erase(side1.begin() + i);
                        }
                        if (j == 2 ) {
                            sides_22.push_back(side[i]);
                            side.erase(side.begin() + i);
                            side2.erase(side2.begin() + i);
                        }
                        if (j == 3 ) {
                            sides_23.push_back(side[i]);
                            side.erase(side.begin() + i);
                            side3.erase(side3.begin() + i);
                        }
                      i=0;
                    }
                    if ( d1 < smallest && d1 < treshold2 && i > 0) { 

                        ref1 = side[i];
                        smallest = d1;
                        k = 1;
                        p = 1; 
                    }
                   
               
                }

            }
        }
        std::ofstream output_file01;
        output_file01.open("C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\1_01.txt");
        for (auto const& point : sides_20) {
            output_file01 << point.x << " " << point.y << std::endl;
        }
        std::ofstream output_file11;
        output_file11.open("C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\1_11.txt");
        for (auto const& point : sides_21) {
            output_file11 << point.x << " " << point.y << std::endl;
        }
        std::ofstream output_file21;
        output_file21.open("C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\1_21.txt");
        for (auto const& point : sides_22) {
            output_file21 << point.x << " " << point.y << std::endl;
        }
        std::ofstream output_file31;
        output_file31.open("C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\1_31.txt");
        for (auto const& point : sides_23) {
            output_file31 << point.x << " " << point.y << std::endl;
        }
      
        status(step, "Isolate clusters");
         int size;
        int comb[4];
      
        for (int j = 0; j < 4; j++) {
           
            std::vector<cv::Point2f> side;
            if (j == 0) {
                for (int k = 0; k < side0.size(); k++) {
                    side.push_back(side0[k]);
                }
               
                size = sides_20.size();
             //   for (int k = 0; k < sides_20.size(); k++) {
             //       side_2.push_back(sides_20[k]);
                //}
              
            }
            if (j == 1) {
            
                for (int k = 0; k < side1.size(); k++) {
                    side.push_back(side1[k]);
                }

               size = sides_21.size();

             //  for (int k = 0; k < sides_21.size(); k++) {
             //      side_2.push_back(sides_21[k]);
             //  }

            }
            if (j == 2) {
           
                for (int k = 0; k < side2.size(); k++) {
                    side.push_back(side2[k]);
                }

                size = sides_22.size();

             //   for (int k = 0; k < sides_22.size(); k++) {
             //       side_2.push_back(sides_22[k]);
             //   }

            }
            if (j == 3) {
          
                for (int k = 0; k < side3.size(); k++) {
                    side.push_back(side3[k]);
                }

               size = sides_23.size();
 
             //  for (int k = 0; k < sides_23.size(); k++) {
              //     side_2.push_back(sides_23[k]);
              // }

            }
            std::cout << side.size() << std::endl;
            if (side.size() !=1) {
                
                for (int i = 0; i < 4; i++) {
                    std::vector<cv::Point2f> side_2;
                    double smallest = INT_MAX;  
                    if (j == 0) {
                        for (int k = 0; k < sides_20.size(); k++) {
                            side_2.push_back(sides_20[k]);
                        }
                    }
                    if (j == 1) {
                        for (int k = 0; k < sides_21.size(); k++) {
                            side_2.push_back(sides_21[k]);
                        }
                    }
                    if (j == 2) {
                        for (int k = 0; k < sides_22.size(); k++) {
                            side_2.push_back(sides_22[k]);
                        }
                    }
                    if (j == 3) {
                        for (int k = 0; k < sides_23.size(); k++) {
                            side_2.push_back(sides_23[k]);
                        }
                    }
                    if (j != i) {
                        for (int k = 0; k < size; k++) {

                            double d1 = cv::norm(side_2[k] - side[1]);
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
            if (j == 0) {
                if (comb[j] == 0) {
                    for (int k = 1; k < side0.size(); k++) {
                        sides_20.push_back(side0[k]);
                    }
                }
                if (comb[j] == 1) {
                    for (int k = 1; k < side0.size(); k++) {
                        sides_21.push_back(side0[k]);
                    }
                }
                if (comb[j] == 2) {
                    for (int k = 1; k < side0.size(); k++) {
                        sides_22.push_back(side0[k]);
                    }
                } 
                if (comb[j] == 3) {
                    for (int k = 1; k < side0.size(); k++) {
                        sides_23.push_back(side0[k]);
                    }
                }
                if (comb[j] == 5)   std::cout << "-----"  << std::endl;
            }
            if (j == 1) {
                if (comb[j] == 0) {
                    for (int k = 1; k < side1.size(); k++) {
                        sides_20.push_back(side1[k]);
                    }
                }
                if (comb[j] == 1) {
                    for (int k = 1; k < side1.size(); k++) {
                        sides_21.push_back(side1[k]);
                    }
                }
                if (comb[j] == 2) {
                    for (int k = 1; k < side1.size(); k++) {
                        sides_22.push_back(side1[k]);
                    }
                }
                if (comb[j] == 3) {
                    for (int k = 1; k < side1.size(); k++) {
                        sides_23.push_back(side1[k]);
                    }
                }
                if (comb[j] == 5)   std::cout << "-----" << std::endl;
            }
            if (j == 2) {
                if (comb[j] == 0) {
                    for (int k = 1; k < side0.size(); k++) {
                        sides_20.push_back(side0[k]);
                    }
                }
                if (comb[j] == 1) {
                    for (int k = 1; k < side2.size(); k++) {
                        sides_21.push_back(side2[k]);
                    }
                }
                if (comb[j] == 2) {
                    for (int k = 1; k < side2.size(); k++) {
                        sides_22.push_back(side2[k]);
                    }
                }
                if (comb[j] == 3) {
                    for (int k = 1; k < side2.size(); k++) {
                        sides_23.push_back(side2[k]);
                    }
                }
                if (comb[j] == 5)   std::cout << "-----" << std::endl;
            }
            if (j == 3) {
                if (comb[j] == 0) {
                    for (int k = 1; k < side3.size(); k++) {
                        sides_20.push_back(side3[k]);
                    }
                }
                if (comb[j] == 1) {
                    for (int k = 1; k < side3.size(); k++) {
                        sides_21.push_back(side3[k]);
                    }
                }
                if (comb[j] == 2) {
                    for (int k = 1; k < side3.size(); k++) {
                        sides_22.push_back(side3[k]);
                    }
                }
                if (comb[j] == 3) {
                    for (int k = 1; k < side3.size(); k++) {
                        sides_23.push_back(side3[k]);
                    }
                }
                if (comb[j] == 5)   std::cout << "-----" << std::endl;
            }
        }
        std::ofstream output_file02;
        output_file02.open("C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\1_01.txt");
        for (auto const& point : sides_20) {
            output_file02 << point.x << " " << point.y << std::endl;
        }
        std::ofstream output_file12;
        output_file12.open("C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\1_11.txt");
        for (auto const& point : sides_21) {
            output_file12 << point.x << " " << point.y << std::endl;
        }
        std::ofstream output_file22;
        output_file22.open("C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\1_21.txt");
        for (auto const& point : sides_22) {
            output_file22 << point.x << " " << point.y << std::endl;
        }
        std::ofstream output_file32;
        output_file32.open("C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\1_31.txt");
        for (auto const& point : sides_23) {
            output_file32 << point.x << " " << point.y << std::endl;
        }

         status(step, "Match clusters");
         status(step, "displayed vector sides_2");

        cv::Mat resized_img;
        cv::resize(img_with_points2, resized_img, cv::Size((194 * 3), (259 * 3)));
        cv::imshow("Points", resized_img);
        cv::waitKey(0);
    }
    return 0;
}