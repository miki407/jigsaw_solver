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
const float pi = 3.14159265358979323846 ;
int puzzle_x = 7;
int puzzle_y = 10;
const int number_of_pieces = puzzle_x * puzzle_y;
std::chrono::milliseconds ms = std::chrono::duration_cast < std::chrono::milliseconds > (std::chrono::system_clock::now().time_since_epoch());

void status(int & steps, std::string task) {
  steps++;
  std::cout << steps << " \t \t" << task << "\t" << (std::chrono::duration_cast < std::chrono::milliseconds > (std::chrono::system_clock::now().time_since_epoch()) - ms).count() << "ms" << "\n";
  ms = std::chrono::duration_cast < std::chrono::milliseconds > (std::chrono::system_clock::now().time_since_epoch());
}

int orientation(cv::Point2f p1, cv::Point2f p2, cv::Point2f p3) {
  // See 10th slides from following link for derivation
  // of the formula
  int val = (p2.y - p1.y) * (p3.x - p2.x) -
    (p2.x - p1.x) * (p3.y - p2.y);

  if (val == 0)
    return 0; // collinear

  return (val > 0) ? 1 : 2; // clock or counterclock wise
}
bool onSegment(cv::Point2f p, cv::Point2f q, cv::Point2f r) {
  if (q.x <= std::max(p.x, r.x) && q.x >= std::min(p.x, r.x) &&
    q.y <= std::max(p.y, r.y) && q.y >= std::min(p.y, r.y))
    return true;

  return false;
}
bool doIntersect(cv::Point2f p1, cv::Point2f q1, cv::Point2f p2, cv::Point2f q2) {
  // Find the four orientations needed for general and
  // special cases
  int o1 = orientation(p1, q1, p2);
  int o2 = orientation(p1, q1, q2);
  int o3 = orientation(p2, q2, p1);
  int o4 = orientation(p2, q2, q1);

  // General case
  if (o1 != o2 && o3 != o4)
    return true;

  // Special Cases
  // p1, q1 and p2 are collinear and p2 lies on segment p1q1
  if (o1 == 0 && onSegment(p1, p2, q1)) return true;

  // p1, q1 and q2 are collinear and q2 lies on segment p1q1
  if (o2 == 0 && onSegment(p1, q2, q1)) return true;

  // p2, q2 and p1 are collinear and p1 lies on segment p2q2
  if (o3 == 0 && onSegment(p2, p1, q2)) return true;

  // p2, q2 and q1 are collinear and q1 lies on segment p2q2
  if (o4 == 0 && onSegment(p2, q1, q2)) return true;

  return false; // Doesn't fall in any of the above cases
}
double compareSide(std::vector<cv::Point2f> v1,std::vector<cv::Point2f> v2, float treshold, float samples) {
    float s_v1 = (float)v1.size() / samples;
    float s_v2 = (float)v2.size() / samples;
    std::vector <float> n_v1;
    std::vector <float> n_v2;
    
    
    float sum = 0;
    int last_break = 0;
    for (int i = 0; i < v1.size(); i++){
        sum += v1[i].x;
        if (i >= s_v1){
            n_v1.push_back(sum / (float)(i - last_break));
            sum = 0;
            last_break = i;
        }
    }
    for (int i = 0; i < v2.size(); i++){
        sum += v2[i].x;
        if (i >= s_v2){
            n_v2.push_back(sum / (float)(i - last_break));
            sum = 0;
            last_break = i;
        }
    }

    float error = 0;
    for (int i = 0; i < samples; i++) {
        error += n_v1[i] + n_v2[samples - i];
    }
    
    return error;
    /*
    if (error < treshold) {
        return true;
    }
    else {
        return false;
    } 
    */

    
}
bool isFlat(std::vector < std::vector < float >> flat, int piece, int side, float treshold) {
  if (flat[piece][side] < treshold) {
    return true;
  }
  else {
    return false;
  }
}
std::vector<std::vector<int>> findBestMatch(std::vector<std::vector<std::vector<cv::Point2f>>> puzzle, int piece, int side, int num, bool flat[70][4],std::vector<int> used) {
    std::cout << "piece-" << piece << "\n";
    std::cout << "side-" << side << "\n";
    std::vector<cv::Point2f> base = puzzle[piece][side];

    std::vector<std::vector<int>> match(num);
  std::vector<float> small;

  for (int i = 0; i < num; i++) {
    small.push_back(INT_MAX);
    //std::cout << "max" << "\n";
  
  }
  if (flat[piece][side]) {

      std::vector<std::vector<int>> issue;
      issue.push_back(std::vector<int>(piece, side));
      return  issue;
 
  } 
  std::cout << used.size() << "\n";
  for (int j = 0; j < num; j++) {
      for (int n = 0; n < puzzle.size(); n++) {

          for (int i = 0; i < puzzle[n].size(); i++) {
              if (flat[n][i] || piece == n) {

              }
              else {

                  float error = compareSide(base, puzzle[n][i], 1, 500);

                  std::vector<int> k;
                  int p = 0;
                  for (int l = 0; l < num; l++) {
                      if (error == small[l])
                          p = 1;
                  }
                  for (int f = 0; f < used.size(); f++) {
                  
                      if (n == used[f]) {
                         p = 1; 
                        // std::cout << used[f] << "\n";
                      }
                  }
                    if (error < small[j] && p !=1) {

                        // small.erase(small.begin() + j);
                        //  small.push_back(error);
                        // match.erase(match.begin() + j);
                        small[j] = error;
                        k.push_back(n);
                        k.push_back(i);
                        match[j] = k;
                        // match.push_back(k);
                        
                    }
              }
              
          }
      }
      std::cout << "piece------" << match[j][0] << "\n";
      std::cout << "side-----" << match[j][1] << "\n";
      std::cout << "------" << "\n";
  }
  std::cout << "ok" << "\n";
  return match;
}/*
std::vector<std::vector<cv::Point2f>> displayBest(std::vector < std::vector < cv::Point2f>> master_white_pixels, std::vector < std::vector < cv::Point2f >> masterCorner,int array[10][7][5]) {
    int k = 0;
    for (int x = 0; x < 10; x++) {
        for (int y = 0; y < 7; y++) {
            if (k == 0) {
                for (int i = 0; i < master_white_pixels[array[x][y][0]].size(); i++) {
                    master_white_pixels[array[x][y][4]][i].x = master_white_pixels[array[x][y][4]][i].x - masterCorner[array[x][y][4]][array[x][y][1]].x;
                    master_white_pixels[array[x][y][4]][i].y = master_white_pixels[array[x][y][4]][i].y - masterCorner[array[x][y][4]][array[x][y][1]].y;
                }
                for (int i = 0; i < 4; i++) {
                    double offsetx = masterCorner[array[x][y][4]][array[x][y][1]].x;
                    double offsety = masterCorner[array[x][y][4]][array[x][y][1]].y;
                    masterCorner[array[x][y][4]][i].x = masterCorner[array[x][y][4]][i].x - offsetx;
                    masterCorner[array[x][y][4]][i].y = masterCorner[array[x][y][4]][i].y - offsety;
                }
                double angle = array[x][y][0] * CV_PI / 2;
                for (int i = 0; i < master_white_pixels[array[x][y][0]].size(); i++) {
                    double x1 = master_white_pixels[array[x][y][0]][i].x;
                    double y1 = master_white_pixels[array[x][y][0]][i].y;
                master_white_pixels[array[x][y][0]][i].x = x1 * cos(angle) - y1 * sin(angle);
                master_white_pixels[array[x][y][0]][i].y = x1 * sin(angle) + y1 * cos(angle);

                }
                for (int i = 0; i < 4; i++) {
                    double x1 = masterCorner[array[x][y][0]][array[x][y][1]].x;
                    double y1 = masterCorner[array[x][y][0]][array[x][y][1]].y;
                    masterCorner[array[x][y][0]][array[x][y][1]].x = x1 * cos(angle) - y1 * sin(angle);
                    masterCorner[array[x][y][0]][array[x][y][1]].y = x1 * sin(angle) + y1 * cos(angle);
                } 
                k = 1;
            }
            else {

            }

        }
    }
}*/
void gridDisplay(std::vector<std::vector < cv::Point2f>> master_white_pixels, int array[10][7][5]) {
  std::vector <cv::Point2f> image;
  for (int x = 0; x < 10; x++) {
    for (int y = 0; y < 7 ; y++) {
      for (int count = 0; count < master_white_pixels[array[x][y][4]].size(); count++) {
        float angle = CV_PI / 2 * array[x][y][0];
        float x1 = master_white_pixels[array[x][y][4]][count].x;
        float y1 = master_white_pixels[array[x][y][4]][count].y;
        image.push_back(cv::Point2f((x1 * cos(angle) - y1 * sin(angle)) + (float)(x * 600) + 600.0, (x1 * sin(angle) + y1 * cos(angle)) + (float)(y * 600) + 600.0));
      }
    }
  }
  cv::Mat display(7000,7000 , CV_8UC3, cv::Scalar(255, 255, 255));
  display = cv::Mat(image).reshape(1);
  cv::resize(display, display, cv::Size((194 * 4), (194 * 4)));
  cv::imshow("Image", display);
  cv::waitKey(0);
} 
cv::Point2f parse_line(const std::string& line) {
    std::istringstream iss(line);
    float x, y;
    iss >> x >> y;
    return cv::Point2f(x, y);
}
std::vector<cv::Point2f> read_vector(const std::string& filename) {
    std::vector<cv::Point2f> points;
    std::ifstream infile(filename);

    if (infile) {
        std::string line;
        while (std::getline(infile, line)) {
            cv::Point2f point = parse_line(line);
            points.push_back(point);
        }
    }
    else {
        std::cerr << "Error opening file: " << filename << std::endl;
    }

    return points;
}

std::vector<float> read_float(const std::string& filename) {
    std::vector<float> points;
    std::ifstream infile(filename);

    if (infile) {
        std::string line;
        while (std::getline(infile, line)) {
            std::istringstream iss(line);
            float x;
            iss >> x;
            points.push_back(x);
        }
    }
    else {
        std::cerr << "Error opening file: " << filename << std::endl;
    }

    return points;
}

int x_sum;
int y_sum;
bool write = true;
int main() {
    std::string state;
    std::cin >> state;
    //Variables
    std::vector < std::vector < float >> flat(70);
    std::vector < std::vector < cv::Point2f >> masterCorner;
    std::vector < std::vector < std::vector < cv::Point2f >>> master_sorted_diviation;
    std::vector < std::vector < cv::Point2f>> master_white_pixels;

    int step = 0;
    //
    status(step, "Initilized code");
    for (int n = 51; n <= number_of_pieces && state == "w"; n++) {
        std::vector < std::vector < cv::Point2f >> normal_diviation(4);
        std::vector < std::vector < cv::Point2f >> sorted_diviation(4);
        std::string imagePath = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(n) + ".jpg";
        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) {
            std::cout << "ERROR IMAGE" << imagePath << " NOT FOUND";
        }
        status(step, "Loaded image:" + imagePath);

        cv::Mat image_r;
        cv::resize(image, image_r, cv::Size((194 * 3), (259 * 3)));
        imshow("Image", image_r);
        cv::waitKey(10);

        cv::Mat new_image;
        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

        int threshold = 72; //Treshold value
        cv::threshold(image, image, threshold, 255, cv::THRESH_BINARY); //treshold the image
        status(step, "Converted " + imagePath + " to binary bitmap");
        cv::resize(image, image, cv::Size((1944), (2592))); // resize
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10, 10));
        cv::morphologyEx(image, image, cv::MORPH_CLOSE, kernel);
        cv::Mat kernel1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10, 10));
        cv::morphologyEx(image, image, cv::MORPH_OPEN, kernel1);

        status(step, "Filtered image");

        cv::resize(image, image_r, cv::Size((194 * 3), (259 * 3))); // resize
        imshow("Image", image_r);
        cv::waitKey(50);

        cv::Mat grad_x, grad_y;
        cv::Mat abs_grad_x, abs_grad_y;
        int scale = 1;
        int delta = 0;
        int ddepth = CV_16S;
        //Sobel(image, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_REPLICATE);
        //Sobel(image, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_REPLICATE);
        //convertScaleAbs(grad_x, abs_grad_x);
        //convertScaleAbs(grad_y, abs_grad_y);
        //addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, image);
        cv::Mat kernel2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(4, 4));
        cv::morphologyEx(image, image, cv::MORPH_GRADIENT, kernel2);
        status(step, "Sobel function ran on:" + imagePath);
        cv::resize(image, image_r, cv::Size((194 * 3), (259 * 3)));
        imshow("Image", image_r);
        cv::waitKey(50);
        status(step, "Displayed image:" + imagePath);

        cv::Mat mask;
        int blockSize = 30;
        int maxCorners = 4;
        double qualityLevel = 0.01;
        double minDistance = 560;
        std::vector < cv::Point2f > corners;
        cv::goodFeaturesToTrack(image, corners, maxCorners, qualityLevel, minDistance, mask, blockSize, 3, true, 0.04);
        status(step, "Ran cornerHarris() on image");
        cv::Mat img_with_points = image.clone();

        if (corners.size() != 4) {
            n--;
            cv::waitKey(0);
        }
        //draw each point as a red circle
        for (int i = 0; i < corners.size(); i++) {
            cv::circle(img_with_points, corners[i], 20, cv::Scalar(255, 255, 255), -1);
            x_sum = x_sum + corners[i].x;
            y_sum = y_sum + corners[i].y;
        }

        cv::Mat resized_img;
        cv::resize(img_with_points, resized_img, cv::Size((194 * 3), (259 * 3)));
        int avrage_x = x_sum / 4;
        int avrage_y = y_sum / 4;
        x_sum = 0;
        y_sum = 0;
        cv::Point2f middle(avrage_x, avrage_y);
        cv::imshow("Image", resized_img);
        cv::waitKey(50);
        //display the image with points in a window

        status(step, "Displayed corners");
        for (int o = 0; o < corners.size(); o++) {
            corners[o].x = corners[o].x - middle.x;
            corners[o].y = corners[o].y - middle.y;
        }

        int smallest1[2] = { INT_MAX, 0 };
        int smallest2[2] = { INT_MAX, 0 };
        for (int i = 0; i < corners.size(); i++) { //find the smallest .x points to get 2 ajacant points
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

        int referance_x = corners[smallest1[1]].x - corners[smallest2[1]].x; //find the angle to align the side between points smallest1[1] and smallest2[1] from corners
        int referance_y = corners[smallest1[1]].y - corners[smallest2[1]].y;
        double angle = atan2(referance_y, referance_x);
        angle = CV_PI / 2 - angle;
        for (cv::Point2f& point : corners) { //rotate the vector by angle
            double x = point.x;
            double y = point.y;
            point.x = x * cos(angle) - y * sin(angle);
            point.y = x * sin(angle) + y * cos(angle);
        }

        cv::Mat img_with_points2 = image.clone();


        std::vector < cv::Point2f > white_pixels; //create a vector for the outline
        for (int y = 0; y < image.rows; y++) { //map the points
            for (int x = 0; x < image.cols; x++) {
                if (image.at < uchar >(y, x) == 255) {
                    white_pixels.push_back(cv::Point(x, y));
                }
            }
        }

        for (int o = 0; o < white_pixels.size(); o++) { //offset the points
            white_pixels[o].x = white_pixels[o].x - middle.x;
            white_pixels[o].y = white_pixels[o].y - middle.y;
        }

        for (cv::Point2f& point : white_pixels) { //rotate the vector by the angle
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

        std::vector < cv::Point2f > reformed_corners(4); //sorted corners
        for (int i = 0; i < corners.size(); i++) { //sort the corner points
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
        for (int i = 0; i < reformed_corners.size(); i++) { //move points for the clossest outline points
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

        for (int i = 0; i < reformed_corners.size(); i++) { //desplay corner points
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
        if (write) {
            std::ofstream output_file;
            output_file.open("C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\1.txt");
            for (auto
                const& point : white_pixels) {
                output_file << point.x << " " << point.y << std::endl;
            }
            status(step, "Wrote vectors to file");
        }
        std::vector < std::vector < cv::Point2f >> sides(4);

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < white_pixels.size(); j++) {
                double angle1 = atan2(white_pixels[j].y, white_pixels[j].x);

                if ((angle1 > angle_min[i] && angle1 < angle_max[i] && i != 1) || (i == 1 && (angle1 < angle_min[i] || angle1 > angle_max[i]))) {

                    sides[i].push_back(white_pixels[j]);

                }
            }

        }
        status(step, "Order by angle");
        if (write) {
            for (int k = 0; k < 4; k++) {
                std::ofstream output_file0;
                output_file0.open("C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\1_" + std::to_string(k) + ".txt");
                for (auto
                    const& point : sides[k]) {
                    output_file0 << point.x << " " << point.y << std::endl;
                }
            }
            status(step, "Wrote vectors to file");
        }
        int treshold = 40;
        int treshold2 = 170;
        std::vector < std::vector < cv::Point2f >> sides_2(4);

        for (int j = 0; j < 4; j++) {
            int p = 0;
            int k = 1;

            cv::Point2f ref;
            cv::Point2f ref1;

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
        if (write) {
            for (int k = 0; k < 4; k++) {
                std::ofstream output_file0;
                output_file0.open("C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\1_" + std::to_string(k) + "1.txt");

                for (auto
                    const& point : sides_2[k]) {
                    output_file0 << point.x << " " << point.y << std::endl;
                }
            }
            status(step, "Wrote vectors to file");
        }
        int size = 0;
        int comb[4];

        for (int j = 0; j < 4; j++) {

            if (sides[j].size() > 1) {
                double smallest = INT_MAX;
                for (int i = 0; i < 4; i++) {

                    if (i != j) {
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

        status(step, "Matched clusters");
        if (write) {
            for (int k = 0; k < 4; k++) {
                std::ofstream output_file02;
                output_file02.open("C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\1_" + std::to_string(k) + "1.txt");
                for (auto
                    const& point : sides_2[k]) {
                    output_file02 << point.x << " " << point.y << std::endl;
                }
            }
            status(step, "wrote vectors to file");
        }
        for (int k = 0; k < 4; k++) {
            float flatness = 0;


            for (int i = 0; i < sides_2[k].size(); i++) {
                float ac = cv::norm(reformed_corners[k] - sides_2[k][i]);
                float bc = cv::norm(reformed_corners[ref[k + 1]] - sides_2[k][i]);
                float ab = cv::norm(reformed_corners[k] - reformed_corners[ref[k + 1]]);

                float S = (ab + ac + bc) / 2;
                float A = sqrt(S * (S - ab) * (S - ac) * (S - bc));

                if (doIntersect(reformed_corners[k], reformed_corners[ref[k + 1]], sides_2[k][i], cv::Point2f(0, 0))) {
                    normal_diviation[k].push_back(cv::Point2f(2 * A / ab, sqrt(abs(pow(ac, 2) - pow(2 * A / ab, 2)))));
                }
                else {
                    normal_diviation[k].push_back(cv::Point2f(-2 * A / ab, sqrt(abs(pow(ac, 2) - pow(2 * A / ab, 2)))));
                }
                if(abs(2 * A / ab)== abs(2 * A / ab)) //ignore nan
                flatness += abs(2 * A / ab);

            }
            flat[n - 1].push_back(flatness / sides_2[k].size());
            if (state == "w") {
                std::ofstream output_file0;
                output_file0.open("C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(n) + "flat.txt");

                for (auto const& float1 : flat[n - 1]) {
                    output_file0 << float1 << std::endl;
                }
                
            }
        }
        status(step, "Claculated normal diviation");
        status(step, "Wrote vectors to file----flat");

        for (int k = 0; k < 4; k++) {
            std::ofstream output_file32;
            if (write) {
                output_file32.open("C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\1_normalDiviation_" + std::to_string(k) + ".txt");
            }
            float smallest0 = 0;
            for (int i = 0; i < normal_diviation[k].size(); i++) {
                float smallest1 = INT_MAX;
                int place{};
                for (int m = 0; m < normal_diviation[k].size(); m++) {
                    if (smallest1 > normal_diviation[k][m].y && smallest0 < normal_diviation[k][m].y) {
                        //sorted_diviation[k].push_back(normal_diviation[k][m]);
                        smallest1 = normal_diviation[k][m].y;
                        place = m;
                    }
                }
                sorted_diviation[k].push_back(normal_diviation[k][place]);
                smallest0 = normal_diviation[k][place].y;
                if (write) {
                    output_file32 << normal_diviation[k][place].x << " " << normal_diviation[k][place].y << std::endl;
                }
            }
            if (write) {
                output_file32.close();
            }
        }

        status(step, "Sorted the normal diviation");

        cv::resize(img_with_points2, resized_img, cv::Size((194 * 3), (259 * 3)));
        cv::imshow("Image", resized_img);
        cv::waitKey(50);
        status(step, "displayed vector sides_2");


       // masterCorner.push_back(reformed_corners);
       // master_sorted_diviation.push_back(sorted_diviation);
       // master_white_pixels.push_back(white_pixels);
        if (state == "w") {
            for (int k = 0; k < 4; k++) {
                std::ofstream output_file0;
                output_file0.open("C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(n) + "sorted_diviation" + std::to_string(k) + ".txt");

                for (auto const& point : sorted_diviation[k]) {
                    output_file0 << point.x << " " << point.y << std::endl;
                }
            }
            status(step, "Wrote vectors to file----sorted_diviation");
        }
        if (state == "w") {
            std::ofstream output_file0;
            output_file0.open("C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(n) + "white_pixels.txt");

            for (auto const& point : white_pixels) {
                output_file0 << point.x << " " << point.y << std::endl;
            }
            status(step, "Wrote vectors to file----white_pixels         ");
        }
        if (state == "w") {
            std::ofstream output_file0;
            output_file0.open("C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(n) + "reformed_corners.txt");

            for (auto const& point : reformed_corners) {
                output_file0 << point.x << " " << point.y << std::endl;
            }
            status(step, "Wrote vectors to file----reformed_corners");
        }
        cout << "________________________________________________________________________________________________" << std::endl;
    }

     /*  if (state == "w") {
        cv::FileStorage fs1("flat.xml", cv::FileStorage::WRITE);
        fs1 << "data" << "[";
        for (int i = 0; i < flat.size(); ++i)
        {
            // Write each vector
            fs1 << "[:";
            for (int j = 0; j < flat[i].size(); ++j)
            {
                // Write each point
                fs1 << "[:" << flat[i][j] << "]";
            }
            fs1 << "]"; // close vector
        }
        fs1 << "]"; // close data
        fs1.release();
        cv::FileStorage fs2("masterCorner.xml", cv::FileStorage::WRITE);
        fs2 << "data" << "[";
        for (int i = 0; i < masterCorner.size(); ++i)
        {
            // Write each vector
            fs2 << "[:";
            for (int j = 0; j < masterCorner[i].size(); ++j)
            {
                // Write each point
                fs2 << "[:" << masterCorner[i][j].x << masterCorner[i][j].y << "]";
            }
            fs2 << "]"; // close vector
        }
        fs2 << "]"; // close data
        fs2.release();
        cv::FileStorage fs3("master_white_pixels.xml", cv::FileStorage::WRITE);
        fs3 << "data" << "[";
        for (int i = 0; i < master_white_pixels.size(); ++i)
        {
            // Write each vector
            fs3 << "[:";
            for (int j = 0; j < master_white_pixels[i].size(); ++j)
            {
                // Write each point
                fs3 << "[:" << master_white_pixels[i][j].x << master_white_pixels[i][j].y << "]";
            }
            fs3 << "]"; // close vector
        }
        fs3 << "]"; // close data
        fs3.release();
        cv::FileStorage fs4("master_sorted_diviation.xml", cv::FileStorage::WRITE);
        fs4 << "data" << "[";
        for (int i = 0; i < master_sorted_diviation.size(); ++i)
        {
            std::cout << "------" << std::endl;
            // Write each vector
            fs4 << "[:";
            for (int j = 0; j < master_sorted_diviation[i].size(); ++j)
            {
                // Write each point
                fs4 << "[:";
                for (int k = 0; j < master_sorted_diviation[i][j].size(); ++k)
                {
                    // Write each point
                    fs4 << "[:" << master_sorted_diviation[i][j][k].x << master_sorted_diviation[i][j][k].y << "]";
                }
                fs4 << "]";
            }
            fs4 << "]"; // close vector
        }
        fs4 << "]"; // close data
        fs4.release();

        cv::FileStorage fs("masterCorner.xml", cv::FileStorage::READ);

        cv::FileNode data = fs["data"];
        for (cv::FileNodeIterator itData = data.begin(); itData != data.end(); ++itData)
        {
            // Read each vector
            std::vector<cv::Point2f> vv;

            cv::FileNode pts = *itData;
            for (cv::FileNodeIterator itPts = pts.begin(); itPts != pts.end(); ++itPts)
            {
                // Read each point
                cv::FileNode pt = *itPts;

                cv::Point2f point;
                cv::FileNodeIterator itPt = pt.begin();
                point.x = *itPt; ++itPt;
                point.y = *itPt;

                vv.push_back(point);
            }
            masterCorner.push_back(vv);
        }
        
    */

    
    if (state == "r") {
      //  status(step, std::to_string(masterCorner.size()));
      //  status(step, std::to_string(master_sorted_diviation.size()));
      //  status(step, std::to_string(master_white_pixels.size()));
      //  status(step, std::to_string(flat.size()));
        for (int l = 1; l < 71; l++) {
            masterCorner.push_back(read_vector("C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(l) + "reformed_corners.txt"));
           // status(step, "Read file----C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(l) + "reformed_corners.txt");
            std::vector<std::vector<cv::Point2f>>sorted_diviation;
            for (int k = 0; k < 4; k++) {
                sorted_diviation.push_back(read_vector("C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(l) + "sorted_diviation" + std::to_string(k) + ".txt"));
           //     status(step, "Read file----C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(l) + "sorted_diviation" + std::to_string(k) + ".txt");
            }
            master_sorted_diviation.push_back(sorted_diviation);
            master_white_pixels.push_back(read_vector("C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(l) + "white_pixels.txt"));
           // status(step, "Read file----C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(l) + "white_pixels.txt");
            flat[l - 1] = read_float("C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(l) + "flat.txt");
           // status(step, "Read file----C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(l) + "flat.txt");
            status(step, std::to_string(l));
        }
        status(step, std::to_string(masterCorner.size()));
        status(step, std::to_string(master_sorted_diviation.size()));
        status(step, std::to_string(master_sorted_diviation[0].size()));
        status(step, std::to_string(master_white_pixels.size()));
        status(step, std::to_string(flat.size()));
        status(step, "started sort");
        int start = 0;
        bool flat_piece[70][4];
        for (int n = 0; n < number_of_pieces; n++) {
            int count = 0;
            for (int k = 0; k < 4; k++) {
                flat_piece[n][k] = isFlat(flat, n, k, 20);
                if (flat_piece[n][k]) {
                    count++;
                }
                //std::cout << n << "\t " << count << "\r";
                //status(step, "is it flat");
            }
            if (count == 2) {
                start = n;
            }
        }
        status(step, "Calculated flat");

        int array[10][7][5];

        array[0][0][4] = start;
        for (int offset = 0, state = 0; state != 1 && offset < 4; offset++) {
            std::cout << std::to_string(flat_piece[start][(0 + offset) % 4]) <<"|"<< std::to_string(flat_piece[start][(1 + offset) % 4]) << "\n";
            if (flat_piece[start][(0 + offset) % 4] && flat_piece[start][(1 + offset) % 4] && state != 1){
            array[0][0][0] = (0 + offset) % 4;
            std::cout << array[0][0][0] << "\n";
            array[0][0][1] = (1 + offset) % 4;
            std::cout << array[0][0][1] << "\n";
            array[0][0][2] = (2 + offset) % 4;
            std::cout << array[0][0][2] << "\n";
            array[0][0][3] = (3 + offset) % 4;
            std::cout << array[0][0][3] << "\n";
            state = 1;
            }
        }
        status(step, "Found first piece");
        int sample_size = 70;
        std::vector<int> used;
        for (int y = 0; y < 7; y++) {
            int x = 0;
            std::cout <<"----"<< array[x][y][4] << "\n";
            used.push_back(array[x][y][4]);

            std::vector<std::vector<int>> match = findBestMatch(master_sorted_diviation, array[x][y][4], array[x][y][2], sample_size, flat_piece,used);
            std::cout << "1" << "\n";
            if (x == 0) {
                std::cout << "11" << "\n";
                std::cout << "y=" <<y<< "\n";
                if (y == 6) { 
                    for (int counter = 0, state = 1; counter < sample_size && state != 0; counter++) {
                        if (flat_piece[match[counter][0]][(match[counter][1] - 1) % 4] && flat_piece[match[counter][0]][(match[counter][1] - 2) % 4]) {
                            std::cout << "10" << "\n";
                            array[x % 10][y+1][4] = match[counter][0];
                            std::cout << match[counter][0] << "\n";
                            array[x % 10][y+1][0] = (match[counter][1] - 1) % 4;
                            std::cout << (match[counter][1] - 1) % 4 << "\n";
                            array[x % 10][y+1][1] = (match[counter][1] + 0) % 4;
                            std::cout << (match[counter][1] + 0) % 4 << "\n";
                            array[x % 10][y+1][2] = (match[counter][1] + 1) % 4;
                            std::cout << (match[counter][1] + 1) % 4 << "\n";
                            array[x % 10][y+1][3] = (match[counter][1] + 2) % 4;
                            std::cout << (match[counter][1] + 2) % 4 << "\n";
                            state = 0;
                        }
                    }
                }
                else {
                    for (int counter = 0, state = 1; counter <sample_size && state != 0; counter++) {
                        std::cout << "-----" << "\n";
                        std::cout << match[counter][0] << "\n";
                        std::cout << match[counter][1] << "\n";
                        std::cout << flat_piece[match[counter][0]][(match[counter][1] - 1) % 4] << "\n";
                        if (flat_piece[match[counter][0]][(match[counter][1] - 1) % 4]) {
                            std::cout << "12" << "\n";
                            array[x % 10][y+1][4] = match[counter][0];
                            std::cout << match[counter][0] << "\n";
                            array[x % 10][y+1][0] = (match[counter][1] - 1) % 4;
                            std::cout << (match[counter][1] - 1) % 4 << "\n";
                            array[x % 10][y+1][1] = (match[counter][1] + 0) % 4;
                            std::cout << (match[counter][1] + 0) % 4 << "\n";
                            array[x % 10][y+1][2] = (match[counter][1] + 1) % 4;
                            std::cout << (match[counter][1] + 1) % 4 << "\n";
                            array[x % 10][y+1][3] = (match[counter][1] + 2) % 4;
                            std::cout << (match[counter][1] + 2) % 4 << "\n";
                            state = 0;
                        }
                    }
                }
            }
        }
        std::cout << "13" << "\n";
        for (int x = 0; x < 10; x++) {
            int y = 0;
            std::vector<std::vector<int>> match = findBestMatch(master_sorted_diviation, array[x][y][4], array[x][y][3], sample_size, flat_piece,used);
            if (y == 0) {
                if (x == 9) {
                    for (int counter = sample_size - 1, state = 1; counter >= 0 && state != 0; counter--) {
                        if (flat_piece[match[counter][0]][(match[counter][1] + 1) % 4] && flat_piece[match[counter][0]][(match[counter][1] - 2) % 4]) {
                            array[(x + 1) % 10][y + (x / 10)][5] = match[counter][0];
                            array[(x + 1) % 10][y + (x / 10)][0] = (match[counter][1] - 1) % 4;
                            array[(x + 1) % 10][y + (x / 10)][1] = (match[counter][1] + 0) % 4;
                            array[(x + 1) % 10][y + (x / 10)][2] = (match[counter][1] + 1) % 4;
                            array[(x + 1) % 10][y + (x / 10)][3] = (match[counter][1] + 2) % 4;
                            state = 0;
                        }
                    }
                }
                else {
                    for (int counter = sample_size - 1, state = 1; counter >= 0 && state != 0; counter--) {
                        if (flat_piece[match[counter][0]][(match[counter][1] + 1) % 4]) {
                            array[(x + 1) % 10][y + (x / 10)][5] = match[counter][0];
                            array[(x + 1) % 10][y + (x / 10)][0] = (match[counter][1] - 1) % 4;
                            array[(x + 1) % 10][y + (x / 10)][1] = (match[counter][1] + 0) % 4;
                            array[(x + 1) % 10][y + (x / 10)][2] = (match[counter][1] + 1) % 4;
                            array[(x + 1) % 10][y + (x / 10)][3] = (match[counter][1] + 2) % 4;
                            state = 0;
                        }
                    }
                }
            }
        }
        status(step, "Sorted");
        gridDisplay(master_white_pixels, array);
        status(step, "Displayed");

    }
  return 0;
}