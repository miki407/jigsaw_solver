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

const int pieces_x = 10;
const int pieces_y = 10;
int lenght = 7;
int hight = 10;
int p_number = lenght * hight;


//[Number of pieces]
double Data[pieces_x * pieces_y][4][1][1];
int match[24];

cv::Mat Jpgpreprocesor(std::string input_image_path) {
    // Load the image
    cv::Mat image = cv::imread(input_image_path);

    // Convert the image to grayscale
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    return gray;
}
cv::Mat runDerivative (cv::Mat image) {
  
    // Run derivative function 
    cv::Mat derivative;
    cv::Sobel(image, derivative, CV_8U, 1, 1, 1, 255);

    return derivative;
}
cv::Mat convertToBinaryBitmap(int threshold, cv::Mat image) {
    // Apply thresholding to the image
    cv::Mat binary;
    cv::threshold(image, binary, threshold, 255, cv::THRESH_BINARY);

    return binary;
}
cv::Mat removeBlackClusters(int cluster_threshold, cv::Mat image) {
    
    // Find all the contours in the image
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(image, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    // Iterate over the contours
    for (size_t i = 0; i < contours.size(); i++) {
        // If the contour is smaller than the cluster threshold,
        // draw it on the result image with a white color
        if (contours[i].size() < cluster_threshold) {
            cv::drawContours(image, contours, i, cv::Scalar(255), -1);
        }
    }

  
    return image;
}
cv::Mat removeWhiteClusters(int cluster_threshold, cv::Mat image) {
   
    // Find all the contours in the image
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(image, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    // Iterate over the contours
    for (size_t i = 0; i < contours.size(); i++) {
        // If the contour is smaller than the cluster threshold,
        // draw it on the result image with a white color
        if (contours[i].size() < cluster_threshold) {
            cv::drawContours(image, contours, i, cv::Scalar(0), -1);
        }
    }
    return image;
}
std::vector<cv::Point2f> findCorners( cv::Mat image) {
    
    // Find the corners in the binary image
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(image, corners, 15, 0.01, 250, cv::Mat(), 35, 3,0,0.4);

    // Return the corner points
    return corners;
}
cv::Mat removeRuggedEdges(cv::Mat image) {
   
    // Remove rugged edges by dilating and eroding the image
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 1));
    cv::dilate(image, image, kernel);
    cv::erode(image, image, kernel);

    return image;
}
std::vector<cv::Point2f> mapWhitePixels(cv::Mat image) {
    std::vector<cv::Point2f> white_pixels;
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            if (image.at<uchar>(y, x) == 255) {
                white_pixels.push_back(cv::Point(x, y));
            }
        }
    }
    return white_pixels;
}
std::vector<cv::Point2f> findWhiteLines(cv::Mat image) {
  
    cv::Mat color_dst = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
    // Find all the lines in the image
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(image, lines, 2, CV_PI / 180, 30, 100, 110);

    //create a vector to store the collision points
    std::vector<cv::Point2f> collision_points;

    // Iterate over the lines
    for (size_t i = 0; i < lines.size(); i++) {
        cv::Vec4i l = lines[i];
        //Extend the lines by 30 pixels
        cv::Point2f p1 = cv::Point2f(l[0], l[1]);
        cv::Point2f p2 = cv::Point2f(l[2], l[3]);
        cv::Point2f diff = p2 - p1;
        cv::Point2f norm = diff / cv::norm(diff);
        p1 -= norm * 30;
        p2 += norm * 30;
        cv::line(color_dst, p1, p2, cv::Scalar(255, 0, 0),1);
        // check for intersection
        for (size_t j = i + 1; j < lines.size(); j++) {
            cv::Vec4i l2 = lines[j];
            cv::Point2f p3 = cv::Point2f(l2[0], l2[1]);
            cv::Point2f p4 = cv::Point2f(l2[2], l2[3]);
            cv::Point2f diff2 = p4 - p3;
            cv::Point2f norm2 = diff2 / cv::norm(diff2);
            p3 -= norm2 * 40;
            p4 += norm2 * 40;
            cv::Point2f collision_candidate;
            double d1 = (p1.x - p2.x) * (p3.y - p1.y) - (p1.y - p2.y) * (p3.x - p1.x);
            double d2 = (p1.x - p2.x) * (p4.y - p1.y) - (p1.y - p2.y) * (p4.x - p1.x);
            if (d1 * d2 >= 0) continue; // lines are parallel or coincident

            double ua = d1 / (d1 - d2);
            collision_candidate = p3 + (p4 - p3) * ua;
            // Check if the intersection point falls within the bounds of both lines
            double max_x1 = std::max(p1.x, p2.x);
            double min_x1 = std::min(p1.x, p2.x);
            double max_x2 = std::max(p3.x, p4.x);
            double min_x2 = std::min(p3.x, p4.x);
            double max_y1 = std::max(p1.y, p2.y);
            double min_y1 = std::min(p1.y, p2.y);
            double max_y2 = std::max(p3.y, p4.y);
            double min_y2 = std::min(p3.y, p4.y);
            if (collision_candidate.x >= min_x1 && collision_candidate.x <= max_x1 &&
                collision_candidate.x >= min_x2 && collision_candidate.x <= max_x2 &&
                collision_candidate.y >= min_y1 && collision_candidate.y <= max_y1 &&
                collision_candidate.y >= min_y2 && collision_candidate.y <= max_y2) {
                collision_points.push_back(collision_candidate);
            }
        }
    }
    return collision_points;

}
void savePointsToTxt(std::vector<cv::Point2f> points, std::string txt_path) {
    std::ofstream output_file;
    output_file.open(txt_path);
    for (auto const& point : points) {
        output_file << point.x << " " << point.y << std::endl;
    }
    output_file.close();
}
void savePointToTxt(cv::Point2f point, std::string txt_path) {
    std::ofstream output_file;
    output_file.open(txt_path);
    output_file << point.x << " " << point.y << std::endl;
    output_file.close();
}
void savefloatToTxt(float point, std::string txt_path) {
    std::ofstream output_file;
    output_file.open(txt_path);
    output_file << point;
    output_file.close();
}
std::vector<cv::Point2f> comparePoints( std::vector<cv::Point2f> corner_points, std::vector<cv::Point2f> line_points, double threshold) {

    std::vector<cv::Point2f> points;

     // Iterate over all the corner points
    for (size_t i = 0; i < corner_points.size(); i++) {
        cv::Point2f corner_point = corner_points[i];
        // Iterate over all the line points
        for (size_t j = 0; j < line_points.size(); j++) {
            cv::Point2f line_point = line_points[j];
            // Calculate the distance between the two points
            double distance = cv::norm(corner_point - line_point);
            // Check if the distance is less than the threshold
            if (distance < threshold) {
                points.push_back(corner_point);
                break;
            }
        }
    }

    return points;
}
std::vector<cv::Point2f> keepFarthestPoints(std::vector<cv::Point2f> points) {

    std::vector<cv::Point2f> farthestPoints;
    if (points.size() < 4) {
        return points;
    }
    double maxDistance = 0;
    cv::Point2f point1, point2, point3, point4;

    for (size_t i = 0; i < points.size() - 3; i++) {
        for (size_t j = i + 1; j < points.size() - 2; j++) {
            for (size_t k = j + 1; k < points.size() - 1; k++) {
                for (size_t l = k + 1; l < points.size(); l++) {
                    double distance1 = cv::norm(points[i] - points[j]);
                    double distance2 = cv::norm(points[i] - points[k]);
                    double distance3 = cv::norm(points[i] - points[l]);
                    double distance4 = cv::norm(points[j] - points[k]);
                    double distance5 = cv::norm(points[j] - points[l]);
                    double distance6 = cv::norm(points[k] - points[l]);
                    double currentMaxDistance = std::max({ distance1, distance2, distance3, distance4, distance5, distance6 });

                    if (currentMaxDistance > maxDistance) {
                        maxDistance = currentMaxDistance;
                        point1 = points[i];
                        point2 = points[j];
                        point3 = points[k];
                        point4 = points[l];
                    }
                }
            }
        }
    }
    farthestPoints.push_back(point1);
    farthestPoints.push_back(point2);
    farthestPoints.push_back(point3);
    farthestPoints.push_back(point4);

    return farthestPoints;
}
std::vector<cv::Point2f> findMiddle(const std::vector<cv::Point2f>& corners) {
    // check if the input vector has 4 points
    if (corners.size() != 4) {
        std::cout << "Error: the corners vector should have 4 points" << std::endl;
        return {};
    }
    // find the middle point
    cv::Point2f middle;
    for (const auto& corner : corners) {
        middle += corner;
    }
    middle.x /= 4;
    middle.y /= 4;
    return { middle };
}
void movePoints(std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2) {
    for (int i = 0; i < points1.size(); i++) {
        int closest = 0;
        float closestDist = cv::norm(points1[i] - points2[0]);
        for (int j = 1; j < points2.size(); j++) {
            float dist = cv::norm(points1[i] - points2[j]);
            if (dist < closestDist) {
                closestDist = dist;
                closest = j;
            }
        }
        points1[i] = points2[closest];
    }
}
std::vector<cv::Point2f> reorderPoints(const std::vector<cv::Point2f>& points) {
    // Initialize the reordered points vector with the first point
    std::vector<cv::Point2f> reorderedPoints;
    reorderedPoints.push_back(points[0]);

    // Remove the first point from the unprocessed points vector
    std::vector<cv::Point2f> unprocessedPoints(points.begin() + 1, points.end());

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
    }

    return reorderedPoints;
}

double distance_between_points(const std::vector<cv::Point2f>& points,int n1, int n2) {
    cv::Point2f p1 = points[n1];
    cv::Point2f p2 = points[n2];
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}
std::vector<cv::Point2f> isolatePoints(const std::vector<cv::Point2f>& points,const std::vector<cv::Point2f>& boundaries,int boundaryPoint1, int boundaryPoint2) {
    // Find the corresponding points in the input vector for the two boundary points
    cv::Point2f boundary1 = boundaries[boundaryPoint1];
    cv::Point2f boundary2 = boundaries[boundaryPoint2];
    int index1 = -1;
    int index2 = -1;
    for (int i = 0; i < points.size(); i++) {
        if (cv::norm(points[i] - boundary1) < 1e-6) {
            index1 = i;
        }
        if (cv::norm(points[i] - boundary2) < 1e-6) {
            index2 = i;
        }
        if (index1 != -1 && index2 != -1) {
            break;
        }
    }

    // Swap the indices if index1 is greater than index2
    if (index1 > index2) {
        int temp = index1;
        index1 = index2;
        index2 = temp;
    }

    // Check if the two boundary points are at opposite ends of the input vector
    bool isCircular = false;
    if ((index2 == points.size() - 500 && index1 == 499) || (index1 == points.size() - 500 && index2 == 499)) {
        isCircular = true;
    }

    // Create a new vector of points containing the points between the boundary points
    std::vector<cv::Point2f> isolatedPoints;
    if (isCircular) {
        for (int i = index1; i <= index2; i++) {
            isolatedPoints.push_back(points[i]);
        }
        isolatedPoints.insert(isolatedPoints.end(), points.begin(), points.begin() + index1);
        isolatedPoints.insert(isolatedPoints.end(), points.begin() + index2 + 1, points.end());
    }
    else {
        for (int i = index1; i <= index2; i++) {
            isolatedPoints.push_back(points[i]);
        }
    }

    return isolatedPoints;
}
void modifyVector(std::vector<cv::Point2f>& vector1, const std::vector<cv::Point2f>& vector2, int num) {
    // Remove any points from vector2 that are already in vector1
   // std::cout << vector1.size();
        std::vector<cv::Point2f> uniquePoints;
        for (const auto& point : vector2) {
            if (std::find(vector1.begin(), vector1.end(), point) == vector1.end()) {
                uniquePoints.push_back(point);
            }
        }

        // Replace the first num points in vector1 with the unique points from vector2
        if (vector1.size() > num) {
            for (int i = 0; i < num; i++) {
                vector1[i] = uniquePoints[i % uniquePoints.size()];
            }
        }
        // If vector1 has fewer than num points, append the remaining unique points from vector2
        else {
            vector1.resize(uniquePoints.size());
            std::copy(uniquePoints.begin(), uniquePoints.end(), vector1.begin());
        }
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
cv::Point2f read_point_from_txt(const std::string& filename) {
    // open the file for reading
    std::ifstream infile(filename);
    if (!infile) {
        throw std::runtime_error("Could not open file for reading");
    }

    // read the x and y coordinates from the file
    float x, y;
    infile >> x >> y;

    // return a new Point2f object with the coordinates
    return cv::Point2f(x, y);
}
float read_float_from_txt(const std::string& filename) {
    // open the file for reading
    std::ifstream infile(filename);
    if (!infile) {
        throw std::runtime_error("Could not open file for reading");
    }

    // read the float value from the file
    float value;
    infile >> value;

    // return the float value
    return value;
}
cv::Point2f move_to_origin(std::vector<cv::Point2f>& points) {
    if (points.empty()) {
        return cv::Point2f(0, 0);
    }
    cv::Point2f first_point = points[0];
    float offset_x = first_point.x;
    float offset_y = first_point.y;

    for (int i = 0; i < points.size(); i++) {
        points[i].x -= offset_x;
        points[i].y -= offset_y;
    }
    return cv::Point2f(offset_x, offset_y);
}
cv::Point2f move_to(std::vector<cv::Point2f>& points, cv::Point2f first_point) {
    if (points.empty()) {
        return cv::Point2f(0, 0);
    }
    float offset_x = first_point.x;
    float offset_y = first_point.y;

    for (int i = 0; i < points.size(); i++) {
        points[i].x -= offset_x;
        points[i].y -= offset_y;
    }
    return cv::Point2f(offset_x, offset_y);
}
double rotatePoints(std::vector<cv::Point2f>& points) {
    // Find the angle between the first and last point
    cv::Point2f lastPoint = points.back();
    double angle = atan2(lastPoint.x, lastPoint.y);

    // Rotate all points around (0, 0) by this angle
    for (cv::Point2f& point : points) {
        double x = point.x;
        double y = point.y;
        point.x = x * cos(angle) - y * sin(angle);
        point.y = x * sin(angle) + y * cos(angle);
    }

    // Calculate the rotation in degrees and return it
  
    return angle;
}
void rotatePointsByAngle(std::vector<cv::Point2f>& points,double angle) {

    // Rotate all points around (0, 0) by this angle
    for (cv::Point2f& point : points) {
        double x = point.x;
        double y = point.y;
        point.x = x * cos(angle) - y * sin(angle);
        point.y = x * sin(angle) + y * cos(angle);
    }

  
}
double compare_vectors_v1(std::vector<cv::Point2f> v1, std::vector<cv::Point2f> v2) {
    double score = 0.0;
    double max_score = 1.0;

    for (const auto& p1 : v1) {
        for (const auto& p2 : v2) {
            double distance = cv::norm(p1 - p2);
            score += 1.0 / (1.0 + distance);
        }
    }

    max_score = std::max(v1.size(), v2.size());
    double n = score / max_score;
  //  std::cout << score / max_score ;
   // std::cout << "\n";
    return n;
}

void displayPoints(const std::vector<cv::Point2f>& points, const cv::Mat& img) {

    int radius = 10; // set the radius of the points to 5 pixels

    // create a copy of the input image to draw the points on
    cv::Mat img_with_points = img.clone();

    // draw each point as a red circle
    for (int i = 0; i < points.size(); i++) {
        cv::circle(img_with_points, points[i], radius, cv::Scalar(255, 255, 255), -1);
    }
    cv::Mat resized_img;
    cv::resize(img_with_points, resized_img, cv::Size((194 * 3), (259 * 3)));

    // display the image with points in a window
    cv::imshow("Points", resized_img);
    cv::waitKey(0);
}
double compare_vectors_v2(const std::vector<cv::Point2f>& v1, const std::vector<cv::Point2f>& v2) {
    // interpolate v1
    std::vector<cv::Point2f> interp1;
    //std::cout << "works 1";
    cv::approxPolyDP(v1, interp1, 0.00001, false);

    // interpolate v2
    std::vector<cv::Point2f> interp2;
    //std::cout << "works 2";
    cv::approxPolyDP(v2, interp2,0.00001, false);
    
    
    // subtract interp1 from interp2 to get a line
    
    int size;

    std::cout << interp1.size();
    std::cout << interp2.size();
    
    if (interp1.size() < interp2.size()) {
        size = interp2.size() - 1;
    }
    else {
        size = interp1.size() - 1;
    }
    double stddev = 0;
    std::vector<cv::Point2f> line;
    for (int i = 0; i < size; i++) {
        //line[i] = interp1[i] - interp2[i];
        stddev +=  cv::norm(interp1[i] - interp1[2]);
    }
    /*
    // calculate the standard deviation of the line
    double stddev = 0;
    for (int i = 0; i < line.size(); i++) {
        stddev += cv::norm(line[i]);
    }
  
    stddev /= line.size();

    */
    // return a score from 0 to 1 based on how straight the line is
    return 1/stddev;// / cv::norm(line[0]);
}
double compare_vectors(const std::vector<cv::Point2f>& v1, const std::vector<cv::Point2f>& v2) {
    // interpolate v1
    std::vector<cv::Point2f> interp1;
    cv::approxPolyDP(v1, interp1, 0.0001, false);

    // interpolate v2
    std::vector<cv::Point2f> interp2;
    cv::approxPolyDP(v2, interp2, 0.0001, false);



    cv::resize(interp1, interp1, cv::Size(300, 1));
    cv::resize(interp2, interp2, cv::Size(300, 1));

    // calculate the difference between the two interpolated vectors
    std::vector<cv::Point2f> diff;
    if (interp1.size() != interp2.size()){
        std::cout << "ERRRRRRRRRROOOOOOOOOORRRRRRRRR";
    }

    for (int i = 0; i < interp1.size(); i++) {
        diff.push_back(cv::Point2f(interp1[i].x - interp2[i].x, interp1[i].y - interp2[i].y));
    }

    // calculate the score based on the average distance between points in the difference vector
    double dist_sum = 0;
    for (int i = 0; i < diff.size(); i++) {
        dist_sum += cv::norm(diff[i]);
    }
    double avg_dist = dist_sum / diff.size();
    double score = 1.0 / avg_dist ;
    return (score);
}
std::vector<cv::Point2f> combineVectors(std::vector<cv::Point2f> v1, std::vector<cv::Point2f> v2) {
    // Append all points from v2 to v1
    for (int i = 0; i < v2.size(); i++) {
        v1.push_back(v2[i]);
    }

    // Sort the points in v1 based on x coordinates
    std::sort(v1.begin(), v1.end(), [](const cv::Point2f& p1, const cv::Point2f& p2) {
        return p1.x < p2.x;
        });

    // Remove any duplicate points from v1
    auto it = std::unique(v1.begin(), v1.end());
    v1.resize(std::distance(v1.begin(), it));

    // Return the combined vector of points
    return v1;
}
void displayimage(cv::Mat img) {
    // resize the image
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size((194*3), (259*3)));

    // display the resized image in a window
    cv::imshow("Points", resized_img);
    cv::waitKey(0);
}
int main() {
//    double lenght_array[100];

    for (int number = 1; number <= p_number; number++) {
        cv::Mat steps;
        std::string input_image_path = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
        input_image_path += ".jpg";
        std::cout << " |#     |  0% \r";
        steps = Jpgpreprocesor(input_image_path);
   //     displayimage(steps);

        std::cout << " |##    |  20% \r";
        int threshold = 100;
        steps = convertToBinaryBitmap(threshold, steps);
   
        std::cout << " |###   |  40% \r";
        int cluster_size_black = 2020;
        steps = removeBlackClusters(cluster_size_black, steps);
    
        int cluster_size_white = 2020;
        steps = removeWhiteClusters(cluster_size_white, steps);
      
        std::cout << " |####  |  60% \r";
        steps = removeRuggedEdges(steps);
  //      displayimage(steps);
        std::cout << " |##### |  80% \r";
        steps = runDerivative(steps);
  //      displayimage(steps);
        std::cout << " |##### |  90% \r";
        std::vector<cv::Point2f> outline = mapWhitePixels(steps);
      
        std::vector<cv::Point2f> corner_points = findCorners(steps);
        displayPoints(corner_points,steps);
    
        int extend_length = 40;
        std::vector<cv::Point2f> line_points = findWhiteLines(steps);

        displayPoints(line_points,steps);
        double threshold_point = 50;
        std::vector<cv::Point2f> compared_points = comparePoints(corner_points, line_points, threshold_point);

        displayPoints(compared_points,steps);
        std::vector<cv::Point2f> four_points = keepFarthestPoints(compared_points);

        displayPoints(four_points,steps);
        std::vector<cv::Point2f> middle = findMiddle(four_points);

        movePoints(four_points, outline);
  
        std::cout << " |##### |  101%\r";
        std::vector<cv::Point2f> reorderedPoints = reorderPoints(outline);
    }
/*
        std::string output_txt5_path = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
        output_txt5_path += "_points.txt";
        savePointsToTxt(reorderedPoints, output_txt5_path);

        std::cout << " |##### |  105%\r";
        std::vector<cv::Point2f> reorderedfourPoints = reorderPoints(four_points);

        std::string output_txt7_path = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
        output_txt7_path += "_corner points.txt";
        savePointsToTxt(reorderedfourPoints, output_txt7_path);

        std::cout << " |##### |  110%\r";
        std::vector<cv::Point2f> isolatedPoints = isolatePoints(reorderedPoints, reorderedfourPoints, 0, 1);

        std::string output_txt8_path1 = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
        output_txt8_path1 += "_n1.txt";
        std::string output_txt8_path1_1 = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
        output_txt8_path1_1 += "_o1.txt";
        std::string output_txt8_path1_2 = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
        output_txt8_path1_2 += "_a1.txt";
        if (isolatedPoints.size() > 1000) {
            modifyVector(isolatedPoints, reorderedPoints, 10000);
        }

        cv::Point2f o = move_to_origin(isolatedPoints);

        savePointToTxt(o, output_txt8_path1_1);

        float a = rotatePoints(isolatedPoints);

        savefloatToTxt(a, output_txt8_path1_2);

        savePointsToTxt(isolatedPoints, output_txt8_path1);

        std::cout << " |##### |  115%\r";
        isolatedPoints = isolatePoints(reorderedPoints, reorderedfourPoints, 1, 2);

        std::string output_txt8_path2 = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
        output_txt8_path2 += "_n2.txt";
        std::string output_txt8_path2_1 = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
        output_txt8_path2_1 += "_o2.txt";
        std::string output_txt8_path2_2 = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
        output_txt8_path2_2 += "_a2.txt";
        if (isolatedPoints.size() > 1000) {
            modifyVector(isolatedPoints, reorderedPoints, 10000);
        }

        o = move_to_origin(isolatedPoints);

        savePointToTxt(o, output_txt8_path2_1);

        a = rotatePoints(isolatedPoints);

        savefloatToTxt(a, output_txt8_path2_2);

        savePointsToTxt(isolatedPoints, output_txt8_path2);

        std::cout << " |##### |  120%\r";
        isolatedPoints = isolatePoints(reorderedPoints, reorderedfourPoints, 2, 3);

        std::string output_txt8_path3 = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
        output_txt8_path3 += "_n3.txt";
        std::string output_txt8_path3_1 = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
        output_txt8_path3_1 += "_o3.txt";
        std::string output_txt8_path3_2 = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
        output_txt8_path3_2 += "_a3.txt";
        if (isolatedPoints.size() > 1000) {
            modifyVector(isolatedPoints, reorderedPoints, 10000);
        }

        o = move_to_origin(isolatedPoints);

        savePointToTxt(o, output_txt8_path3_1);

        a = rotatePoints(isolatedPoints);

        savefloatToTxt(a, output_txt8_path3_2);

        savePointsToTxt(isolatedPoints, output_txt8_path3);
        std::cout << " |##### |  125%\r";

        isolatedPoints = isolatePoints(reorderedPoints, reorderedfourPoints, 3, 0);

        std::string output_txt8_path4 = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
        output_txt8_path4 += "_n4.txt";
        std::string output_txt8_path4_1 = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
        output_txt8_path4_1 += "_o4.txt";
        std::string output_txt8_path4_2 = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
        output_txt8_path4_2 += "_a4.txt";
        if (isolatedPoints.size() > 1000) {
            modifyVector(isolatedPoints, reorderedPoints, 10000);
        }

        o = move_to_origin(isolatedPoints);

        savePointToTxt(o, output_txt8_path4_1);

        a = rotatePoints(isolatedPoints);

        savefloatToTxt(a, output_txt8_path4_2);

        savePointsToTxt(isolatedPoints, output_txt8_path4);
        std::cout << " |##### |  130%\r";

        double  d1 = distance_between_points(reorderedfourPoints, 0, 1);
        double  d2 = distance_between_points(reorderedfourPoints, 1, 2);
        double  d3 = distance_between_points(reorderedfourPoints, 2, 3);
        double  d4 = distance_between_points(reorderedfourPoints, 3, 0);

        lenght_array[number * 4] = d1;
        lenght_array[number * 4 + 1] = d2;
        lenght_array[number * 4 + 2] = d3;
        lenght_array[number * 4 + 3] = d4;
    }
/*
    }
    for (int n = 0; n < 24; n++) {
        std::cout << lenght_array[n] << "\n";
    }
    double matching[24 * 24];
    std::string name[24 * 24];
    int counter = 0;
    for (int number1 = 0; number1 < 6; number1++) {
        for (int number2 = 1; number2 < 5; number2++) {
            std::string n0 = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\";
            n0 += std::to_string(number1) + "_n" + std::to_string(number2) + ".txt";
            for (int number = 0; number < 6; number++) {
                if (number != number1) {


                    std::string n1 = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
                    n1 += "_n1.txt";
                    std::string n2 = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
                    n2 += "_n2.txt";
                    std::string n3 = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
                    n3 += "_n3.txt";
                    std::string n4 = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
                    n4 += "_n4.txt";
                    std::vector<cv::Point2f> v0 = read_vector(n0);
                    std::vector<cv::Point2f> v1 = read_vector(n1);
                    std::vector<cv::Point2f> v2 = read_vector(n2);
                    std::vector<cv::Point2f> v3 = read_vector(n3);
                    std::vector<cv::Point2f> v4 = read_vector(n4);
                    matching[counter * 4] = compare_vectors(v0, v1);
                    // std::cout << "_" << number1 << "_" << number << "_0-1\n";
                    matching[counter * 4 + 1] = compare_vectors(v0, v2);
                    //  std::cout << "_" << number1 << "_" << number << "_1-2\n";
                    matching[counter * 4 + 2] = compare_vectors(v0, v3);
                    //  std::cout << "_" << number1 << "_" << number << "_2-3\n";
                    matching[counter * 4 + 3] = compare_vectors(v0, v4);
                    //  std::cout << "_" << number1 << "_" << number << "_3-0\n";
                }
                std::cout << " |######|";
                counter++;


            }
        }
    }
    std::cout << "\n";

    for (int n = 0; n < 24; n++) {
        double largest = 0;
        int p3 = 0;
        int p2 = 0;
        for (int i = 0 + (n * 20); i < 20 + (n * 20); i++) {
            p3 = i;
            if (matching[i] > largest) {
                largest = matching[i];

                p2 = i - (n * 20) + 1;
            }
            if (i == 19 + (n * 20)) {
                std::cout << largest << "\n" << p3 << "\n" << p2 << "\n" << n + 1 << "\n";
               
                
            }
            
        }
         match[(n + 1)] = p2;
    }

    for (int n = 0; n < 6;n++) {
        for (int p2 = 1; p2 < 5; p2++) {
            std::string o1 = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(n);
            o1 += "_o" + std::to_string(p2) + ".txt";
            std::string input1_txt_path = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(n);
            input1_txt_path += "_points.txt";
            std::string a1 = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(n);
            a1 += "_a" + std::to_string(p2) + ".txt";
            std::vector<cv::Point2f> vector1 = read_vector(input1_txt_path);
            cv::Point2f offset = read_point_from_txt(o1);
            double angle = double(read_float_from_txt(a1));
            move_to(vector1, offset);
            rotatePointsByAngle(vector1, angle);
            //  savePointsToTxt(vector1, "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\vector.txt");
            //  std::cout << angle;
            //  cv::Point2f offset_0(-400, -200);
            //  move_to(vector1, offset_0);
            int g1 = match[n * 4 + p2];
            // int p3 = match[n*4+p2]- n1*4;
            // int n1 = match[24] / 4;
            int n1 = g1 /4;
            int p3 = g1 %4+1;
            std::cout << n1 << "\n";
            std::cout << p3 << "\n";
          
            std::string o2 = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(n1);
            o2 += "_o" + std::to_string(p3) + ".txt";
            std::string input2_txt_path = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(n1);
            input2_txt_path += "_points.txt";
            std::string a2 = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(n1);
            a2 += "_a" + std::to_string(p3) + ".txt";
            std::vector<cv::Point2f> vector2 = read_vector(input2_txt_path);
            cv::Point2f offset2 = read_point_from_txt(o2);
           double angle2 = double(read_float_from_txt(a2));
           move_to(vector2, offset2);
            rotatePointsByAngle(vector2, angle2);
            std::vector<cv::Point2f> vector3 = combineVectors(vector1, vector2);
            cv::Point2f offset3(-500, -300);
            move_to(vector3, offset3);
           displayPoints(vector3);
        }
    }
    */
    return 0;
}