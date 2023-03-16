#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <opencv2/core.hpp>
#include <cmath>

const float pi = 3.14159265358979323846f;
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
    cv::goodFeaturesToTrack(image, corners, 10, 0.01, 75, cv::Mat(), 10, 3,0,0.4);

    // Return the corner points
    return corners;
}
cv::Mat removeRuggedEdges(cv::Mat image) {
   
    // Remove rugged edges by dilating and eroding the image
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 1));
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
    cv::HoughLinesP(image, lines, 1, CV_PI / 180, 30, 20, 15);

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
            p3 -= norm2 * 30;
            p4 += norm2 * 30;
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
void savePoints(std::string filename, std::vector<std::vector<cv::Point2f>> points) {
    std::ofstream out_file;
    out_file.open(filename);
    if (!out_file) {
        std::cout << "Error: unable to open file for writing" << std::endl;
        return;
    }
    for (const auto& side : points) {
        for (const auto& point : side) {
            out_file << point.x << " " << point.y << std::endl;
        }
        out_file << std::endl;
    }
    out_file.close();
    std::cout << "Successfully saved points to file: " << filename << std::endl;
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
std::vector<cv::Vec4f> extendLines(const std::vector<cv::Point2f>& start, const std::vector<cv::Point2f>& end) {

    // check if the input vectors have 1 and 4 points respectively
    if (start.size() != 1 || end.size() != 4) {
        std::cout << "Error: the first vector should have 1 point and the second vector should have 4 points" << std::endl;
        return {};
    }

    std::vector<cv::Vec4f> lines;

    // Iterate over the 4 points in the end vector
    for (int i = 0; i < 4; i++) {
        // Find the direction vector
        cv::Point2f direction = end[i] - start[0];
        // Scale the direction vector by 200 pixels
        direction *= 200;
        // Find the end point of the line
        cv::Point2f line_end = start[0] + direction;
        // Store the line as a Vec4f
        lines.push_back(cv::Vec4f(start[0].x, start[0].y, line_end.x, line_end.y));
    }
    // return the vector of lines
    return lines;
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
void displayPoints(const std::vector<cv::Point2f>& points, const cv::Scalar& color = cv::Scalar(255, 255, 255)) {
    // Create a black image to display the points on
    cv::Mat img = cv::Mat::zeros(cv::Size( 567,426), CV_8UC3);

    // Draw the points on the image
    for (int i = 0; i < points.size(); i++) {
        cv::circle(img, points[i], 1, color, -1);
    }

    // Display the image
    cv::imshow("Points", img);
    cv::waitKey(0);
    cv::destroyAllWindows();
}
void displayImage(const std::string& imagePath) {
    // Load the image from the file path
    cv::Mat img = cv::imread(imagePath);

    // Check if the image was loaded successfully
    if (img.empty()) {
        std::cerr << "Error: could not load image '" << imagePath << "'" << std::endl;
        return;
    }

    // Display the image
    cv::imshow("Image", img);
    cv::waitKey(0);
    cv::destroyAllWindows();
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
void compare_vectors(std::vector<cv::Point2f> v1, std::vector<cv::Point2f> v2) {
    double score = 0.0;
    double max_score = 0.0;

    for (const auto& p1 : v1) {
        for (const auto& p2 : v2) {
            double distance = cv::norm(p1 - p2);
            score += 1.0 / (1.0 + distance);
        }
    }

    max_score = std::max(v1.size(), v2.size());

    std::cout << score / max_score ;
    std::cout << "\n";
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
cv::Point2f move_to_origin(std::vector<cv::Point2f>& points) {
    if (points.empty()) {
        return cv::Point2f(0, 0);
    }
    cv::Point2f first_point = points[0];
    float offset_x = first_point.x;
    float offset_y = first_point.y;
  //  for (int i = 1; i < points.size(); i++) {
   //     offset_x = std::min(offset_x, points[i].x);
  //      offset_y = std::min(offset_y, points[i].y);
  //  }
    for (int i = 0; i < points.size(); i++) {
        points[i].x -= offset_x;
        points[i].y -= offset_y;
    }
    return cv::Point2f(offset_x, offset_y);
}
    double rotatePoints(std::vector<cv::Point2f>&points) {
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
        double rotation = angle * 180 / pi;
        return rotation;
    }
    void delete_similar_points(std::vector<cv::Point2f>& points) {
        if (points.size() < 11) {
            return;
        }
        cv::Point2f first = points[0];
        for (int i = points.size() -1 ; i >= points.size() - 11; i--) {
            if (std::abs(points[i].x - first.x) < 20 && std::abs(points[i].y - first.y) < 20) {
                points.erase(points.begin() + i);
            }
        }
    }
int main() {
    double lenght_array[100];

    for (int number = 0; number < 6; number++) {
        cv::Mat steps;
        std::string input_image_path = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
        input_image_path += ".jpg";
        std::cout << " |#     |  0% \r";
        steps = Jpgpreprocesor(input_image_path);

        std::cout << " |##    |  20% \r";
        int threshold = 120;
        steps = convertToBinaryBitmap(threshold, steps);

        std::cout << " |###   |  40% \r";
        int cluster_size_black = 120;
        steps = removeBlackClusters(cluster_size_black, steps);

        int cluster_size_white = 120;
        steps = removeWhiteClusters(cluster_size_white, steps);

        std::cout << " |####  |  60% \r";
        steps = removeRuggedEdges(steps);

        std::cout << " |##### |  80% \r";
        steps = runDerivative(steps);

        std::cout << " |##### |  90% \r";
        std::vector<cv::Point2f> outline = mapWhitePixels(steps);

        std::vector<cv::Point2f> corner_points = findCorners(steps);

        int extend_length = 40;
        std::vector<cv::Point2f> line_points = findWhiteLines(steps);

        double threshold_point = 10;
        std::vector<cv::Point2f> compared_points = comparePoints(corner_points, line_points, threshold_point);

        std::vector<cv::Point2f> four_points = keepFarthestPoints(compared_points);

        std::vector<cv::Point2f> middle = findMiddle(four_points);

        movePoints(four_points, outline);
        std::vector<cv::Vec4f> lines_midle = extendLines(middle, four_points);

        std::cout << " |##### |  101%\r";
        std::vector<cv::Point2f> reorderedPoints = reorderPoints(outline);

        std::string output_txt5_path = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
        output_txt5_path += "_5.txt";
        savePointsToTxt(reorderedPoints, output_txt5_path);

        std::cout << " |##### |  105%\r";
        std::vector<cv::Point2f> reorderedfourPoints = reorderPoints(four_points);

        std::string output_txt7_path = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
        output_txt7_path += "_7.txt";
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

        savefloatToTxt(a, output_txt8_path1_1);

        savePointsToTxt(isolatedPoints, output_txt8_path1);

        std::cout << " |##### |  115%\r";
        isolatedPoints = isolatePoints(reorderedPoints, reorderedfourPoints, 1, 2);

        std::string output_txt8_path2 = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
        output_txt8_path2 += "_n2.txt";
        std::string output_txt8_path2_1 = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
        output_txt8_path2_1 += "_o2.txt";
        std::string output_txt8_path2_2 = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
        output_txt8_path2_2 += "_a1.txt";
        if (isolatedPoints.size() > 1000) {
            modifyVector(isolatedPoints, reorderedPoints, 10000);
        }  

         o = move_to_origin(isolatedPoints);

        savePointToTxt(o, output_txt8_path2_1);

        a = rotatePoints(isolatedPoints);

        savefloatToTxt(a, output_txt8_path2_1);

        savePointsToTxt(isolatedPoints, output_txt8_path2);

      std::cout << " |##### |  120%\r";
       isolatedPoints = isolatePoints(reorderedPoints, reorderedfourPoints, 2, 3);

        std::string output_txt8_path3 = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
        output_txt8_path3 += "_n3.txt";
        std::string output_txt8_path3_1 = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
        output_txt8_path3_1 += "_o3.txt";
        std::string output_txt8_path3_2 = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
        output_txt8_path3_2 += "_a1.txt";
        if (isolatedPoints.size() > 1000) {
            modifyVector(isolatedPoints, reorderedPoints, 10000);
        } 

       o = move_to_origin(isolatedPoints);

       savePointToTxt(o, output_txt8_path3_1);

       a = rotatePoints(isolatedPoints);

       savefloatToTxt(a, output_txt8_path3_1);

       savePointsToTxt(isolatedPoints, output_txt8_path3);
       std::cout << " |##### |  125%\r";
    
       isolatedPoints = isolatePoints(reorderedPoints, reorderedfourPoints, 3, 0);

        std::string output_txt8_path4 = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
        output_txt8_path4 += "_n4.txt";
        std::string output_txt8_path4_1 = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
        output_txt8_path4_1 += "_o4.txt";
        std::string output_txt8_path4_2 = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
        output_txt8_path4_2 += "_a1.txt";
        if (isolatedPoints.size() > 1000) {
            modifyVector(isolatedPoints, reorderedPoints, 10000);
        } 

        o = move_to_origin(isolatedPoints);

        savePointToTxt(o, output_txt8_path4_1);

         a = rotatePoints(isolatedPoints);

        savefloatToTxt(a, output_txt8_path4_1);

        savePointsToTxt(isolatedPoints, output_txt8_path4);
        std::cout << " |##### |  130%\r";
       
     double  d1 = distance_between_points(reorderedfourPoints,0,1);
     double  d2 = distance_between_points(reorderedfourPoints,1,2);
     double  d3 = distance_between_points(reorderedfourPoints,2,3);
     double  d4 = distance_between_points(reorderedfourPoints,3,0);

        lenght_array[number*4] = d1;
        lenght_array[number*4 + 1] = d2;
        lenght_array[number*4 + 2] = d3;
        lenght_array[number*4 + 3] = d4;
        
    }
    for (int n = 0; n < 24; n++) {
        std::cout << lenght_array[n] << "\n";
    }
    for (int number1 = 0; number1 < 6; number1++) {
        for (int number = 1; number < 5; number++) { 

        }
       
       
    }
   
    return 0;
}