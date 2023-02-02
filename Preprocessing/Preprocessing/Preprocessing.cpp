#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cmath>

void Jpgpreprocesor(std::string input_image_path, std::string output_image_path) {
    // Load the image
    cv::Mat image = cv::imread(input_image_path);

    // Convert the image to grayscale
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // Save the grayscale image as a bitmap
    cv::imwrite(output_image_path, gray);
}
void runDerivative(std::string input_image_path, std::string output_image_path) {
    // Load the image
    cv::Mat image = cv::imread(input_image_path, cv::IMREAD_GRAYSCALE);

    // Run derivative function 
    cv::Mat derivative;
    cv::Sobel(image, derivative, CV_8U, 1, 1, 1, 255);

    // Save the derivative image 
    cv::imwrite(output_image_path, derivative);

}
void convertToBinaryBitmap(std::string input_image_path, std::string output_image_path, int threshold) {
    // Load the image
    cv::Mat image = cv::imread(input_image_path, cv::IMREAD_GRAYSCALE);

    // Apply thresholding to the image
    cv::Mat binary;
    cv::threshold(image, binary, threshold, 255, cv::THRESH_BINARY);

    // Save the binary image
    cv::imwrite(output_image_path, binary);
}

void removeBlackClusters(std::string input_image_path, std::string output_image_path, int cluster_threshold) {
    // Load the binary image
    cv::Mat image = cv::imread(input_image_path, cv::IMREAD_GRAYSCALE);

    // Create a binary image for storing the result
    cv::Mat result;

    // Convert the image to a binary image
    cv::threshold(image, result, 128, 255, cv::THRESH_BINARY);

    // Find all the contours in the image
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(result, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    // Iterate over the contours
    for (size_t i = 0; i < contours.size(); i++) {
        // If the contour is smaller than the cluster threshold,
        // draw it on the result image with a white color
        if (contours[i].size() < cluster_threshold) {
            cv::drawContours(result, contours, i, cv::Scalar(255), -1);
        }
    }

    // Save the result image
    cv::imwrite(output_image_path, result);
}
void removeWhiteClusters(std::string input_image_path, std::string output_image_path, int cluster_threshold) {
    // Load the binary image
    cv::Mat image = cv::imread(input_image_path, cv::IMREAD_GRAYSCALE);

    // Create a binary image for storing the result
    cv::Mat result;

    // Convert the image to a binary image
    cv::threshold(image, result, 128, 255, cv::THRESH_BINARY);

    // Find all the contours in the image
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(result, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    // Iterate over the contours
    for (size_t i = 0; i < contours.size(); i++) {
        // If the contour is smaller than the cluster threshold,
        // draw it on the result image with a white color
        if (contours[i].size() < cluster_threshold) {
            cv::drawContours(result, contours, i, cv::Scalar(0), -1);
        }
    }

    // Save the result image
    cv::imwrite(output_image_path, result);
}

std::vector<cv::Point2f> findCorners(std::string input_image_path, std::string output_image_path) {
    // Load the image
    cv::Mat image = cv::imread(input_image_path, cv::IMREAD_GRAYSCALE);

    // Create a binary image for storing the result
    cv::Mat binary;

    // Convert the image to a binary image
    cv::threshold(image, binary, 128, 255, cv::THRESH_BINARY);

    // Find the corners in the binary image
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(binary, corners, 10, 0.01, 75, cv::Mat(), 10, 3,0,0.4); 

    // Create an image for drawing the corners
    cv::Mat corner_image = cv::Mat::zeros(binary.size(), CV_8UC3);

    // Draw the corners on the image
    for (auto corner : corners) {
        cv::circle(corner_image, corner, 1, cv::Scalar(0, 0, 255), -1);
    }

    // Save the image with the corners
    cv::imwrite(output_image_path, corner_image);

    // Return the red points
    return corners;
}

void removeRuggedEdges(std::string input_image_path, std::string output_image_path) {
    // Load the binary image
    cv::Mat image = cv::imread(input_image_path, cv::IMREAD_GRAYSCALE);

    // Create a binary image for storing the result
    cv::Mat result;

    // Convert the image to a binary image
    cv::threshold(image, result, 128, 255, cv::THRESH_BINARY);

    // Remove rugged edges by dilating and eroding the image
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 1));
    cv::dilate(result, result, kernel);
    cv::erode(result, result, kernel);

    // Save the result image
    cv::imwrite(output_image_path, result);
}
std::vector<cv::Point2f> mapWhitePixels(const std::string image_path) {
    std::vector<cv::Point2f> white_pixels;
    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            if (image.at<uchar>(y, x) == 255) {
                white_pixels.push_back(cv::Point(x, y));
            }
        }
    }
    return white_pixels;
}
std::vector<cv::Point2f> findWhiteLines(std::string input_image_path, std::string output_image_path) {
    // Load the image
    cv::Mat image = cv::imread(input_image_path, cv::IMREAD_GRAYSCALE);

    // Create a binary image for storing the result
    cv::Mat binary;

    // Convert the image to a binary image
    cv::threshold(image, binary, 128, 255, cv::THRESH_BINARY);
    cv::Mat color_dst = cv::Mat::zeros(binary.rows, binary.cols, CV_8UC3);
    // Find all the lines in the image
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(binary, lines, 1, CV_PI / 180, 30, 20, 15);

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
                // color the intersection area differently
                cv::circle(color_dst, collision_candidate, 1, cv::Scalar(0, 0, 255), -1);
            }
        }
    }
    // Save the result image
    cv::imwrite(output_image_path, color_dst);
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
std::vector<cv::Point2f> comparePoints( std::vector<cv::Point2f> corner_points, std::vector<cv::Point2f> line_points, double threshold, std::string output_path) {

    cv::Mat color_dst = cv::Mat::zeros(426, 567, CV_8UC3);
    //create a vector to store the points
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
                // Mark the point with a red dot
                cv::circle(color_dst, corner_point, 3, cv::Scalar(0, 0, 255), -1);
                break;
            }
        }
    }
    // Save the output image
    cv::imwrite(output_path, color_dst);
    return points;
}
std::vector<cv::Point2f> keepFarthestPoints(std::vector<cv::Point2f> points, std::string output_path) {

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

    cv::Mat color_dst = cv::Mat::zeros(426, 567, CV_8UC3);
    for (size_t i = 0; i < farthestPoints.size(); i++) {
        // Mark the point with a red dot
        cv::Point2f point = farthestPoints[i];
        cv::circle(color_dst, point, 3, cv::Scalar(0, 0, 255), -1);
    }
    // Save the output image
    cv::imwrite(output_path, color_dst);


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
void displayEdgeTables(const std::vector<std::vector<std::pair<int, int>>>& subTables, const std::vector<cv::Point2f>& points) {
    cv::Mat img(426, 567, CV_8UC3, cv::Scalar(255, 255, 255));
    for (int i = 0; i < subTables.size(); i++) {
        for (int j = 0; j < subTables[i].size(); j++) {
            cv::line(img, points[subTables[i][j].first], points[subTables[i][j].second], cv::Scalar(0, 0, 0), 1);
        }
        cv::imshow("Edge Tables", img);
    }
 
    cv::waitKey(0);
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
void displayEdgeTable(const std::vector<std::pair<int, int>>& edgeTable, const std::vector<cv::Point2f>& points) {
    cv::Mat img(426, 567, CV_8UC3, cv::Scalar(255, 255, 255));
    for (int i = 0; i < edgeTable.size(); i++) {
        cv::line(img, points[edgeTable[i].first], points[edgeTable[i].second], cv::Scalar(0, 0, 0), 1);
    }
    cv::imshow("Edge Table", img);
    cv::waitKey(0);
}
void saveEdgeTable(const std::vector<std::pair<int, int>>& edgeTable, const std::string& filename) {
    std::ofstream outFile;
    outFile.open(filename);
    for (int i = 0; i < edgeTable.size(); i++) {
        outFile << edgeTable[i].first << " " << edgeTable[i].second << std::endl;
    }
    outFile.close();
}
void drawLines(std::string imagePath, std::vector<cv::Vec4f> lines) {
    cv::Mat image = cv::imread(imagePath);
    for (auto line : lines) {
        cv::line(image, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 0, 255), 2);
    }
  
    cv::waitKey(0);
    cv::imwrite(imagePath, image);
}
std::vector<std::pair<int, int>> createEdgeTable(const std::vector<cv::Point2f>& input1, const std::vector<cv::Point2f>& input2) {
    std::vector<std::pair<int, int>> edgeTable;
    std::vector<bool> usedPoints1(input1.size(), false);
    std::vector<bool> usedPoints2(input2.size(), false);

    // Find the closest point in input2 to the first point of input1
    int closest = 0;
    float closestDist = cv::norm(input2[0] - input1[0]);
    for (int i = 1; i < input2.size(); i++) {
        float dist = cv::norm(input2[i] - input1[0]);
        if (dist < closestDist) {
            closestDist = dist;
            closest = i;
        }
    }

    // Continue to find the closest point in input1 to the last point of the edge table
    int last = closest;
    while (true) {
        closestDist = std::numeric_limits<float>::max();
        closest = -1;
        for (int i = 0; i < input1.size(); i++) {
            if (!usedPoints1[i]) {
                float dist = cv::norm(input1[i] - input2[last]);
                if (dist < closestDist) {
                    closestDist = dist;
                    closest = i;
                }
            }
        }

        if (closest == -1) {
            break;
        }

        edgeTable.push_back(std::make_pair(last, closest));
        usedPoints1[closest] = true;
        last = closest;
    }

    // Find the closest point in input2 to the last point of the edge table
    closestDist = std::numeric_limits<float>::max();
    closest = -1;
    for (int i = 0; i < input2.size(); i++) {
        if (!usedPoints2[i]) {
            float dist = cv::norm(input2[i] - input1[last]);
            if (dist < closestDist) {
                closestDist = dist;
                closest = i;
            }
        }
    }

    edgeTable.push_back(std::make_pair(last, closest));
    usedPoints2[closest] = true;

    return edgeTable;
}
int main() {
    
    for (int number = 0; number < 6; number++) {
        std::string input_image_path = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
        input_image_path += ".jpg";
        std::string output_image_path = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
        output_image_path += "_0.bmp";
        std::cout << " |#     |  0% \r";
        Jpgpreprocesor(input_image_path, output_image_path);
        std::cout << " |##    |  20% \r";
        int threshold = 120;
        std::string output_binary_image_path = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
        output_binary_image_path += "_1.bmp";
        convertToBinaryBitmap(output_image_path, output_binary_image_path, threshold);
        std::cout << " |###   |  40% \r";
        std::string output_black_filtered_binary_image_path = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
        output_black_filtered_binary_image_path += "_2.bmp";
        int cluster_size_black = 120;
        removeBlackClusters(output_binary_image_path, output_black_filtered_binary_image_path, cluster_size_black);
        std::string output_white_filtered_binary_image_path = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
        output_white_filtered_binary_image_path += "_3.bmp";
        int cluster_size_white = 120;
        removeWhiteClusters(output_black_filtered_binary_image_path, output_white_filtered_binary_image_path, cluster_size_white);
        std::cout << " |####  |  60% \r";
        std::string output_soft_image_path = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
        output_soft_image_path += "_4.bmp";
        removeRuggedEdges(output_white_filtered_binary_image_path, output_soft_image_path);
        std::cout << " |##### |  80% \r";
        std::string output_derivative_image_path = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
        output_derivative_image_path += "_5.bmp";
        runDerivative(output_soft_image_path, output_derivative_image_path);
        std::cout << " |##### |  90% \r";
        std::string output_point_image_path = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
        output_point_image_path += "_6.bmp";
        std::vector<cv::Point2f> corner_points = findCorners(output_soft_image_path, output_point_image_path);
        std::string output_point_line_image_path = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
        output_point_line_image_path += "_7.bmp";
        int extend_length = 40;
        std::vector<cv::Point2f> line_points = findWhiteLines(output_derivative_image_path, output_point_line_image_path);
        std::string output_combine_point_image_path = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
        output_combine_point_image_path += "_9.bmp";
        double threshold_point = 10;
        std::vector<cv::Point2f> compared_points = comparePoints(corner_points, line_points, threshold_point, output_combine_point_image_path);
       
        std::string output_txt_path = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
        output_txt_path += "_1.txt";
        savePointsToTxt(corner_points, output_txt_path);
        std::string output_txt2_path = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
        output_txt2_path += "_2.txt";
        savePointsToTxt(line_points, output_txt2_path);
        std::string output_four_point_image_path = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
        output_four_point_image_path += "_10.bmp";
        std::vector<cv::Point2f> four_points = keepFarthestPoints(compared_points, output_four_point_image_path);
        std::string output_txt3_path = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
        output_txt3_path += "_3.txt";
        savePointsToTxt(four_points, output_txt3_path);
        std::vector<cv::Point2f> outline = mapWhitePixels(output_derivative_image_path);
        std::vector<cv::Point2f> middle = findMiddle(four_points);
        std::string output_txt4_path = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
        output_txt4_path += "_4.txt";
        savePointsToTxt(middle, output_txt4_path);
        std::cout << " |##### |  100%\r";
        movePoints(four_points, outline);
        std::vector<cv::Vec4f> lines_midle = extendLines(middle, four_points);
        std::cout << " |##### |  101%\r";
        std::string output_e_line_image_path = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
        output_e_line_image_path += "_11.bmp";
        std::vector<std::pair<int, int>> createEdgeTable(outline,four_points)
   //     std::vector<std::pair<int, int>> edge_table = createEdgeTable(outline, four_points);
   //     std::cout << " |##### |  102%\r";
  //      std::vector<std::vector<std::pair<int, int>>> eges = cutEdgeTable(edge_table, outline, four_points);
  //      std::cout << " |##### |  103%\r";
 //       displayEdgeTables(eges, outline);
   //     std::cout << " |##### |  104%\r";
  //      std::string output_txt5_path = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\" + std::to_string(number);
 //       output_txt5_path += "_5.txt";
  //      saveEdgeTable(edge_table, output_txt5_path);
        //displayEdgeTable(edge_table, outline);
    }
    return 0;
}