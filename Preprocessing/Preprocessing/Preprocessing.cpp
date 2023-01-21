#include <opencv2/opencv.hpp>

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
    cv::Sobel(image, derivative, CV_8U, 1, 1, 1, 40);

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

void findCorners(std::string input_image_path, std::string output_image_path, double min_angle, double max_angle) {
    // Load the image
    cv::Mat image = cv::imread(input_image_path, cv::IMREAD_GRAYSCALE);

    // Create a binary image for storing the result
    cv::Mat binary;

    // Convert the image to a binary image
    cv::threshold(image, binary, 128, 255, cv::THRESH_BINARY);

    // Find the corners in the binary image
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(binary, corners, 4, 0.1, 100, cv::Mat(), 25, true);

    // Create an image for drawing the corners
    cv::Mat corner_image = cv::Mat::zeros(binary.size(), CV_8UC3);

    // Draw the corners on the image
    for (auto corner : corners) {
        cv::circle(corner_image, corner, 5, cv::Scalar(0, 0, 255), -1);
    }

    // Save the image with the corners
    cv::imwrite(output_image_path, corner_image);
}

void removeRuggedEdges(std::string input_image_path, std::string output_image_path) {
    // Load the binary image
    cv::Mat image = cv::imread(input_image_path, cv::IMREAD_GRAYSCALE);

    // Create a binary image for storing the result
    cv::Mat result;

    // Convert the image to a binary image
    cv::threshold(image, result, 128, 255, cv::THRESH_BINARY);

    // Remove rugged edges by dilating and eroding the image
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10, 10));
    cv::dilate(result, result, kernel);
    cv::erode(result, result, kernel);

    // Save the result image
    cv::imwrite(output_image_path, result);
}


int main() {
    std::cout << " |#     |  0% \r";
    std::string input_image_path = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\Test_Image.jpg";
    std::string output_image_path = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\Test_Image.bmp";
    Jpgpreprocesor(input_image_path, output_image_path);
    std::cout << " |##    |  20% \r";
    int threshold = 120;
    std::string output_binary_image_path = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\Test_Image_binary.bmp";
    convertToBinaryBitmap(output_image_path, output_binary_image_path, threshold);
    std::cout << " |###   |  40% \r";
    std::string output_filtered_binary_image_path = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\Test_Image_Filtered_binary.bmp";
    int cluster_size = 100;
    removeBlackClusters(output_binary_image_path, output_filtered_binary_image_path, cluster_size);
    std::cout << " |####  |  60% \r";
    std::string output_soft_image_path = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\Test_Image_soft.bmp";
    removeRuggedEdges(output_filtered_binary_image_path, output_soft_image_path);
    std::string output_derivative_image_path = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\Test_Image_Derative.bmp";
    runDerivative(output_soft_image_path, output_derivative_image_path);
    std::cout << " |##### |  80% \r";
    std::cout << " |##### |  90% \r";
    std::string output_line_image_path = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\Test_Image_Line.bmp";
    int min_angle = 80;
    int max_angle = 100;
    findCorners(output_soft_image_path, output_line_image_path, min_angle, max_angle);
    std::cout << " |######|  99% ";
    int number_of_points = 10000;
   // std::string output_point_image_path = "C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\Test_Image_Point.bmp";
    //make(output_derivative_image_path, output_point_image_path, number_of_points);
    std::cout << " |######|  100% ";
    return 0;
}