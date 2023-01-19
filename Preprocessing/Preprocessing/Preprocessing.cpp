#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    // Load the image
    cv::Mat image = cv::imread("C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\Test_Image.jpg");

    // Convert the image to grayscale
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // Save the grayscale image as a bitmap
    cv::imwrite("C:\\Users\\a\\source\\repos\\jigsaw_solver\\Test_Image\\Test_Image.bmp", gray);
    return 0;
}