#include <opencv2/opencv.hpp>
#include <cmath>
#include <chrono>

const int WIDTH = 768, HEIGHT = 60;
const double gammaC = 2.2;
const int thickness = 1;

cv::Mat createTestImage() {
    cv::Mat testImage(HEIGHT, WIDTH, CV_8UC1);
    uchar currentIntensity = 0;

    for (int x = 0; x < WIDTH; x++) {
        if (x % 3 == 0 && x != 0) {
            currentIntensity++;
        }
        for (int y = 0; y < HEIGHT; y++) {
            testImage.at<uchar>(y, x) = currentIntensity;
        }
    }

    return testImage;
}

cv::Mat gammaCorrection(cv::Mat& source, double gammaCoefficient) {
    cv::Mat result_64F(HEIGHT, WIDTH, CV_64F), result;

    auto start = std::chrono::steady_clock::now();
    cv::pow(source, gammaCoefficient, result_64F);
    auto end = std::chrono::steady_clock::now();

    std::chrono::duration<double> duration = (end - start) * 1000;
    std::cout << "cv::pow execution time: " << duration.count() << " ms" << std::endl;
    result_64F.convertTo(result, CV_8UC1, 255);

    return result;
}

cv::Mat gammaCorrectionOverrided(cv::Mat source, double gammaCoefficient) {
    cv::Mat result;

    auto start = std::chrono::steady_clock::now();
    for (int x = 0; x < WIDTH; x++) {
        for (int y = 0; y < HEIGHT; y++) {
            source.at<double>(y, x) = pow((float)source.at<double>(y, x), gammaCoefficient) * 255.0f;
        }
    }
    auto end = std::chrono::steady_clock::now();

    std::chrono::duration<double> duration = (end - start) * 1000;
    std::cout << "changing the value of cells execution time: " << duration.count() << " ms" << std::endl;
    source.convertTo(result, CV_8UC1);

    return result;
}

cv::Mat mergeImages(std::vector<cv::Mat> images) {
    cv::Mat canvas(HEIGHT * images.size() + thickness * 2, WIDTH + thickness * 2, CV_8UC1);
    double height = images.size() * HEIGHT;
    cv::Rect2d rc = { thickness, thickness, WIDTH, HEIGHT };
    cv::Rect2d rcBox = { thickness, thickness, WIDTH, height};

    for (cv::Mat element : images) {
        element.copyTo(canvas(rc));
        cv::rectangle(canvas, rc, { 255 }, thickness);
        rc.y += HEIGHT;
    }
    cv::rectangle(canvas, rcBox, { 0 }, thickness);

    return canvas;
}

int main() {
    cv::Mat I_1, I_1_64F, G_1, G_2, result;
    I_1 = createTestImage();
    I_1.convertTo(I_1_64F, CV_64F, 1.0 / 255);
    G_1 = gammaCorrection(I_1_64F, gammaC);
    G_2 = gammaCorrectionOverrided(I_1_64F, gammaC);
    result = mergeImages({ I_1, G_1, G_2 });
    cv::imwrite("image1.png", result);
}