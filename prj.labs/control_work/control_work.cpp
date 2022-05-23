#include <opencv2/opencv.hpp>
#include <cmath>
#include <fstream>

cv::Mat geometryAverage(cv::Mat lhs, cv::Mat rhs) {
	cv::Mat result(lhs);
	result = 0;

	for (int i = 0; i < result.cols; i++) {
		for (int j = 0; j < result.rows; j++) {
			float value = sqrt(pow(lhs.at<float>(j, i), 2) + pow(rhs.at<float>(j, i), 2));
			result.at<float>(j, i) = value < 0 ? (value + 255) / 2 : value;
		}
	}

	return result;
}

int main() {
	int rows = 2, cols = 3, cellSize = 150;
	cv::Mat filtered1, filtered2;
	cv::Mat img(rows * cellSize, cols * cellSize, CV_32FC1);
	cv::Mat filter1(3, 3, CV_32FC1, new float[9] {1, 0, -1, 2, 0, -2, 1, 0, -1});
	cv::Mat filter2(3, 3, CV_32FC1, new float[9] {1, 2, 1, 0, 0, 0, -1, -2, -1});
	cv::Rect2i rect(0, 0, cellSize, cellSize);
	std::vector<int> colors = {0, 127, 255, 127, 255, 0};

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			cv::Mat cell(cellSize, cellSize, CV_32FC1);
			cell = colors[i * cols + j];
			int circleColor = colors[(i * cols + j + cols) % (rows * cols)];
			int radius = cellSize / 2.5;
			cv::Point circleCenter(cellSize / 2, cellSize / 2);
			cv::circle(cell, circleCenter, radius, circleColor, cv::FILLED);
			cell.copyTo(img(rect));
			rect.x = (j + 1) * rect.width;
		}
		rect.x = 0;
		rect.y = (i + 1) * rect.height;
	}
	cv::filter2D(img, filtered1, -1, filter1);
	cv::filter2D(img, filtered2, -1, filter2);

	cv::imwrite("result.png", img);
	cv::imwrite("filtered1.png", filtered1);
	cv::imwrite("filtered2.png", filtered2);
	cv::imwrite("geometry_average.png", geometryAverage(filtered1, filtered2));
}