#include <opencv2/opencv.hpp>
#include <cmath>

const std::string 
	SOURCE_PATH = "data/cross_0256x0256.png",
	RGB_PATH = "lab03_rgb.png",
	GRE_PATH = "lab03_gre.png",
	GRE_RES_PATH = "lab03_gre_res.png",
	RGB_RES_PATH = "lab03_rgb_res.png",
	VIZ_FUNC_PATH = "lab03_viz_func.png";

int intensityFunction(int input) {
	return pow(input, (log(input) / log(256)));
}

cv::Mat visualisation(cv::Mat lookUpTable, int width, int height) {
	int bin = width / 256;
	int thinkness = 1;
	cv::Mat diagram(width, height, CV_8UC3);
	diagram = cv::Scalar(255, 255, 255);
	cv::Rect2i oX = {0, 0, thinkness, height};
	cv::Rect2i oY = {0, height - thinkness, width, thinkness};
	cv::rectangle(diagram, oX, cv::Scalar(0, 0, 255));
	cv::rectangle(diagram, oY, cv::Scalar(0, 0, 255));

	for (int i = 0; i < 512; i += bin) {
		cv::Rect2i point = {i, height - bin - lookUpTable.at<uchar>(0, i / bin) * bin, bin, bin};
		cv::rectangle(diagram, point, cv::Scalar(0, 0, 0));
	}
	return diagram;
}

int main() {
	cv::Mat source = cv::imread(SOURCE_PATH);
	cv::Mat sourceGS, sourceLUT, sourceGSLUT;
	cv::cvtColor(source, sourceGS, cv::COLOR_BGR2GRAY);
	cv::Mat lut(1, 256, CV_8U);
	for (int i = 0; i < 256; i++) {
		lut.at<uchar>(cv::Point(i, 0)) = intensityFunction(i);
	}
	cv::LUT(source, lut, sourceLUT);
	cv::Mat diagram = visualisation(lut, 512, 512);
	cv::LUT(sourceGS, lut, sourceGSLUT);

	cv::imwrite(RGB_PATH, source);
	cv::imwrite(GRE_PATH, sourceGS);
	cv::imwrite(GRE_RES_PATH, sourceGSLUT);
	cv::imwrite(RGB_RES_PATH, sourceLUT);
	cv::imwrite(VIZ_FUNC_PATH, diagram);
}