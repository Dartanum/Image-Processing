#include <opencv2/opencv.hpp>
#include <cmath>
#include <fstream>
#include "json.hpp"

using json = nlohmann::json;

const std::string SOURCE_VIDEOS_PREFIX = "videos/vid_";
const std::string SOURCE_ETALON_MASKS = "masks.json";
const std::string OUTPUT_DIR = "output/";
const std::string VIDEO_FORMAT = ".mp4";
const int VIDEO_COUNT = 5;

cv::Mat readFrame(cv::VideoCapture& video, double frameNum) {
	cv::Mat frame;
	video.set(cv::CAP_PROP_POS_FRAMES, frameNum);
	video.read(frame);
	return frame;
}

void addBorders(cv::Mat& img) {
	cv::rectangle(img, cv::Point(0, 0), cv::Point(img.cols, img.rows), 0, 1);
}

cv::Mat morphProcessing(cv::Mat& img) {
	cv::Mat result;
	cv::InputArray kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(35, 10));

	cv::morphologyEx(img, result, cv::MORPH_CLOSE, kernel);
	cv::morphologyEx(result, result, cv::MORPH_OPEN, kernel);
	cv::dilate(result, result, kernel);

	return result;
}

bool intersect(cv::Rect& rect1, cv::Rect& rect2) {
	return rect1.contains(cv::Point(rect2.x, rect2.y)) || rect1.contains(cv::Point(rect2.x + rect2.width - 1, rect2.y + rect2.height - 1));
}

void createMask(cv::Mat& mask, cv::Mat& img) {
	for (int i = 0; i < mask.rows; i++) {
		int lastUpdatedInd = 0;
		for (int j = 0; j < mask.cols; j++) {
			if (img.at<uchar>(i, j) == 255) break;
			else {
				lastUpdatedInd++;
				mask.at<uchar>(i, j) = 0;
			}
		}
		for (int k = mask.cols - 1; k > lastUpdatedInd; k--) {
			if (img.at<uchar>(i, k) == 255) break;
			else {
				mask.at<uchar>(i, k) = 0;
			}
		}
	}
}

cv::Mat getMask(cv::Mat& img) {
	cv::Mat labels, stats, centroids;
	cv::Mat selectedImgPart(img.rows, img.cols, img.type(), cv::Scalar(0, 0, 0));
	cv::Mat mask(img.rows, img.cols, img.type(), 255);
	int maxCompInd = 0, maxCompStat = 0;

	cv::connectedComponentsWithStats(img, labels, stats, centroids);

	for (int i = 1; i < stats.rows; i++) {
		int square = stats.at<int>(i, 4);
		if (square > maxCompStat) {
			maxCompInd = i;
			maxCompStat = square;
		}
	}

	cv::Rect2i maxComponent = { stats.at<int>(maxCompInd, 0), stats.at<int>(maxCompInd, 1), stats.at<int>(maxCompInd, 2), stats.at<int>(maxCompInd, 3) };

	for (int i = maxComponent.x; i < maxComponent.x + maxComponent.width; i++) {
		for (int j = maxComponent.y; j < maxComponent.y + maxComponent.height; j++) {
			selectedImgPart.at<uchar>(j, i) = img.at<uchar>(j, i);
			mask.at<uchar>(j, i) = 255;
		}
	}

	for (int i = 1; i < stats.rows; i++) {
		cv::Rect2i component = { stats.at<int>(i, 0), stats.at<int>(i, 1), stats.at<int>(i, 2), stats.at<int>(i, 3) };
		if (i != maxCompInd && intersect(maxComponent, component)) {
			cv::rectangle(selectedImgPart, component, 255, cv::FILLED);
		}
	}

	createMask(mask, selectedImgPart);

	return mask;
}

std::string createFileName(int vidNum, int photoNum, std::string type = "") {
	return OUTPUT_DIR + "frame_" + type + std::to_string(photoNum) + "_vid_" + std::to_string(vidNum) + ".png";
}

std::vector<std::array<cv::Point, 4>> readMasksPoints(std::string path) {
	std::ifstream input(path);
	std::vector<std::array<cv::Point, 4>> result;
	auto jf = json::parse(input);

	for (auto& e : jf) {
		std::array<cv::Point, 4> points;
		points[0] = cv::Point(e["bottom-left"].get<std::array<int, 2>>()[0], e["bottom-left"].get<std::array<int, 2>>()[1]);
		points[1] = cv::Point(e["top-left"].get<std::array<int, 2>>()[0], e["top-left"].get<std::array<int, 2>>()[1]);
		points[2] = cv::Point(e["top-right"].get<std::array<int, 2>>()[0], e["top-right"].get<std::array<int, 2>>()[1]);
		points[3] = cv::Point(e["bottom-right"].get<std::array<int, 2>>()[0],e["bottom-right"].get<std::array<int, 2>>()[1]);
		result.push_back(points);
	}

	return result;
}

cv::Mat createEtalonMask(std::array<cv::Point, 4>& points, cv::Mat& img) {
	int i = 0;
	cv::Mat result(img.rows, img.cols, CV_8U, cv::Scalar(0, 0, 0));
	cv::Point maskPoints[1][4];

	maskPoints[0][0] = points[0];
	maskPoints[0][1] = points[1];
	maskPoints[0][2] = points[2];
	maskPoints[0][3] = points[3];
	const cv::Point* ppt[1] = { maskPoints[0] };
	int npt[] = { 4 };

	cv::fillPoly(result, ppt, npt, 1, cv::Scalar(255, 255, 255));

	return result;
}

cv::Mat createConcatenatedMasks(cv::Mat originalImg, cv::Mat& programmedMask, cv::Mat& etalonMask) {
	cv::Mat rgbImageChannels[3];
	
	cv::split(originalImg, rgbImageChannels);
	cv::max(rgbImageChannels[2], programmedMask, rgbImageChannels[2]);
	cv::max(rgbImageChannels[1], etalonMask, rgbImageChannels[1]);
	cv::merge(rgbImageChannels, 3, originalImg);

	return originalImg;
}

int calculateIntersectionMasks(cv::Mat programmedMask, cv::Mat etalonMask) {
	int rightValue = 255;
	int result = 0;

	for (int i = 0; i < programmedMask.rows; i++) {
		for (int j = 0; j < programmedMask.cols; j++) {
			if (programmedMask.at<uchar>(i, j) == rightValue && programmedMask.at<uchar>(i, j) == etalonMask.at<uchar>(i, j)) {
				result++;
			}
		}
	}

	return result;
}

int calculateUnionMasks(cv::Mat programmedMask, cv::Mat etalonMask) {
	int rightValue = 255;
	int result = 0;

	for (int i = 0; i < programmedMask.rows; i++) {
		for (int j = 0; j < programmedMask.cols; j++) {
			if (programmedMask.at<uchar>(i, j) == rightValue || etalonMask.at<uchar>(i, j) == rightValue) {
				result++;
			}
		}
	}

	return result;
}

float calculateAccuracy(cv::Mat programmedMask, cv::Mat etalonMask) {
	return (float) calculateIntersectionMasks(programmedMask, etalonMask) / calculateUnionMasks(programmedMask, etalonMask);
}

int main() {
	std::vector<cv::Mat> result;
	std::vector<std::array<cv::Point, 4>> masksPoints = readMasksPoints(SOURCE_ETALON_MASKS);
	int ind = 0;
	std::ofstream out(OUTPUT_DIR + "accuracy.txt", std::ios_base::out);

	for (int i = 1; i <= VIDEO_COUNT; i++) {
		cv::VideoCapture vid(SOURCE_VIDEOS_PREFIX + std::to_string(i) + VIDEO_FORMAT);
		int frameCount = vid.get(cv::CAP_PROP_FRAME_COUNT);

		for (int j = 1; j < 4; j++) {
			cv::Mat frame, frame_grayscale, frame_bin, frame_morph, programmed_mask, etalon_mask, concatenated_masks;
			std::string frameFileName = createFileName(i, j);

			frame = readFrame(vid, frameCount * (j + 1) / VIDEO_COUNT);
			cv::imwrite(frameFileName, frame);

			cv::cvtColor(frame, frame_grayscale, cv::COLOR_BGR2GRAY, 1);
			cv::imwrite(createFileName(i, j, "grayscale_"), frame_grayscale);

			cv::threshold(frame_grayscale, frame_bin, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
			addBorders(frame_bin);
			cv::imwrite(createFileName(i, j, "bin_"), frame_bin);

			frame_morph = morphProcessing(frame_bin);
			addBorders(frame_morph);
			cv::imwrite(createFileName(i, j, "morph_"), frame_morph);

			programmed_mask = getMask(frame_morph);
			addBorders(programmed_mask);
			cv::imwrite(createFileName(i, j, "mask_"), programmed_mask);

			etalon_mask = createEtalonMask(masksPoints[ind], frame);
			addBorders(etalon_mask);
			cv::imwrite(createFileName(i, j, "etalon_mask_"), etalon_mask);

			concatenated_masks = createConcatenatedMasks(frame, programmed_mask, etalon_mask);
			cv::imwrite(createFileName(i, j, "concatenated_masks_"), concatenated_masks);

			out << frameFileName.substr(OUTPUT_DIR.size()) << " " << std::setprecision(3) << calculateAccuracy(programmed_mask, etalon_mask) << '\n';
			ind++;
		}
	}
	out.close();
}