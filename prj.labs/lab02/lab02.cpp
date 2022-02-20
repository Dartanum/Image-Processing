#include <opencv2/opencv.hpp>

const std::string 
SOURCE_PATH = "data/cross_0256x0256.png",
JPEG_SOURCE_PATH = "cross_0256x0256_025.jpg",
PNG_CHANNELS_PATH = "cross_0256x0256_png_channels.png",
JPG_CHANNELS_PATH = "cross_0256x0256_jpg_channels.png",
HISTOGRAMS_PATH = "cross_0256x0256_hists.png";

//0 - blue, 1 - green, 2 - red
cv::Mat getMatWithSimpleChannel(cv::Mat source, int channel) {
	cv::Mat nullMat, result;
	std::vector<cv::Mat> sourceImageChannels(3), resultChannels;

	cv::split(source, sourceImageChannels); //split source image into channels
	nullMat = cv::Mat::zeros(cv::Size(source.cols, source.rows), CV_8UC1); //make image with all null channels
	resultChannels = { nullMat, nullMat , nullMat };
	resultChannels[channel] = sourceImageChannels[channel];
	cv::merge(resultChannels, result);

	return result;
}

cv::Mat combineFourImage(std::vector<cv::Mat> items) {
	double height = items[0].rows, width = items[0].cols;
	cv::Mat result(height * 2, width * 2, items[0].type());
	cv::Rect2d rc = {0, 0, width, height};
	items[0].copyTo(result(rc));
	rc.x += width;
	items[1].copyTo(result(rc));
	rc.x -= width;
	rc.y += height;
	items[2].copyTo(result(rc));
	rc.x += width;
	items[3].copyTo(result(rc));

	return result;
}

cv::Mat getHistogram(cv::Mat source, std::string name) {
	cv::Mat b_hist, g_hist, r_hist; //array consists of number pixels certain color 
	int histSize = 256;
	std::vector<cv::Mat> sourceImageChannels(3);
	cv::split(source, sourceImageChannels);

	float range[] = { 0, 256 }; //every colors range (the upper boundary is exclusive)
	const float* histRange[] = { range };
	//calculate number of pixels for every shade of color and save to appropriate array 
	cv::calcHist(&sourceImageChannels[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, histRange, true, false);
	cv::calcHist(&sourceImageChannels[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, histRange, true, false);
	cv::calcHist(&sourceImageChannels[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, histRange, true, false);

	int hist_w = 512, hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);
	cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
	//linear normalization histogram source arrays for height of histogram
	cv::normalize(b_hist, b_hist, 0, hist_h, cv::NORM_MINMAX);
	cv::normalize(g_hist, g_hist, 0, hist_h, cv::NORM_MINMAX);
	cv::normalize(r_hist, r_hist, 0, hist_h, cv::NORM_MINMAX);
	//visualize histogram
	for (int i = 1; i < histSize; i++) {
		cv::line(histImage, 
			cv::Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			cv::Point(bin_w * i, hist_h - cvRound(b_hist.at<float>(i))),
			cv::Scalar(255, 0, 0), 2);
		cv::line(histImage, 
			cv::Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			cv::Point(bin_w * i, hist_h - cvRound(g_hist.at<float>(i))),
			cv::Scalar(0, 255, 0), 2);
		cv::line(histImage, 
			cv::Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			cv::Point(bin_w * i, hist_h - cvRound(r_hist.at<float>(i))),
			cv::Scalar(0, 0, 255), 2);
	}
	//make text under the histogram
	cv::Size textSize = cv::getTextSize(name, cv::FONT_HERSHEY_DUPLEX, 1.0, 1, 0);
	cv::Mat textArea(50, histImage.cols, histImage.type());
	cv::putText(textArea, name, cv::Point(textArea.cols / 2 - textSize.width / 2, textArea.rows / 2 + textSize.height / 2), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(0, 0, 0), 1);
	histImage.push_back(textArea);

	return histImage;
}

int main() {
	cv::Mat source = cv::imread(SOURCE_PATH);
	cv::imwrite(JPEG_SOURCE_PATH, source, { cv::IMWRITE_JPEG_QUALITY, 25 }); //convert source image to jpeg with quality 25%
	//visualisation 3 channels of source image
	cv::Mat sourceB, sourceG, sourceR, sourceMosaic;
	sourceB = getMatWithSimpleChannel(source, 0);
	sourceG = getMatWithSimpleChannel(source, 1);
	sourceR = getMatWithSimpleChannel(source, 2);
	sourceMosaic = combineFourImage({source, sourceR, sourceG, sourceB});
;	cv::imwrite(PNG_CHANNELS_PATH, sourceMosaic);
	//visualisation 3 channels of jpeg version source's image
	cv::Mat jpeg, jpegB, jpegG, jpegR, jpegMosaic;
	jpeg = cv::imread(JPEG_SOURCE_PATH);
	jpegB = getMatWithSimpleChannel(jpeg, 0);
	jpegG = getMatWithSimpleChannel(jpeg, 1);
	jpegR = getMatWithSimpleChannel(jpeg, 2);
	sourceMosaic = combineFourImage({ jpeg, jpegR, jpegG, jpegB });
	cv::imwrite(JPG_CHANNELS_PATH, sourceMosaic);
	//create histograms
	std::string sourceHistogramText = "Source image histogram", jpegHistogramText = "JPG 25% quality histogram";
	cv::Mat sourceImageHistogram = getHistogram(source, sourceHistogramText);
	cv::Mat jpegImageHistogram = getHistogram(jpeg, jpegHistogramText);
	sourceImageHistogram.push_back(jpegImageHistogram);
	cv::imwrite(HISTOGRAMS_PATH, sourceImageHistogram);
}