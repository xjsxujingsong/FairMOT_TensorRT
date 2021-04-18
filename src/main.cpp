
#include <opencv2/opencv.hpp>
#include <cc_util.hpp>
#include "builder/trt_builder.hpp"
#include "infer/trt_infer.hpp"

using namespace cv;
using namespace std;


#define GPUID		0


void drawArrow(cv::Mat& img, cv::Point pStart, cv::Point pEnd, int len, int alpha,
	cv::Scalar color, int thickness, int lineType)
{
	const double PI = 3.1415926;
	Point arrow;
	double angle = atan2((double)(pStart.y - pEnd.y), (double)(pStart.x - pEnd.x));
	line(img, pStart, pEnd, color, thickness, lineType);

	arrow.x = pEnd.x + len * cos(angle + PI * alpha / 180);
	arrow.y = pEnd.y + len * sin(angle + PI * alpha / 180);
	line(img, pEnd, arrow, color, thickness, lineType);
	arrow.x = pEnd.x + len * cos(angle - PI * alpha / 180);
	arrow.y = pEnd.y + len * sin(angle - PI * alpha / 180);
	line(img, pEnd, arrow, color, thickness, lineType);
}

static Rect restoreCenterNetBox(float dx, float dy, float dw, float dh, float cellx, float celly, int stride, Size netSize, Size imageSize) {

	float scale = 0;
	if (imageSize.width >= imageSize.height)
		scale = netSize.width / (float)imageSize.width;
	else
		scale = netSize.height / (float)imageSize.height;

	float x = ((cellx + dx - dw * 0.5) * stride - netSize.width * 0.5) / scale + imageSize.width * 0.5;
	float y = ((celly + dy - dh * 0.5) * stride - netSize.height * 0.5) / scale + imageSize.height * 0.5;
	float r = ((cellx + dx + dw * 0.5) * stride - netSize.width * 0.5) / scale + imageSize.width * 0.5;
	float b = ((celly + dy + dh * 0.5) * stride - netSize.height * 0.5) / scale + imageSize.height * 0.5;
	return Rect(Point(x, y), Point(r + 1, b + 1));
}

static Scalar restoreCenterTracking(float ox, float oy, float cellx, float celly, int stride, Size netSize, Size imageSize) {

	float scale = 0;
	if (imageSize.width >= imageSize.height)
		scale = netSize.width / (float)imageSize.width;
	else
		scale = netSize.height / (float)imageSize.height;

	float x = ((cellx + ox) * stride - netSize.width * 0.5) / scale + imageSize.width * 0.5;
	float y = ((celly + oy) * stride - netSize.height * 0.5) / scale + imageSize.height * 0.5;
	float x0 = ((cellx) * stride - netSize.width * 0.5) / scale + imageSize.width * 0.5;
	float y0 = ((celly) * stride - netSize.height * 0.5) / scale + imageSize.height * 0.5;
	return Scalar(x0, y0, x, y);
}


static void preprocessCenterNetImageToTensorMOT(const Mat& image, int numIndex, const shared_ptr<TRTInfer::Tensor>& tensor) {

	float scale = 0;
	int outH = tensor->height();
	int outW = tensor->width();
	if (image.cols >= image.rows)
		scale = outW / (float)image.cols;
	else
		scale = outH / (float)image.rows;

	Mat matrix = getRotationMatrix2D(Point2f(image.cols*0.5, image.rows*0.5), 0, scale);
	matrix.at<double>(0, 2) -= image.cols*0.5 - outW * 0.5;
	matrix.at<double>(1, 2) -= image.rows*0.5 - outH * 0.5;

	Mat outImage;
	cv::warpAffine(image, outImage, matrix, Size(outW, outH));

	vector<Mat> ms(image.channels());
	for (int i = 0; i < ms.size(); ++i)
		ms[i] = Mat(tensor->height(), tensor->width(), CV_32F, tensor->cpu<float>(numIndex, i));

	outImage.convertTo(outImage, CV_32F, 1 / 255.0);
	split(outImage, ms);

}


static vector<ccutil::BBox> detectBoundingbox_FairMot(const shared_ptr<TRTInfer::Engine>& boundingboxDetect_, const Mat& image, float threshold = 0.3) {

	if (boundingboxDetect_ == nullptr) {
		INFO("detectBoundingbox failure call, model is nullptr");
		return vector<ccutil::BBox>();
	}

	preprocessCenterNetImageToTensorMOT(image, 0, boundingboxDetect_->input());

	boundingboxDetect_->forward();



	auto outHM = boundingboxDetect_->tensor("hm");
	auto outHMPool = boundingboxDetect_->tensor("hm_pool");
	auto outWH = boundingboxDetect_->tensor("wh");
	auto outXY = boundingboxDetect_->tensor("reg");
	auto outID = boundingboxDetect_->tensor("id");
	const int stride = 4;

	vector<ccutil::BBox> bboxs;
	Size inputSize = boundingboxDetect_->input()->size();
	float sx = image.cols / (float)inputSize.width * stride;
	float sy = image.rows / (float)inputSize.height * stride;
	std::vector<cv::Mat> features;
	for (int class_ = 0; class_ < outHM->channel(); ++class_) {
		for (int i = 0; i < outHM->height(); ++i) {
			float* ohmptr = outHM->cpu<float>(0, class_, i);
			float* ohmpoolptr = outHMPool->cpu<float>(0, class_, i);
			for (int j = 0; j < outHM->width(); ++j) 
			{
				if (*ohmptr == *ohmpoolptr && *ohmpoolptr > threshold) 
				{

					float dx = outXY->at<float>(0, 0, i, j);
					float dy = outXY->at<float>(0, 1, i, j);
					float dw = outWH->at<float>(0, 0, i, j);
					float dh = outWH->at<float>(0, 1, i, j);
					ccutil::BBox box = restoreCenterNetBox(dx, dy, dw, dh, j, i, stride, inputSize, image.size());
					box = box.box() & Rect(0, 0, image.cols, image.rows);
					box.label = class_;
					box.score = *ohmptr;

					if (box.area() > 0)
					{
						bboxs.push_back(box);
						cv::Mat fea(1, outWH->width(), outWH->at<float>(0, i, j), CV_32FC1);
						float*tmpdata = (float*)fea.data;
						features.emplace_back(fea);
					}
				}
				++ohmptr;
				++ohmpoolptr;
			}
		}
	}
	return bboxs;
}


static float commonExp(float value) {

	float gate = 1;
	float base = exp(gate);
	if (fabs(value) < gate) 
		return value * base;

	if (value > 0) {
		return exp(value);
	}
	else {
		return -exp(-value);
	}
}


Mat padImage(const Mat& image, int stride = 32) {

	int w = image.cols;
	if (image.cols % stride != 0) 
		w = image.cols + (stride - (image.cols % stride));

	int h = image.rows;
	if (image.rows % stride != 0)
		h = image.rows + (stride - (image.rows % stride));

	if (Size(w, h) == image.size())
		return image;

	Mat output(h, w, image.type(), Scalar(0));
	image.copyTo(output(Rect(0, 0, image.cols, image.rows)));
	return output;
}

void dladcnOnnx_fairmot_cauthy()
{

	INFOW("onnx to trtmodel...");

	if (!ccutil::exists("models/all_dla34.fp32.trtmodel")) {

		if (!ccutil::exists("models/all_dla34.onnx")) {
			INFOW(
				"models/all_dla34.onnx not found, download url: http://zifuture.com:1000/fs/public_models/dladcnv2.onnx "
				"or use centerNetDLADCNOnnX/dladcn_export_onnx.py to generate"
			);
			return;
		}

		TRTBuilder::compileTRT(
			TRTBuilder::TRTMode_FP32, {}, 1,
			TRTBuilder::ModelSource("models/all_dla34.onnx"),
			"models/all_dla34.fp32.trtmodel", nullptr, "", "",
			{ TRTBuilder::InputDims(3, 512, 512) }
		);
	}

	INFO("load model: models/all_dla34.fp32.trtmodel");
	auto engine = TRTInfer::loadEngine("models/all_dla34.fp32.trtmodel");
	if (!engine) {
		INFO("can not load model.");
		return;
	}

	INFO("forward...");
	
	//Mat image = imread("www.jpg");
	Mat image = imread("fps20[00_00_00][20210330-145814-0].jpg");
	vector<ccutil::BBox> objs;
	//for (int i = 0; i < 1; i++)
	{
		objs = detectBoundingbox_FairMot(engine, image, 0.3);
		objs = ccutil::nms(objs, 0.5);
	}

// 	auto total_start = std::chrono::steady_clock::now();
// 	int run_num = 100;
// 	for (int i = 0; i < run_num; i++)
// 	{
// 		objs = detectBoundingbox_FairMot(engine, image, 0.3);
// 		objs = ccutil::nms(objs, 0.5);
// 	}
// 	auto total_end = std::chrono::steady_clock::now();
// 	float inference_fps = run_num * 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
// 	std::ostringstream stats_ss;
// 	stats_ss << std::fixed << std::setprecision(2);
// 	stats_ss << "\n\n\n\nInference FPS: " << inference_fps << std::endl;
// 	auto stats = stats_ss.str();
// 	std::cout << stats << std::endl;

	INFO("objs.length = %d", objs.size());
	for (int i = 0; i < objs.size(); ++i) {
		auto& obj = objs[i];
		ccutil::drawbbox(image, obj);
	}

	imwrite("www.dla.draw.jpg", image);

#ifdef _WIN32
	cv::imshow("dla dcn detect", image);
	cv::waitKey();
	cv::destroyAllWindows();
#endif
	INFO("done.");
}


void dladcnOnnx_fairmot_hrnet18_cauthy()
{

	INFOW("onnx to trtmodel...");

	if (!ccutil::exists("models/cauthy_fairmot_all_hrnet_v2_w18.fp32.trtmodel")) {

		if (!ccutil::exists("models/cauthy_fairmot_all_hrnet_v2_w18.onnx")) {
			INFOW(
				"models/cauthy_fairmot_all_hrnet_v2_w18.onnx not found, download url: http://zifuture.com:1000/fs/public_models/dladcnv2.onnx "
				"or use centerNetDLADCNOnnX/dladcn_export_onnx.py to generate"
			);
			return;
		}

		TRTBuilder::compileTRT(
			TRTBuilder::TRTMode_FP32, {}, 1,
			TRTBuilder::ModelSource("models/cauthy_fairmot_all_hrnet_v2_w18.onnx"),
			"models/cauthy_fairmot_all_hrnet_v2_w18.fp32.trtmodel", nullptr, "", "",
			{ TRTBuilder::InputDims(3, 512, 512) }
		);
	}

	INFO("load model: models/cauthy_fairmot_all_hrnet_v2_w18.fp32.trtmodel");
	auto engine = TRTInfer::loadEngine("models/cauthy_fairmot_all_hrnet_v2_w18.fp32.trtmodel");
	if (!engine) {
		INFO("can not load model.");
		return;
	}

	INFO("forward...");

	//Mat image = imread("www.jpg");
	Mat image = imread("000020.jpg");
	vector<ccutil::BBox> objs;
	for (int i = 0; i < 5; i++)
	{
		objs = detectBoundingbox_FairMot(engine, image, 0.3);
		objs = ccutil::nms(objs, 0.5);
	}

	auto total_start = std::chrono::steady_clock::now();
	int run_num = 100;
	for (int i = 0; i < run_num; i++)
	{
		objs = detectBoundingbox_FairMot(engine, image, 0.3);
		objs = ccutil::nms(objs, 0.5);
	}
	auto total_end = std::chrono::steady_clock::now();
	float inference_fps = run_num * 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
	std::ostringstream stats_ss;
	stats_ss << std::fixed << std::setprecision(2);
	stats_ss << "\n\n\n\nInference FPS: " << inference_fps << std::endl;
	auto stats = stats_ss.str();
	std::cout << stats << std::endl;

	INFO("objs.length = %d", objs.size());
	for (int i = 0; i < objs.size(); ++i) {
		auto& obj = objs[i];
		ccutil::drawbbox(image, obj);
	}

	imwrite("www.dla.draw.jpg", image);

#ifdef _WIN32
	cv::imshow("dla dcn detect", image);
	cv::waitKey();
	cv::destroyAllWindows();
#endif
	INFO("done.");
}



int main() {
	//log保存为文件
	ccutil::setLoggerSaveDirectory("logs");
	TRTBuilder::setDevice(GPUID);
	//demoLinearOnnx();
	//demoOnnx_yolov3();

	//dladcnOnnx_cauthy();
	dladcnOnnx_fairmot_cauthy();
	
	return 0;
}