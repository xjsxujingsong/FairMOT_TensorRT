#pragma once
#include "opencv2/opencv.hpp"
#include <cc_util.hpp>
#include "builder/trt_builder.hpp"
#include "infer/trt_infer.hpp"

using namespace cv;
#include "config.h"
//template <typename T>
class Detection //should use boost::noncopyable //interface 
{
public:
	virtual ~Detection() {};

	virtual bool init() = 0;
	virtual void get_detection(cv::Mat& frame, std::vector<DetectionBox>& vec_db, std::vector<cv::Mat>& vec_features) = 0;
};

class FileDetector :public Detection
{
public:
	FileDetector() = delete;
	explicit FileDetector(FileDetectorConfig& config);
	virtual ~FileDetector();
	FileDetector(const FileDetector&) = delete;
	FileDetector& operator=(const FileDetector&) = delete;

	bool init();
#ifdef __ANDROID__
	bool init(AAssetManager* mgr) { return true; }
#endif
	void get_detection(cv::Mat& frame, std::vector<DetectionBox>& vec_db, std::vector<cv::Mat>& vec_features);
private:
	bool read_from_list();
	bool read_from_file();
	std::vector<DetectionBox> read_one_file(int index);
private:
	FileDetectorConfig& params_;
	int frame_idx_;
	std::vector<std::string> vec_det_filename_;
	std::vector<std::vector<DetectionBox>> vec_vec_det_;
};


class FairMOTDetector :public Detection
{
public:
	FairMOTDetector() = delete;
	explicit FairMOTDetector(FairMOTDetectorConfig& config);
	virtual ~FairMOTDetector();
	FairMOTDetector(const FairMOTDetector&) = delete;
	FairMOTDetector& operator=(const FairMOTDetector&) = delete;

	bool init();
	void get_detection(cv::Mat& frame, std::vector<DetectionBox>& vec_db, std::vector<cv::Mat>& vec_features);
private:
	cv::Rect restoreCenterNetBox(float dx, float dy, float dw, float dh, float cellx, float celly, int stride, cv::Size netSize, cv::Size imageSize);
	cv::Rect restoreCenterNetBox(float dx, float dy, float l, float t, float r, float b, float cellx, float celly, int stride, cv::Size netSize, cv::Size imageSize);
		cv::Scalar restoreCenterTracking(float ox, float oy, float cellx, float celly, int stride, cv::Size netSize, cv::Size imageSize);
	void preprocessCenterNetImageToTensorMOT(const cv::Mat& image, int numIndex, const std::shared_ptr<TRTInfer::Tensor>& tensor);

private:
	FairMOTDetectorConfig& params_;
	std::shared_ptr<TRTInfer::Engine> engine;
	int frame_idx_;
};

//    -m 
//template <typename T>
class DetectorFactory
{
public:
	// 	FaceDetectionFactory();
	// 	virtual ~FaceDetectionFactory();
	static Detection* create_object(/*const*/ DetectorConfig& config)
	{
		switch (config.method)
		{
		case DetectorMethod::FromFile:
			return new FileDetector(config.fd);
		case DetectorMethod::FromFairMOT:
			return new FairMOTDetector(config.fairmot);
			
		}
	}
};

