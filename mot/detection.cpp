#include <fstream>
#include "utils.h"
#include "detection.h"

FileDetector::FileDetector(FileDetectorConfig& config)
	:params_(config)
	, frame_idx_(-1)
{
	clear_2d_vector(vec_vec_det_);
}

FileDetector::~FileDetector()
{
}

std::vector<DetectionBox> FileDetector::read_one_file(int index)
{
	std::vector<DetectionBox > vec_det;
	float score;

	std::ifstream infile(vec_det_filename_[index]);
	if (infile.fail())
	{
		std::cout << "read file fails: " << vec_det_filename_[index] << ", cannot read annotation file." << std::endl;
		return vec_det;
	}

	int fileindex, id;
	std::string detLine;
	std::istringstream ss;
	char ch;
	float tpx, tpy, tpw, tph;

	while (std::getline(infile, detLine))
	{
		ss.str(detLine);
		ss >> fileindex >> ch >> id >> ch;
		ss >> tpx >> ch >> tpy >> ch >> tpw >> ch >> tph >> ch >> score;
		ss.str("");
		if (score < params_.threshold)
		{
			continue;
		}
		int size = tph * tpw;
		if (params_.min_size != -1)
		{
			if (size < params_.min_size) continue;
		}
		if (params_.max_size != -1)
		{
			if (size > params_.max_size) continue;
		}
		vec_det.push_back(DetectionBox(fileindex, cv::Rect_<float>(cv::Point_<float>(tpx, tpy), cv::Point_<float>(tpx + tpw, tpy + tph)), score));
	}
	infile.close();
	return vec_det;
}

bool FileDetector::read_from_list()
{
	if (params_.det_list_name == "")
		return false;
	read_filelist(params_.det_list_name, vec_det_filename_);
	return true;
}

bool FileDetector::read_from_file()
{
	if (params_.det_file_name == "")
		return false;
	std::ifstream detectionFile;
	detectionFile.open(params_.det_file_name);

	if (!detectionFile.is_open())
	{
		std::cerr << "Error: can not find file " << params_.det_file_name << std::endl;
		return false;
	}
	float score;
	int fileindex, id;
	std::string detLine;
	std::istringstream ss;
	std::vector<DetectionBox> detbbx;
	char ch;
	float tpx, tpy, tpw, tph;

	while (std::getline(detectionFile, detLine))
	{
		ss.str(detLine);
		ss >> fileindex >> ch >> id >> ch;
		ss >> tpx >> ch >> tpy >> ch >> tpw >> ch >> tph >> ch >> score;
		ss.str("");
		if (score < params_.threshold)
		{
			continue;
		}
		detbbx.push_back(DetectionBox(fileindex, cv::Rect_<float>(cv::Point_<float>(tpx, tpy), cv::Point_<float>(tpx + tpw, tpy + tph)), score));
	}
	detectionFile.close();

	// 2. group detData by frame
	int maxFrame = 0;
	for (auto tb : detbbx) // find max frame number
	{
		if (maxFrame < tb.frame)
			maxFrame = tb.frame;
	}

	std::vector<DetectionBox> tempVec;
	for (int fi = 0; fi < maxFrame; fi++)
	{
		for (auto tb : detbbx)
			if (tb.frame == fi) // frame num starts from 1
				tempVec.push_back(tb);
		vec_vec_det_.push_back(tempVec);
		tempVec.clear();
	}
	return true;
}

bool FileDetector::init()
{
	if (params_.det_file_name != "" && params_.det_list_name != "")
	{
		std::cout << "please check two files exists, only one should be non-empty..." << std::endl;
		return false;
	}
	if (params_.det_file_name != "")
		return read_from_file();
	else if (params_.det_list_name != "")
		return read_from_list();
}

void FileDetector::get_detection(cv::Mat& frame, std::vector<DetectionBox>& vec_db, std::vector<cv::Mat>& vec_features)
{
	int im_w = frame.cols;
	int im_h = frame.rows;
	vec_db.clear();
	vec_features.clear();
	frame_idx_++;

	std::ifstream infile(vec_det_filename_[frame_idx_]+".size");
	if (infile.fail())
	{
		std::cout << "read file fails: " << vec_det_filename_[frame_idx_] << ", cannot read annotation file." << std::endl;
		return;
	}
	int det_cnt, det_dim, fea_cnt, fea_dim;
	infile >> det_cnt >> det_dim >> fea_cnt >> fea_dim;
	infile.close();
	std::vector<float> dets(det_cnt*det_dim);
	std::ifstream in_det(vec_det_filename_[frame_idx_] + ".det", ios::in | ios::binary);
	if (det_cnt>0)
		in_det.read((char *)&dets[0], sizeof(float)*dets.size());
	in_det.close();
	for (int i=0;i<det_cnt;i++)
	{
		float x1 = dets[i*det_dim]; if (x1 < 0)x1 = 0; if (x1 > (im_w - 1)) x1 = im_w - 1;
		float y1 = dets[i*det_dim + 1]; if (y1 < 0)y1 = 0; if (y1 > (im_h - 1)) y1 = im_h - 1;
		float x2 = dets[i*det_dim + 2]; if (x2 < 0)x2 = 0; if (x2 > (im_w - 1)) x2 = im_w - 1;
		float y2 = dets[i*det_dim + 3]; if (y2 < 0)y2 = 0; if (y2 > (im_h - 1)) y2 = im_h - 1;
		vec_db.push_back(DetectionBox(frame_idx_,
			cv::Rect_<float>(cv::Point_<float>(x1, y1),
				cv::Point_<float>(x2, y2)), dets[i*det_dim + 4]));
	}
	std::vector<float> features(fea_cnt*fea_dim);
	std::ifstream in_fea(vec_det_filename_[frame_idx_] + ".feature", ios::in | ios::binary);
	if (fea_cnt > 0)
		in_fea.read((char *)&features[0], sizeof(float)*features.size());
	in_fea.close();
	for (int i = 0; i < fea_cnt; i++)
	{
		cv::Mat im(1, fea_dim, CV_32FC1, &features[0] + i * fea_dim);
		float*data = (float*)im.data;
		vec_features.push_back(im.clone());
	}
}

FairMOTDetector::FairMOTDetector(FairMOTDetectorConfig& config)
	:params_(config)
	, frame_idx_(-1)
{
}

FairMOTDetector::~FairMOTDetector()
{
}

bool FairMOTDetector::init()
{
	INFOW("onnx to trtmodel...");
	auto aa = params_.model_file + ".fp32.trtmodel";
	if (!ccutil::exists(params_.model_file+".fp32.trtmodel")) {
		if (!ccutil::exists(params_.model_file + ".onnx")) {
			INFOW("onnx models not found");
			return false;
		}

		TRTBuilder::compileTRT(
			TRTBuilder::TRTMode_FP32, {}, 1,
			TRTBuilder::ModelSource(params_.model_file+".onnx"),
			params_.model_file+".fp32.trtmodel", nullptr, "", "",
				{ TRTBuilder::InputDims(3, 608, 1088) }
		);
	}

	engine = TRTInfer::loadEngine(params_.model_file + ".fp32.trtmodel");
	if (!engine) {
		INFO("can not load model.");
		return false;
	}

	return true;
}

cv::Rect FairMOTDetector::restoreCenterNetBox(float dx, float dy, float l, float t, float r, float b, float cellx, float celly, int stride, cv::Size netSize, cv::Size imageSize)
{
	float scale = 0;
	if (imageSize.width >= imageSize.height)
		scale = netSize.width / (float)imageSize.width;
	else
		scale = netSize.height / (float)imageSize.height;

	float xx = ((cellx + dx - l) * stride - netSize.width * 0.5) / scale + imageSize.width * 0.5;
	float yy = ((celly + dy - t) * stride - netSize.height * 0.5) / scale + imageSize.height * 0.5;
	float rr = ((cellx + dx + r) * stride - netSize.width * 0.5) / scale + imageSize.width * 0.5;
	float bb = ((celly + dy + b) * stride - netSize.height * 0.5) / scale + imageSize.height * 0.5;
	return cv::Rect(cv::Point(xx, yy), cv::Point(rr + 1, bb + 1));
}

cv::Rect FairMOTDetector::restoreCenterNetBox(float dx, float dy, float dw, float dh, float cellx, float celly, int stride, cv::Size netSize, cv::Size imageSize) 
{
	float scale = 0;
	if (imageSize.width >= imageSize.height)
		scale = netSize.width / (float)imageSize.width;
	else
		scale = netSize.height / (float)imageSize.height;
	
	float x = ((cellx + dx - dw * 0.5) * stride - netSize.width * 0.5) / scale + imageSize.width * 0.5;
	float y = ((celly + dy - dh * 0.5) * stride - netSize.height * 0.5) / scale + imageSize.height * 0.5;
	float r = ((cellx + dx + dw * 0.5) * stride - netSize.width * 0.5) / scale + imageSize.width * 0.5;
	float b = ((celly + dy + dh * 0.5) * stride - netSize.height * 0.5) / scale + imageSize.height * 0.5;
	return cv::Rect(cv::Point(x, y), cv::Point(r + 1, b + 1));
}


void FairMOTDetector::preprocessCenterNetImageToTensorMOT(const cv::Mat& image, int numIndex, const std::shared_ptr<TRTInfer::Tensor>& tensor) 
{

	float scale = 0;
	int outH = tensor->height();
	int outW = tensor->width();
	if (image.cols >= image.rows)
		scale = outW / (float)image.cols;
	else
		scale = outH / (float)image.rows;

	cv::Mat matrix = getRotationMatrix2D(cv::Point2f(image.cols*0.5, image.rows*0.5), 0, scale);
	matrix.at<double>(0, 2) -= image.cols*0.5 - outW * 0.5;
	matrix.at<double>(1, 2) -= image.rows*0.5 - outH * 0.5;

	cv::Mat outImage;
	cv::warpAffine(image, outImage, matrix, cv::Size(outW, outH));
	std::vector<cv::Mat> ms(image.channels());
	for (int i = 0; i < ms.size(); ++i)
		ms[i] = cv::Mat(tensor->height(), tensor->width(), CV_32F, tensor->cpu<float>(numIndex, i));

	outImage.convertTo(outImage, CV_32F, 1 / 255.0);

	split(outImage, ms);
}


void FairMOTDetector::get_detection(cv::Mat& frame, std::vector<DetectionBox>& vec_db, std::vector<cv::Mat>& vec_features)
{
	int im_w = frame.cols;
	int im_h = frame.rows;
	vec_db.clear();
	vec_features.clear();
	frame_idx_++;
	auto pre_start = std::chrono::steady_clock::now();
	preprocessCenterNetImageToTensorMOT(frame, 0, engine->input());
	auto pre_end = std::chrono::steady_clock::now();
	auto pre_time = std::chrono::duration_cast<std::chrono::milliseconds>(pre_end - pre_start).count();
	
	auto forward_start = std::chrono::steady_clock::now();
	engine->forward();
	auto forward_end = std::chrono::steady_clock::now();
	auto forward_time = std::chrono::duration_cast<std::chrono::milliseconds>(forward_end - forward_start).count();


	auto post_start = std::chrono::steady_clock::now();
	
	auto outHM = engine->tensor("hm");
	auto outHMPool = engine->tensor("hm_pool");
	auto outWH = engine->tensor("wh");
	auto outXY = engine->tensor("reg");
	auto outID = engine->tensor("id");
	const int stride = 4;
	float*dataoutHM = (float*)outHM->cpu();
	float*dataoutHMPool = (float*)outHMPool->cpu();
	float*dataoutWH = (float*)outWH->cpu();
	float*dataoutXY = (float*)outXY->cpu();
	float*dataoutID = (float*)outID->cpu();
	Size inputSize = engine->input()->size();
	float correct_data[128];
	int hm_width = outWH->width();
	int outsize = outWH->height()*outWH->width();
	int x, y;
	for (int idx = 0; idx < outsize; idx++)
	{
		if (*dataoutHM == *dataoutHMPool && *dataoutHMPool > 0.4f)
		{
			x = idx % hm_width;
			y = idx / hm_width;
			
			float dx = *(dataoutXY + idx); 
			float dy = *(dataoutXY + outsize + idx); 
			float l = *(dataoutWH + idx); 
			float t = *(dataoutWH + outsize + idx);
			float r = *(dataoutWH + outsize*2 + idx);
			float b = *(dataoutWH + outsize*3 + idx);

			ccutil::BBox box; 
			box = restoreCenterNetBox(dx, dy, l, t, r, b, x, y, stride, inputSize, frame.size());

			box = box.box() & Rect(0, 0, frame.cols, frame.rows);
			box.score = *dataoutHMPool;

			if (box.area() > 0)
			{
				vec_db.push_back(DetectionBox(frame_idx_,
					cv::Rect_<float>(cv::Point_<float>(box.x, box.y),
						cv::Point_<float>(box.r, box.b)), box.score));
				cv::Mat fea(1, outID->channel(), CV_32FC1, outID->cpu<float>(0, y, x));
				vec_features.emplace_back(fea.clone());
			}
		}
		dataoutHM++; 
		dataoutHMPool++;
	}

	auto post_end = std::chrono::steady_clock::now();
	auto post_time = std::chrono::duration_cast<std::chrono::milliseconds>(post_end - post_start).count();
	auto post_pre_time = std::chrono::duration_cast<std::chrono::milliseconds>(post_end - pre_start).count();

	//std::cout << "pre:" << pre_time << ",forward:" << forward_time << ",post:" << post_time <<",pre-post:"<< post_pre_time << std::endl;
}