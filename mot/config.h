#pragma once
#include "opencv2/opencv.hpp"
struct STrackConfig
{
	float tlwh[4];
	float score;
	cv::Mat temp_feat;
	int buffer_size;
};

// Table for the 0.95 quantile of the chi - square distribution with N degrees of
// freedom(contains values for N = 1, ..., 9).Taken from MATLAB / Octave's chi2inv
// function and used as Mahalanobis gating threshold.

const std::map<int, float>chi2inv95 = {
	{1, 3.8415},
    {2 , 5.9915},
	{3 , 7.8147},
	{4 , 9.4877},
	{5 , 11.070},
	{6 , 12.592},
	{7 , 14.067},
	{8 , 15.507},
	{9 , 16.919} };

typedef struct DetectionBox
{
	int frame;
	cv::Rect_<float> box;
	float score;
	cv::Mat mask;
	std::vector<int> points;
	DetectionBox()
	{
		frame = -1; score = 0; box = cv::Rect_<float>(cv::Point_<float>(-1, -1), cv::Point_<float>(-1, -1));
	}
	DetectionBox(int frame_, const cv::Rect_<float>& box_, float score_)
	{
		frame = frame_; box = box_; score = score_;
	}
	void add_mask(const cv::Mat& mask_)
	{
		mask = mask_;
	}
	void add_points(const std::vector<int>& vec_points)
	{
		points = vec_points;
	}
}DetectionBox;

struct FileDetectorConfig
{
	//std::string image_list_name;
	std::string det_list_name;
	std::string det_file_name;
	float threshold;
	int start_x;
	int start_y;
	int min_size;
	int max_size;
};
struct FairMOTDetectorConfig
{
	std::string model_file;
	float threshold;
	int min_size;
	int max_size;
	bool ltrb;
};

enum struct DetectorMethod :unsigned char {
	FromYOLOV3DLL, FromOpenVINO, FromFile, FromMaskFile, FromPointFile, FromYOLOV3PointDLL
	, FromYOLOV3PointNCNN, FromYOLOV3PointTensorRT, FromThunderCentR, FromYOLOV5PointTensorRT
	, FromRCenterNetTensorRT, FromYolactTensorRT, FromCentPtTensorRT, FromFairMOT
};
struct DetectorConfig
{
	DetectorMethod method;
	FileDetectorConfig fd;
	FairMOTDetectorConfig fairmot;
};

struct JDETrackerConfig
{
	DetectorConfig det_config;
	float conf_thres;
	int K;
	int track_buffer;
};

const int kColorNum = 40;
const int kColorArray[kColorNum * 4] =
{ 246, 156, 192, 7,
165, 166, 2, 179,
25, 147, 24, 67,
31, 132, 123, 250,
111, 208, 249, 149,
234, 37, 55, 147,
143, 29, 214, 169,
215, 84, 190, 204,
110, 239, 216, 103,
221, 142, 83, 166,//10
251, 222, 243, 67,
115, 91, 244, 128,
151, 254, 47, 13,
132, 253, 137, 127,
236, 246, 66, 169,
131, 63, 5, 237,
28, 12, 58, 99,
6, 49, 196, 195,
163, 9, 82, 197,
157, 103, 213, 44,//20
227, 86, 106, 79,
30, 72, 46, 152,
204, 9, 223, 80,
25, 202, 70, 6,
141, 195, 106, 193,
166, 178, 228, 113,
105, 208, 175, 243,
193, 105, 28, 96,
251, 108, 207, 168,
110, 165, 55, 38, //30
58, 190, 234, 236,
48, 225, 141, 239,
251, 163, 161, 73,
140, 59, 204, 205,
118, 19, 202, 34,
81, 104, 215, 42,
109, 175, 221, 223,
64, 0, 104, 10,
156, 242, 254, 136,
245, 33, 67, 208
//,86, 249, 228, 80, 89, 153, 124, 43, 171, 65, 132, 37, 185, 42, 208, 151, 231, 159, 143, 78, 95, 212, 130, 81, 91, 159, 204, 94, 5, 87, 14, 132, 154, 8, 95, 82, 239, 122, 254, 197, 236, 175, 44, 62, 137, 176, 218, 96, 212, 143, 209, 192, 150, 112, 230, 181, 92, 41, 111, 248, 190, 120, 94, 175, 245, 100, 13, 26, 163, 161, 234, 12, 140, 198, 12, 6, 203, 100, 66, 182, 193, 80, 101, 71, 8, 61, 108, 128, 25, 78, 8, 234, 107, 95, 54, 71, 108, 25, 77, 213, 150, 16, 122, 76, 251, 40, 240, 100, 128, 218, 40, 89, 218, 174, 121, 111, 188, 201, 122, 247
};
