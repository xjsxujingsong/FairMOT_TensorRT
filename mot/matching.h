#pragma once
#include "opencv2/opencv.hpp"
#include "tracker.h"
#define INFTY_COST 1e5
namespace matching
{
	std::vector<std::vector<double>> embedding_distance(std::vector<std::shared_ptr<STrack>>&tracks, std::vector<std::shared_ptr<STrack>>&detections, std::string metric = "cosine");
	std::vector<std::vector<double>> iou_distance(std::vector<std::shared_ptr<STrack>>&atracks, std::vector<std::shared_ptr<STrack>>&btracks);
	float iou(const cv::Rect_<float> &bb_det, const cv::Rect_<float> &bb_pre);
	std::tuple<std::vector<cv::Point>, std::set<int>, std::set<int>> linear_assignment(std::vector<std::vector<double>>& dists, float thresh, int dim_a, int dim_b);
	void fuse_motion(std::shared_ptr<KalmanFilterTracking>kf
		, std::vector<std::vector<double>>&cost_matrix
		, std::vector<std::shared_ptr<STrack>>&tracks
		, std::vector<std::shared_ptr<STrack>>&detections
		, bool only_position = false
		, float lambda_ = 0.98f);
};