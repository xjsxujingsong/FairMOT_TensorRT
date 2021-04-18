#pragma once
#include <deque>
#include <array>
#include <memory>
#include <vector>
#include "opencv2/opencv.hpp"
#include "config.h"
#include "kalmanfilter.h"
#include "detection.h"
#include "dataType.h"

enum struct TrackState :unsigned char { New, Tracked, Lost, Removed};

class BaseTrack
{
public:
	static int _count;
	int track_id = 0;
	bool is_activated = false;
	TrackState state = TrackState::New;
	std::deque<cv::Mat> features;
	cv::Mat curr_feat;
	float score = 0;
	int start_frame = 0;
	int frame_id = 0;
	int time_since_update = 0;
public:
	int end_frame() {
		return frame_id;
	}
	static int next_id()
	{
		_count += 1;
		return _count;
	}
	void mark_lost()
	{
		state = TrackState::Lost;
	}
	void mark_removed()
	{
		state = TrackState::Removed;
	}
};

class STrack : public BaseTrack
{
public:
	STrack() = delete;
	explicit STrack(cv::Rect_<float>& tlwh, float score, cv::Mat temp_feat, int buffer_size);
	virtual ~STrack();
	STrack(const STrack&) = delete;
	STrack& operator=(const STrack&) = delete;
public:
	//TrackerStateType get_current_status();
	void predict();
	void activate(std::shared_ptr<KalmanFilterTracking> kf, int frame_id);
	void re_activate(std::shared_ptr<STrack> new_track, int frame_id, bool new_id = false);
	void update(std::shared_ptr<STrack> new_track, int frame_id, bool update_feature = true);
	DETECTBOX to_tlwh_box();
	cv::Rect_<float> to_tlwh_rect();
	DETECTBOX tlwh_to_xyah(const cv::Rect_<float>& tlwh);
private:
	void update_features(cv::Mat &feat);
private:
	std::shared_ptr<KalmanFilterTracking> kf;
	int tracklet_len = 0;
	float alpha = 0.9f;
	cv::Rect_<float> _tlwh;
public:
	KAL_MEAN mean;
	KAL_COVA covariance;
	bool is_activated = false;
	cv::Mat smooth_feat;

};

class JDETracker
{
public:
	JDETracker() = delete;
	explicit JDETracker(JDETrackerConfig &config, int frame_rate = 30);
	virtual ~JDETracker();
	JDETracker(const JDETracker&) = delete;
	JDETracker& operator=(const JDETracker&) = delete;
public:
	std::vector<std::shared_ptr<STrack>> update(std::vector<DetectionBox>& vec_db, std::vector<cv::Mat>& vec_features);
private:

	std::tuple<std::vector<std::shared_ptr<STrack>>, std::vector<std::shared_ptr<STrack>>> remove_duplicate_stracks(std::vector<std::shared_ptr<STrack>>& stracksa, std::vector<std::shared_ptr<STrack>>& stracksb);
	std::vector<std::shared_ptr<STrack>> sub_stracks(std::vector<std::shared_ptr<STrack>>& tlista, std::vector<std::shared_ptr<STrack>>& tlistb);
	std::vector<std::shared_ptr<STrack>> joint_stracks(std::vector<std::shared_ptr<STrack>>& tlista, std::vector<std::shared_ptr<STrack>>& tlistb);
private:
	JDETrackerConfig& opt;

	std::unique_ptr<Detection> ptr_detection_;
	std::vector<std::shared_ptr<STrack>> tracked_stracks;
	std::vector<std::shared_ptr<STrack>> lost_stracks;
	std::vector<std::shared_ptr<STrack>> removed_stracks;

	int frame_id = 0;
	float det_thresh;
	int buffer_size;
	int max_time_lost;
	int max_per_image;
	std::shared_ptr<KalmanFilterTracking> kalman_filter;
};