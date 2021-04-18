#include "tracker.h"
#include "matching.h"

int BaseTrack::_count = 0;

STrack::STrack(cv::Rect_<float>& tlwh, float score_, cv::Mat temp_feat, int buffer_size)
	:_tlwh(tlwh)
{
	mean.setZero();
	covariance.setZero();
	score = score_;
	float*data1 = (float*)temp_feat.data;
	update_features(temp_feat);
}

STrack::~STrack()
{
}

void STrack::update_features(cv::Mat &feat)
{
	cv::Mat norm_feat;
	cv::normalize(feat, norm_feat, 1.0, 0, cv::NORM_L2);
	feat = norm_feat;
	curr_feat = feat.clone();
	if (smooth_feat.empty()) smooth_feat = feat;
	else
	{
		smooth_feat = alpha * smooth_feat + (1 - alpha)*feat;
		features.push_back(feat);
		cv::normalize(smooth_feat, norm_feat, 1.0, 0, cv::NORM_L2);
		smooth_feat = norm_feat;
	}
}
// check if tracked, cauthy---a little different, TBD
void STrack::predict()
{
	if (this->state != TrackState::Tracked)
	{
		this->mean(7) = 0;
	}
	this->kf->predict(this->mean, this->covariance);
}

// Convert bounding box to format `(center x, center y, aspect ratio,
// height)`, where the aspect ratio is `width / height`.
DETECTBOX STrack::tlwh_to_xyah(const cv::Rect_<float>& tlwh)
{
	DETECTBOX box;
	float x = tlwh.x + tlwh.width / 2;
	float y = tlwh.y + tlwh.height / 2;
	box << x, y, tlwh.width / tlwh.height, tlwh.height;
	return box;
}

//Start a new tracklet
//void STrack::activate(KalmanFilter*kf, int frame_id)
void STrack::activate(std::shared_ptr<KalmanFilterTracking> kf, int frame_id)
{
	this->kf = kf;
	this->track_id = next_id();
	auto ret = this->kf->initiate(tlwh_to_xyah(this->_tlwh));
	this->mean = ret.first;
	this->covariance = ret.second;
	tracklet_len = 0;
	state = TrackState::Tracked;
	if (frame_id == 1)
	{
		is_activated = true;
	}
	this->frame_id = frame_id;
	start_frame = frame_id;
	
}

void STrack::re_activate(std::shared_ptr<STrack> new_track, int frame_id, bool new_id)
{
	auto ret = this->kf->update(this->mean, this->covariance, tlwh_to_xyah(new_track->to_tlwh_rect()));
	this->mean = ret.first;
	this->covariance = ret.second;
	update_features(new_track->curr_feat);
	tracklet_len = 0;
	state = TrackState::Tracked;
	is_activated = true;
	this->frame_id = frame_id;
	if (new_id)
	{
		track_id = next_id();
	}
}


DETECTBOX STrack::to_tlwh_box()
{
	if (this->mean.isZero())
	{
		DETECTBOX box;
		box << _tlwh.x, _tlwh.y, _tlwh.width, _tlwh.height;
		return box;
	}

	DETECTBOX ret = this->mean.leftCols(4);
	ret(2) *= ret(3);
	ret.leftCols(2) -= (ret.rightCols(2) / 2);
	return ret;
}

cv::Rect_<float> STrack::to_tlwh_rect()
{
	DETECTBOX ret = to_tlwh_box();
	return cv::Rect_<float>(cv::Point_<float>(ret[0], ret[1]), cv::Point_<float>(ret[0]+ret[2], ret[1] +ret[3]));
}


void STrack::update(std::shared_ptr<STrack> new_track, int frame_id, bool update_feature)
{
	this->frame_id = frame_id;
	this->tracklet_len += 1;

	auto ret = this->kf->update(this->mean, this->covariance, tlwh_to_xyah(new_track->to_tlwh_rect()));
	this->mean = ret.first;
	this->covariance = ret.second;

	state = TrackState::Tracked;
	is_activated = true;
	score = new_track->score;
	if (update_feature)
	{
		update_features(new_track->curr_feat);
	}
	
}



JDETracker::JDETracker(JDETrackerConfig &config, int frame_rate)
	:opt(config)
{

	det_thresh = opt.conf_thres;
	buffer_size = int(frame_rate / 30.0 * opt.track_buffer);
	max_time_lost = buffer_size;
	max_per_image = opt.K;
	kalman_filter = std::shared_ptr<KalmanFilterTracking>(new KalmanFilterTracking());
	ptr_detection_ = std::unique_ptr<Detection>(DetectorFactory::create_object(config.det_config));
	
}

JDETracker::~JDETracker()
{
}
std::vector<std::shared_ptr<STrack>> JDETracker::update(std::vector<DetectionBox>& dets, std::vector<cv::Mat>& id_feature)
{
	frame_id += 1;
	if (frame_id == 878)
	{
		int a = 10;
	}

	std::vector<std::shared_ptr<STrack>> activated_starcks;
	std::vector<std::shared_ptr<STrack>> refind_stracks;
	std::vector<std::shared_ptr<STrack>> lost_stracks;
	std::vector<std::shared_ptr<STrack>> removed_stracks;
	std::vector<std::shared_ptr<STrack>> detections;
	for (int i = 0; i < dets.size(); i++)
	{
		auto&box = dets[i].box;
		detections.push_back(std::shared_ptr<STrack>(new STrack(box, dets[i].score, id_feature[i], 30)));
	}
	//Add newly detected tracklets to tracked_stracks'''
	std::vector<std::shared_ptr<STrack>> unconfirmed;
	std::vector<std::shared_ptr<STrack>> tracked_stracks;
	for (auto& track : this->tracked_stracks)
	{
		if (!track->is_activated)
		{
			unconfirmed.push_back(track);
		}
		else
		{
			tracked_stracks.push_back(track);
		}
	}
	// Step 2: First association, with embedding'''
	auto strack_pool = joint_stracks(tracked_stracks, this->lost_stracks);
	//# Predict the current location with KF
	for (auto& strack : strack_pool)
		strack->predict();
	//#for strack in strack_pool :
	//STrack.multi_predict(strack_pool)
	auto dists = matching::embedding_distance(strack_pool, detections);
	matching::fuse_motion(this->kalman_filter, dists, strack_pool, detections);
	auto[matches, u_track, u_detection] = matching::linear_assignment(dists, 0.4f, strack_pool.size(), detections.size());

	for (auto pt : matches)
	{
		auto& track = strack_pool[pt.x];
		auto& det = detections[pt.y];
		if (track->state == TrackState::Tracked)
		{
			track->update(det, this->frame_id);
			activated_starcks.push_back(track);
		}
		else
		{
			track->re_activate(det, this->frame_id, false);
			refind_stracks.push_back(track);
		}
	}
	//''' Step 3: Second association, with IOU'''
	std::vector<std::shared_ptr<STrack>> detections_tmp;
	for (auto& ud : u_detection) detections_tmp.push_back(detections[ud]);
	detections = detections_tmp;

	std::vector<std::shared_ptr<STrack>> r_tracked_stracks;
	for (auto& ut : u_track)
	{
		if (strack_pool[ut]->state == TrackState::Tracked)
		{
			r_tracked_stracks.push_back(strack_pool[ut]);
		}
	}
	dists = matching::iou_distance(r_tracked_stracks, detections);
	auto[matches2, u_track2, u_detection2] = matching::linear_assignment(dists, 0.5f, r_tracked_stracks.size(), detections.size());
	for (auto pt : matches2)
	{
		auto& track = r_tracked_stracks[pt.x];
		auto& det = detections[pt.y];
		if (track->state == TrackState::Tracked)
		{
			track->update(det, this->frame_id);
			activated_starcks.push_back(track);
		}
		else
		{
			track->re_activate(det, this->frame_id, false);
			refind_stracks.push_back(track);
		}
	}
	for (auto& it:u_track2)
	{
		auto& track = r_tracked_stracks[it];
		if (track->state != TrackState::Lost)
		{
			track->mark_lost();
			lost_stracks.push_back(track);
		}
	}
	//Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
	detections_tmp.clear();
	for (auto& ud : u_detection2) detections_tmp.push_back(detections[ud]);
	detections = detections_tmp;
	dists = matching::iou_distance(unconfirmed, detections);
	auto[matches3, u_unconfirmed3, u_detection3] = matching::linear_assignment(dists, 0.7f, unconfirmed.size(), detections.size());
	for (auto pt : matches3)
	{
		unconfirmed[pt.x]->update(detections[pt.y], this->frame_id);
		activated_starcks.push_back(unconfirmed[pt.x]);
	}
	for (auto& it : u_unconfirmed3)
	{
		auto& track = unconfirmed[it];
		track->mark_removed();
		removed_stracks.push_back(track);
	}

	//Step 4: Init new stracks"""
	for (auto& inew : u_detection3)
	{
		auto& track = detections[inew];
		if (track->score < this->det_thresh)
		{
			continue;
		}
		track->activate(this->kalman_filter, this->frame_id);
		activated_starcks.push_back(track);
	}
	//Step 5: Update state"""
	for (auto& track:this->lost_stracks)
	{
		if ((this->frame_id - track->end_frame()) > this->max_time_lost)
		{
			track->mark_removed();
			removed_stracks.push_back(track);
		}
	}
	std::vector<std::shared_ptr<STrack>> tracked_stracks_tmp;
	for (auto&t:this->tracked_stracks)
	{
		if (t->state == TrackState::Tracked)
		{
			tracked_stracks_tmp.push_back(t);
		}
	}
	this->tracked_stracks = tracked_stracks_tmp;
	this->tracked_stracks = joint_stracks(this->tracked_stracks, activated_starcks);
	this->tracked_stracks = joint_stracks(this->tracked_stracks, refind_stracks);
	this->lost_stracks = sub_stracks(this->lost_stracks, this->tracked_stracks);
	this->lost_stracks.insert(this->lost_stracks.end(), lost_stracks.begin(), lost_stracks.end());
	this->lost_stracks = sub_stracks(this->lost_stracks, this->removed_stracks);
	this->removed_stracks.insert(this->removed_stracks.end(), removed_stracks.begin(), removed_stracks.end());
	auto[stracksa, stracksb] = remove_duplicate_stracks(this->tracked_stracks, this->lost_stracks);
	this->tracked_stracks = stracksa; 
	this->lost_stracks = stracksb; 
	std::vector<std::shared_ptr<STrack>> output_stracks;
	for (auto& track : this->tracked_stracks)
	{
		if (track->is_activated)
		{
			output_stracks.push_back(track);
		}
	}
	return output_stracks;
}


std::vector<std::shared_ptr<STrack>> JDETracker::joint_stracks(std::vector<std::shared_ptr<STrack>>& tlista, std::vector<std::shared_ptr<STrack>>& tlistb)
{
	std::map<int, int> exists;
	std::vector<std::shared_ptr<STrack>> res;
	for (auto& t: tlista)
	{
		exists[t->track_id] = 1;
		res.push_back(t);
	}
	for (auto& t : tlistb)
	{
		int tid = t->track_id;
		if (exists.find(tid) == exists.end())
		{
			exists[tid] = 1;
			res.push_back(t);
		}
	}
	return res;
}

std::vector<std::shared_ptr<STrack>> JDETracker::sub_stracks(std::vector<std::shared_ptr<STrack>>& tlista, std::vector<std::shared_ptr<STrack>>& tlistb)
{
	std::vector<std::shared_ptr<STrack>> res;
	std::map<int, std::shared_ptr<STrack>> stracks;
	for (auto&t:tlista)
	{
		stracks[t->track_id] = t;
	}
	for (auto&t : tlistb)
	{
		auto tid = t->track_id;
		auto key = stracks.find(tid);
		if (key != stracks.end())
		{
			stracks.erase(key);
		}
	}
	for (auto &v : stracks)
		res.push_back(v.second);
	return res;
}


std::tuple<std::vector<std::shared_ptr<STrack>>, std::vector<std::shared_ptr<STrack>>> JDETracker::remove_duplicate_stracks(std::vector<std::shared_ptr<STrack>>& stracksa, std::vector<std::shared_ptr<STrack>>& stracksb)
{
	auto pdist = matching::iou_distance(stracksa, stracksb);
	std::set<int>dupa;
	std::set<int>dupb;
	for (int p = 0; p < pdist.size(); p++)
	{
		for (int q = 0; q < pdist[p].size(); q++)
		{
			if (pdist[p][q] < 0.15f)
			{
				auto timep = stracksa[p]->frame_id - stracksa[p]->start_frame;
				auto timeq = stracksb[q]->frame_id - stracksb[q]->start_frame;
				if (timep > timeq)
					dupb.insert(q);
				else
					dupa.insert(p);
			}
		}
	}
	std::vector<std::shared_ptr<STrack>>resa, resb;
	for (int i = 0; i < stracksa.size(); i++)
	{
		if (dupa.find(i) == dupa.end())
			resa.push_back(stracksa[i]);
	}
	for (int i = 0; i < stracksb.size(); i++)
	{
		if (dupb.find(i) == dupb.end())
			resb.push_back(stracksb[i]);
	}
	return { resa, resb };
}
