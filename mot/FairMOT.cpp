// FairMOT.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>

#include "tracker.h"
void draw_tracks(cv::Mat&img, std::vector<std::shared_ptr<STrack>>& tracks)
{
	for (auto& tr:tracks)
	{
		cv::Rect_<float> loc = tr->to_tlwh_rect();
		int color_idx = (tr->track_id) % kColorNum * 4;
		cv::Scalar_<int> color = cv::Scalar_<int>(kColorArray[color_idx], kColorArray[color_idx + 1], kColorArray[color_idx + 2], kColorArray[color_idx + 3]);
		cv::rectangle(img, loc, color, 3, cv::LINE_AA, 0);

		int fontCalibration = cv::FONT_HERSHEY_COMPLEX;
		float fontScale = 1; //0.6; //1.2f;
		int fontThickness = 2; //1; // 2;
		char text[15];
		//sprintf(text, "%d:%.2f", tr->track_id, traj.back().detection_score);
		sprintf(text, "%d", tr->track_id);
		std::string buff = text;
		putText(img, buff, cv::Point(loc.x, loc.y), fontCalibration, fontScale, color, fontThickness);/**/

	}
}
int main()
{
	DetectorConfig detconfig;
// 	detconfig.method = DetectorMethod::FromFile;
// 	detconfig.fd.det_list_name = "list.txt";
	detconfig.method = DetectorMethod::FromFairMOT;
	detconfig.fairmot.threshold = 0.4f;
	detconfig.fairmot.ltrb = true;
	detconfig.fairmot.model_file = "models/fairmot_dla34_switch_id_dim_1088x608";//true
	Detection* det = DetectorFactory::create_object(detconfig);
	det->init(); 

	JDETrackerConfig config;
	config.conf_thres = 0.4f;
	config.K = 500;
	config.track_buffer = 30;
	int frame_rate = 30;
	JDETracker jde(config, frame_rate);
	cv::VideoCapture capture;
	capture.open("MOT16-03.mp4");
	cv::Mat frame;
	std::vector<DetectionBox> vec_db;
	std::vector<cv::Mat> vec_features;
	int frame_index = 0;
	std::string window_name = "tracking";
	cv::namedWindow(window_name, cv::WINDOW_NORMAL);
	cv::VideoWriter filesavewriter;
	while (true)
	{
 		capture >> frame;
		if (frame.empty()) break;
// 		if (frame_index == 0)
// 		{
// 			filesavewriter.open("result.mp4", cv::VideoWriter::fourcc('X', 'V', 'I', 'D')
// 				, frame_rate, cv::Size(frame.cols, frame.rows), true);
// 		}
		auto detect_start = std::chrono::steady_clock::now();
		det->get_detection(frame, vec_db, vec_features); 
		auto detect_end = std::chrono::steady_clock::now();
		auto detection_time = std::chrono::duration_cast<std::chrono::milliseconds>(detect_end - detect_start).count();

		
		auto track_start = std::chrono::steady_clock::now();
		std::vector<std::shared_ptr<STrack>> tracks = jde.update(vec_db, vec_features);
		auto track_end = std::chrono::steady_clock::now(); 
		auto tracking_time = std::chrono::duration_cast<std::chrono::milliseconds>(track_end - track_start).count();
		float inference_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(track_end - detect_start).count();
		std::cout << "detection:" << detection_time << ", tracking:" << tracking_time << ", fps:" << inference_fps << std::endl;
		int a = 10;
		draw_tracks(frame, tracks);
		//char tmp[100] = { '\0'};
		//sprintf(tmp, "tmp/%03d.jpg", frame_index);
		//std::string save_name = tmp;
		//if (frame_index>=120)
		//cv::imwrite(save_name, frame);
		cv::imshow(window_name, frame);
		char c = (char)cv::waitKey(1);
		if (c == 27)
			break;
		//filesavewriter.write(frame);
		frame_index++;
		//std::cout << "frame_index:" << frame_index << std::endl;
	}
	//filesavewriter.release();
	return 1;
}
