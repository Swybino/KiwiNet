from side_script.data_processor import DataProcessor
from side_script.viewer import Viewer
import config
import os
import json
import numpy as np
from side_script.pose_estimator import solve_head_pose, face_orientation
from side_script.pose_solver import PoseEstimator

img_dir_path = "data/videos/171214_1"
max_frame = 9900
eval_file = "data/171214_1/171214_1_eval.json"


class Evaluator:
    def __init__(self, root_dir, video_title):

        self.dp = DataProcessor(root_dir, video_title)
        self.viewer = Viewer
        self.max_frame = max_frame
        self.results = {}
        with open(eval_file, "r") as f:
            self.results = json.load(f)
        self.img_path_list = self.get_img_path_list()

    def label_confidence(self, dump_file_path, overwrite=False):
        for idx in range(0, self.dp.data_length):
            img_path = self.img_path_list[idx]
            if idx % 5 != 0:
                continue
            if str(idx) in self.results and not overwrite:
                continue

            self.results[idx] = {}
            viewer = Viewer(img_path)
            viewer.plt_frame_idx(idx)
            bboxes_list = []

            # viewer.plt_axis(yaw=0, pitch=0, roll=0, tdx=60, tdy=60)
            # viewer.plt_axis(yaw=0, pitch=0, roll=30, tdx=120, tdy=60)
            # viewer.plt_axis(yaw=0, pitch=0, roll=45, tdx=180, tdy=60)
            # viewer.plt_axis(yaw=0, pitch=0, roll=80, tdx=240, tdy=60)
            #
            # viewer.plt_axis(yaw=30, pitch=0, roll=0, tdx=60, tdy=120)
            # viewer.plt_axis(yaw=45, pitch=0, roll=0, tdx=120, tdy=120)
            # viewer.plt_axis(yaw=80, pitch=0, roll=0, tdx=180, tdy=120)
            #
            # viewer.plt_axis(yaw=0, pitch=30, roll=0, tdx=60, tdy=180)
            # viewer.plt_axis(yaw=0, pitch=45, roll=0, tdx=120, tdy=180)
            # viewer.plt_axis(yaw=0, pitch=80, roll=0, tdx=180, tdy=180)
            names_list = []
            for name, data in self.dp.get_item(idx).items():
                names_list.append(name)
                bbox = data[config.BBOX_KEY]

                landmarks = np.array(data[config.LANDMARKS_KEY], dtype=np.float32)
                landmarks = np.transpose(landmarks)

                ### POSE ESTMIATION 1
                # _, _, pose, _ = face_orientation(viewer.img, landmarks)
                # viewer.plt_axis(pose[2], pose[1], pose[0], bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)

                ### POSE ESTMIATION 2
                # roll, pitch, yaw = solve_head_pose(landmarks)
                # print(roll, pitch, yaw)
                # viewer.plt_axis(yaw, pitch, roll, bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)

                ### POSE ESTMIATION 3
                # pose_estimator = PoseEstimator()
                # pose = pose_estimator.solve_pose(landmarks)
                # print(pose)

                ###
                pose = data[config.POSE_KEY]
                viewer.plt_axis(pose[0], pose[1], pose[2], bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
                bbox_color = (128, 128, 128)
                viewer.plt_bbox(bbox, color=bbox_color)
                color = (0, 255, 0) if data[config.CONFIDENCE_KEY] == 1 else (0, 0, 255)
                viewer.plt_landmarks(data[config.LANDMARKS_KEY], color)
                bboxes_list.append(bbox)

            # eval = viewer.show()

            eval = viewer.landmarks_evaluation(bboxes=bboxes_list)
            for i, a in enumerate(eval):
                self.results[idx][names_list[i]] = a
            self.save_results(dump_file_path)
        return

    def get_img_path_list(self):
        img_names = sorted(os.listdir(img_dir_path))
        img_list = [os.path.join(img_dir_path, img_name) for img_name in
                    img_names[:min(self.max_frame, len(img_names))]]
        return img_list

    def save_results(self, file_path):
        with open(file_path, 'w') as outfile:
            json.dump(self.results, outfile)
        print("file written")

    def compare_results(self):
        total = 0
        condition_positive = 0
        condition_negative = 0
        right_count = 0
        true_positive = 0
        true_negative = 0

        for idx, data in self.results.items():
            for name, conf in data.items():
                total = total + 1
                # print(self.data[name][config.CONFIDENCE_KEY])
                if conf == 1:
                    condition_positive = condition_positive + 1
                elif conf == 0:
                    condition_negative = condition_negative + 1
                if self.dp.get_item(int(idx), name, config.CONFIDENCE_KEY) == conf:
                    right_count = right_count + 1
                    if conf == 1:
                        true_positive = true_positive + 1
                    elif conf == 0:
                        true_negative = true_negative + 1
        true_positive_rate = true_positive / condition_positive
        true_negative_rate = true_negative / condition_negative
        results = {"true_positive_rate": true_positive_rate,
                   "true_negative_rate": true_negative_rate,
                   "false_negative_rate": 1 - true_positive_rate,
                   "false_positive_rate": 1 - true_negative_rate,
                   "positive/negative_ratio": condition_positive / condition_negative,
                   "condition_positive": condition_positive,
                   "condition_negative": condition_negative,
                   "precision": right_count / total}
        for a, b in results.items():
            results[a] = round(b, 4)
        return results


if __name__ == "__main__":
    # evaluator = Evaluator("data/171214_1/correction_size", "171214_1")
    # print(evaluator.compare_results())

    # evaluator = Evaluator("data/171214_1/correction_angle_100_45", "171214_1")
    # print(evaluator.compare_results())
    # evaluator = Evaluator("data/171214_1/correction_angle_100_50", "171214_1")
    # print(evaluator.compare_results())
    # evaluator = Evaluator("data/171214_1/correction_angle_100_55", "171214_1")
    # print(evaluator.compare_results())
    # evaluator = Evaluator("data/171214_1/correction_angle_90_50", "171214_1")
    # print(evaluator.compare_results())
    # evaluator = Evaluator("data/171214_1/correction_angle_80_50", "171214_1")
    # print(evaluator.compare_results())

    evaluator = Evaluator("data/171214_1/correction_angle", "171214_1")
    print(evaluator.compare_results())
    evaluator.label_confidence(eval_file, overwrite=False)
