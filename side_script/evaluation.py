from side_script.data_processing import DataProcessor
from side_script.viewer import Viewer
import config
import os
import json
import numpy as np
from side_script.pose_estimator import solve_head_pose, face_orientation
from side_script.pose_solver import PoseEstimator

img_dir_path = "data/videos/171214_1"
max_frame = 5000
eval_file = "data/171214_1/171214_1_eval.json"


class Evaluator:
    def __init__(self, video_title):

        self.dp = DataProcessor(video_title)

        self.viewer = Viewer
        self.max_frame = 5000
        self.results = {}
        with open(eval_file, "r") as f:
            self.results = json.load(f)
        self.img_path_list = self.get_img_path_list()

    def evaluate_accuracy(self, dump_file_path, overwrite=False):
        for idx in range(self.dp.data_length):
            img_path = self.img_path_list[idx]
            if idx % 5 != 0:
                continue
            if str(idx) in self.results and not overwrite:
                continue

            self.results[idx] = {}
            viewer = Viewer(img_path)

            viewer.plt_frame_idx(idx)
            bboxes_list = []

            viewer.plt_axis(yaw=0, pitch=0, roll=0, tdx=60, tdy=60)
            # viewer.plt_axis(yaw=0, pitch=0, roll=30, tdx=120, tdy=60)
            # viewer.plt_axis(yaw=0, pitch=0, roll=45, tdx=180, tdy=60)
            # viewer.plt_axis(yaw=0, pitch=0, roll=80, tdx=240, tdy=60)
            #
            viewer.plt_axis(yaw=30, pitch=0, roll=0, tdx=60, tdy=120)
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
                roll, pitch, yaw = solve_head_pose(landmarks)
                print(roll, pitch, yaw)
                viewer.plt_axis(yaw, pitch, roll, bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)

                ### POSE ESTMIATION 3
                # pose_estimator = PoseEstimator()
                # pose = pose_estimator.solve_pose(landmarks)
                # print(pose)

                color = (0, 255, 0) if data[config.CONFIDENCE_KEY] == 1 else (0, 0, 255)
                viewer.plt_bbox(bbox, color=(128, 128, 128))
                viewer.plt_landmarks(data[config.LANDMARKS_KEY], color)
                # viewer.plt_axis(data[config.POSE_KEY][0], data[config.POSE_KEY][1], data[config.POSE_KEY][2],
                #                 bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
                bboxes_list.append(bbox)

                #

            # eval = viewer.show()

            eval = viewer.evaluation(bboxes=bboxes_list)
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
        positive_total = 0
        negative_total = 0
        right_count = 0
        positive_count = 0
        negative_count = 0

        for idx, data in self.results.items():
            for name, conf in data.items():
                total = total + 1
                # print(self.data[name][config.CONFIDENCE_KEY])
                if conf == 1:
                    positive_total = positive_total + 1
                elif conf == 0:
                    negative_total = negative_total + 1
                if self.dp.get_item(int(idx),name,config.CONFIDENCE_KEY) == conf:
                    right_count = right_count + 1
                    if conf == 1:
                        positive_count = positive_count + 1
                    elif conf == 0:
                        negative_count = negative_count + 1
        results = {"true_positive": positive_count/positive_total,
                   "false positive": 1 - positive_count/positive_total,
                   "true negative": negative_count/negative_total,
                   "false negative": 1 - negative_count/negative_total,
                   "positive/negative_ratio": positive_total / negative_total,
                   "positive_total": positive_total,
                   "negative_total": negative_total,
                   "precision": right_count/total}
        return results


if __name__ == "__main__":
    evaluator = Evaluator("171214_1")
    print(evaluator.compare_results())
    evaluator.evaluate_accuracy(eval_file, overwrite=False)
