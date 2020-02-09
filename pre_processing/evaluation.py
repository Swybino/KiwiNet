from data_processor import DataProcessor
from utils.viewer import Viewer
import config
import os
import json
import numpy as np
import utils.utils as utils

img_dir_path = "data/videos/171214_1"
max_frame = 9900
eval_file = "data/171214_1_eval.json"


class Evaluator:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        # self.dp = DataProcessor(root_dir)
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
        video = "171214_1"
        for idx, result in self.results.items():
            file = "%s_%s.json" % (video, idx)

            data = utils.read_input(os.path.join(self.root_dir, file))
            for name, conf in result.items():
                total = total + 1
                # print(self.result[name][config.CONFIDENCE_KEY])
                if conf == 1:
                    condition_positive += 1
                elif conf == 0:
                    condition_negative += 1
                if data[name][config.CONFIDENCE_KEY] == conf:
                    right_count += 1
                    if conf == 1:
                        true_positive += 1
                    elif conf == 0:
                        true_negative += 1
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

    evaluator = Evaluator("data/inputs_roll_delete_100")
    print(evaluator.compare_results())
    # evaluator.label_confidence(eval_file, overwrite=False)
