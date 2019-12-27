import json
import math
import config
import utils
import os
from side_script.pose_estimator import solve_head_pose
import numpy as np
from side_script.range_processor import RangeProcessor

class DataProcessor:
    def __init__(self, root_dir, video_title):
        self.root_dir = root_dir
        self.video_title = video_title
        self.data_length = 9900
        self.frame_idx = None
        self.frame_data = None
        self.max_yaw = 90
        self.max_pitch = 50

    def max_len(self):
        return len(os.listdir(self.root_dir))  # dir is your directory path


    def set_max_angles(self, yaw, pitch):
        self.max_yaw = yaw
        self.max_pitch = pitch

    def get_item(self, frame_idx, name=None, key=None):
        if frame_idx != self.frame_idx:
            self.frame_idx = frame_idx
            self.get_frame_data()
        if name is None:
            return self.frame_data
        else:
            if key is None:
                return self.frame_data[name]
            else:
                return self.frame_data[name][key]

    def get_original_frame_data(self, bbox_data):
        assert type(self.frame_idx) is int, "frame index is not Integer"
        path = os.path.join(self.root_dir, "%s_%d.txt" % (self.video_title, self.frame_idx))
        self.frame_data = self.import_data(path)
        for name, data in self.frame_data.items():
            data[config.BBOX_KEY] = bbox_data[name][config.BBOX_KEY][self.frame_idx]

    def get_frame_data(self):
        assert type(self.frame_idx) is int, "frame index is not Integer"
        path = os.path.join(self.root_dir, "%s_%d.json" % (self.video_title, self.frame_idx))
        self.frame_data = self.import_data(path)

    def import_data(self, file_path):
        if file_path[-4:] == "json":
            with open(file_path, 'r') as f:
                data = json.load(f)
        else:
            with open(file_path, "r") as f:
                data = eval(f.read())
        return data

    def write_data(self, dump_file, data=None):
        if dump_file[-4:] == "json":
            with open(dump_file, 'w') as outfile:
                json.dump(data, outfile)
            print("file written")

    def write_frame_data(self, out_dir=None):
        """
        Writes the frame data as json file in the output directory (root_dir if None)
        :param out_dir: target directory
        :return:
        """
        if out_dir is None:
            out_dir = self.root_dir
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        path = os.path.join(out_dir, "%s_%d.json" % (self.video_title, self.frame_idx))
        with open(path, 'w') as outfile:
            json.dump(self.frame_data, outfile)
        print("file written", path)

    def do_all(self, function, out_dir=None, start=0, end=None):
        if end is None:
            end = self.data_length
        if end < 0:
            end = self.data_length + end
        for idx in range(start, end):
            self.get_item(idx)
            function(out_dir)

    def compare_sizes(self, out_dir=None):
        for name, data in self.frame_data.items():
            bbox = data[config.BBOX_KEY]
            landmarks = data[config.LANDMARKS_KEY]
            if not utils.bbox_landmarks_match(bbox, landmarks):
                data[config.CONFIDENCE_KEY] = 0
        self.write_frame_data(out_dir)

    def detect_high_angles_all(self, out_dir=None):
        for idx in range(self.data_length):
            self.get_item(idx)
            self.detect_high_angles(out_dir)

    def detect_high_angles(self, out_dir=None):
        for name, kid_data in self.frame_data.items():
            if not -self.max_yaw < kid_data[config.POSE_KEY][0] < self.max_yaw \
                    or not -self.max_pitch < kid_data[config.POSE_KEY][1] < self.max_pitch:
                kid_data[config.CONFIDENCE_KEY] = 0
        self.write_frame_data(out_dir)

    def remove_high_gradient(self):
        frame_idx = self.frame_idx
        data_previous = self.get_item(self.frame_idx-1)
        self.get_frame_data()
        return

    def fill_in_gradient(self, out_dir=None):
        frame_idx = self.frame_idx
        data_previous = self.get_item(frame_idx - 1)
        data_next = self.get_item(frame_idx + 1)
        self.get_item(frame_idx)
        for name, kid_data in self.frame_data.items():
            if data_next[name][config.CONFIDENCE_KEY] == 1 \
                    and data_previous[name][config.CONFIDENCE_KEY] == 1 \
                    and kid_data[config.CONFIDENCE_KEY] == 0:
                gradient = np.array(data_next[name][config.POSE_KEY]) \
                           - np.array(data_previous[name][config.POSE_KEY]) / 2
                if np.linalg.norm(gradient) < 20:
                    print("aaaa")

    def get_pose_from_landmarks(self, out_dir=None):
        for name, kid_data in self.frame_data.items():
            roll, pitch, yaw = solve_head_pose(kid_data[config.LANDMARKS_KEY])
            kid_data[config.POSE_KEY] = [yaw, pitch, roll]
        self.write_frame_data(out_dir)

    def percentage_blank(self):
        """
        Count the percentage of detection with a confidence of 0
        :return:
        """
        total_count = 0
        blank_count = 0
        for idx in range(self.data_length):
            for name, kid_data in self.get_item(idx).items():
                total_count = total_count + 1
                if kid_data[config.CONFIDENCE_KEY] == 0:
                    blank_count = blank_count + 1
        return round(blank_count / total_count, 2)

    def data_initialization(self):
        bbox_data = self.import_data(os.path.join(self.root_dir, "%s.txt" % self.video_title))
        for idx in range(self.data_length):
            self.frame_idx = idx
            self.get_original_frame_data(bbox_data)
            self.remove_lists()
            self.radian_to_degrees()
            self.write_frame_data()

    def remove_lists(self):
        for name, data in self.frame_data.items():
            if type(data[config.CONFIDENCE_KEY]) != list:
                continue
            else:
                data[config.CONFIDENCE_KEY] = data[config.CONFIDENCE_KEY][0]
                data[config.POSE_KEY] = data[config.POSE_KEY][0]
                data[config.LANDMARKS_KEY] = data[config.LANDMARKS_KEY][0]
                data[config.ROTATION_MATRIX_KEY] = data[config.ROTATION_MATRIX_KEY][0]

    def radian_to_degrees(self):
        for name, kid_data in self.frame_data.items():
            kid_data[config.POSE_KEY] = [round(a * 180 / math.pi, 1) for a in
                                         kid_data[config.POSE_KEY]]

    def make_data_range(self, range_size=5):
        frame_idx = self.frame_idx
        data = [self.get_item(frame_idx)]
        for i in range(range_size):
            data.insert(0, self.get_item(frame_idx - i))
            data.append(self.get_item(frame_idx + i))
        RangeProcessor(data)
        self.get_item(frame_idx)


if __name__ == "__main__":

    dp = DataProcessor("data/171214_1/correction_angle_100_55", "171214_1")
    dp.do_all(dp.fill_in_gradient, "", start=1, end=9899)
    print("OK")
    # print(dp.percentage_blank())

    # dp = DataProcessor("data/171214_1/correction_size", "171214_1")
    # dp.do_all(dp.detect_high_angles, "data/171214_1/correction_angle")

    # dp = DataProcessor("data/171214_1/correction_size", "171214_1")
    # dp.set_max_angles(100, 45)
    # dp.do_all(dp.detect_high_angles, "data/171214_1/correction_angle_100_45")
    # dp.set_max_angles(100, 50)
    # dp.do_all(dp.detect_high_angles, "data/171214_1/correction_angle_100_50")
    # dp.set_max_angles(100, 55)
    # dp.do_all(dp.detect_high_angles, "data/171214_1/correction_angle_100_55")
    # dp.set_max_angles(90, 50)
    # dp.do_all(dp.detect_high_angles, "data/171214_1/correction_angle_90_50")
    # dp.set_max_angles(80, 50)
    # dp.do_all(dp.detect_high_angles, "data/171214_1/correction_angle_80_50")

    # print(dp.percentage_blank())

    # dp.data_initialization()
    # dp.radian_to_degrees()

    # dp.detect_high_angles()
    # dp.write_data("data/171214_1.json")
