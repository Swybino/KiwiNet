import json
import math
import config
from utils import utils
import os
from pose_estimator import solve_head_pose


class DataProcessor:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_name = None
        self.data_length = len(os.listdir(root_dir))
        self.frame_idx = None
        self.frame_data = None
        self.max_yaw = 90
        self.max_pitch = 50
        self.max_roll = 100

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
            if name not in self.frame_data:
                return None
            elif key is None:
                return self.frame_data[name]
            elif key not in self.frame_data[name]:
                return None
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
        if not os.path.exists(path):
            path = os.path.join(self.root_dir, "%s_%d.txt" % (self.video_title, self.frame_idx))
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
        path = os.path.join(out_dir, self.file_name)
        with open(path, 'w') as outfile:
            json.dump(self.frame_data, outfile)
        print("file written", path)

    def do_all(self, functions, out_dir=None, start=0, end=None):
        for f in os.listdir(self.root_dir):
            self.frame_data = utils.read_input(os.path.join(self.root_dir, f))
            self.file_name = f
            if type(functions) == list:
                for function in functions:
                    function()
            self.write_frame_data(out_dir)

    def compare_sizes(self):
        for name, data in self.frame_data.items():
            bbox = data[config.BBOX_KEY]
            landmarks = data[config.LANDMARKS_KEY]
            if not utils.bbox_landmarks_match(bbox, landmarks):
                data[config.CONFIDENCE_KEY] = 0

    def detect_high_angles(self):
        for name, kid_data in self.frame_data.items():
            img_angle = utils.get_angle(kid_data[config.BBOX_KEY]) + 90
            # print(img_angle, "######",kid_data[config.POSE_KEY][2]*180, ((img_angle+kid_data[config.POSE_KEY][2]*180)+180)%360 -180)
            if not -self.max_yaw < kid_data[config.POSE_KEY][0]*180 < self.max_yaw \
                    or not -self.max_pitch < kid_data[config.POSE_KEY][1]*180 < self.max_pitch \
                    or not -self.max_roll < ((img_angle+kid_data[config.POSE_KEY][2]*180)+180)%360 - 180< self.max_roll:
                kid_data[config.CONFIDENCE_KEY] = 0

    def get_pose_from_landmarks(self):
        for name, kid_data in self.frame_data.items():
            roll, pitch, yaw = solve_head_pose(kid_data[config.LANDMARKS_KEY])
            kid_data[config.POSE_KEY] = [yaw/180, pitch/180, roll/180]

    def percentage_blank(self):
        """
        Count the percentage of detection with a confidence of 0
        :return:
        """
        total_count = 0
        blank_count = 0
        for f in os.listdir(self.root_dir):
            self.frame_data = utils.read_input(os.path.join(self.root_dir, f))
            self.file_name = f
            for name, kid_data in self.frame_data.items():
                total_count = total_count + 1
                if kid_data[config.CONFIDENCE_KEY] == 0:
                    blank_count = blank_count + 1
        return round(blank_count / total_count, 2)

    def radian_to_degrees(self):
        for name, kid_data in self.frame_data.items():
            kid_data[config.POSE_KEY] = [round(a * 180 / math.pi, 1) for a in
                                         kid_data[config.POSE_KEY]]


if __name__ == "__main__":
    intake = "data/inputs_new"
    # intake = "data/test"
    out = "data/inputs_roll_delete_100"

    dp = DataProcessor(intake)
    # print(dp.percentage_blank())
    dp.set_max_angles(100, 55)
    dp.do_all([dp.detect_high_angles], out)
    dp = DataProcessor(out)
    print(dp.percentage_blank())
    print("----OK----")

    # print(dp.percentage_blank())
