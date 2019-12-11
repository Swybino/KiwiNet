import json
import math
import config
import utils
import os


class DataProcessor:
    def __init__(self, root_dir, video_title):
        self.root_dir = root_dir
        self.video_title = video_title


        self.data_length = 9900
        self.frame_idx = None
        self.frame_data = None

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

    def round(self, angle_ndigits=4, lm_ndigits=1):
        return

    def compare_sizes_all(self, out_dir=None):
        for idx in range(self.data_length):
            self.get_item(idx)
            self.compare_sizes(out_dir)

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
            if not -90 < kid_data[config.POSE_KEY][0] < 90 or not -45 < kid_data[config.POSE_KEY][1] < 45:
                kid_data[config.CONFIDENCE_KEY] = 0
        self.write_frame_data(out_dir)

    def remove_high_gradient(self):
        return

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
        return blank_count / total_count

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


if __name__ == "__main__":
    dp = DataProcessor("data/171214_1/171214_1", "171214_1")

    print(dp.percentage_blank())

    # dp.data_initialization()
    # dp.radian_to_degrees()

    # dp.detect_high_angles()
    # dp.write_data("data/171214_1.json")
