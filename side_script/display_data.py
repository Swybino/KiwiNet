import os
from utils.video import Video
import utils.utils as utils
from utils.viewer import Viewer
import config
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kiwi Training')
    parser.add_argument('-b', '--bbox', action='store_true', help='')
    parser.add_argument('-l', '--landmarks', action='store_true', help='')
    parser.add_argument('-p', '--pose', action='store_true', help='')
    parser.add_argument('-a', '--anonymize', action='store_true', help='')
    parser.add_argument('-o', '--out', type=str)
    args = parser.parse_args()

    print(args.landmarks)

    for f in os.listdir(config.inputs_dir):
        data = utils.read_input(os.path.join(config.inputs_dir, f))
        tmp = f[:-4].split("_")
        frame = tmp[-1]
        video = "_".join(tmp)
        img = Video(os.path.join(config.video_root, "%s.MP4" % video))[int(frame)]
        img_size = img.shape[0]
        viewer = Viewer(img)

        for name, item in data.items():
            if args.anonymize:
                bbox = np.array(item[config.BBOX_KEY]) * img_size / 640
                viewer.plt_bbox(item[config.BBOX_KEY], name)

            if args.bbox:
                bbox = np.array(item[config.BBOX_KEY]) * img_size / 640
                viewer.plt_bbox(item[config.BBOX_KEY], name)

            if args.landmarks:
                landmarks = np.array(data[config.LANDMARKS_KEY], dtype=np.float32) * img_size / 640
                landmarks = np.transpose(landmarks)
                viewer.plt_landmarks(landmarks)

            if args.pose:
                bbox = np.array(item[config.BBOX_KEY]) * img_size / 640
                pose = data[config.POSE_KEY]
                viewer.plt_axis(pose[0], pose[1], pose[2], bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
        viewer.save_img(os.path.join(args.out, f))
