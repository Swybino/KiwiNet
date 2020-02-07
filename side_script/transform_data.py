import argparse
import os
import config
import json
import utils.utils as utils
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kiwi Training')

    parser.add_argument('-d', '--dir', type=str, help='')
    parser.add_argument('-s', '--start', default=0, type=int, help='')
    parser.add_argument('-o', '--out_dir', type=str, help='')
    parser.add_argument('-v', '--video', type=str, help='')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    for file in os.listdir(args.dir):
        data = utils.read_input(os.path.join(args.dir, file))
        frame_data = {}
        for name, item in data.items():
            frame_data[name] = {}
            frame_data[name][config.BBOX_KEY] = (np.around(np.array(data[name][config.BBOX_KEY])/640, 4)).tolist()
            frame_data[name][config.LANDMARKS_KEY] = (np.around(np.array(data[name][config.LANDMARKS_KEY])/640, 4)).tolist()
            frame_data[name][config.POSE_KEY] = (np.around(np.array(data[name][config.POSE_KEY])/180, 4)).tolist()
            frame_data[name][config.CONFIDENCE_KEY] = data[name][config.CONFIDENCE_KEY]

        # print(frame_data)
        path = os.path.join(args.out_dir, file)
        with open(path, 'w') as outfile:
            json.dump(frame_data, outfile)
        print("file written", path)

