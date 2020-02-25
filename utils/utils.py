import os

import cv2
import numpy as np
import math

import pandas as pd
import torch
import config
from utils.video import Video
import json


def read_input(file_path):
    with open(file_path, 'r') as f:
        frame_data = json.load(f)
    return frame_data


def bbox_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[0] + boxA[2], boxB[0] + boxB[2]), min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] + 1) * (boxA[3] + 1)
    boxBArea = (boxB[2] + 1) * (boxB[3] + 1)
    return interArea / float(boxAArea + boxBArea - interArea)


def bbox_landmarks_match(bbox, landmarks):
    x, y = min(landmarks[0]), min(landmarks[1])
    width = max(landmarks[0]) - min(landmarks[0])
    height = max(landmarks[1]) - min(landmarks[1])
    lm_bbox = [x, y, width, height]
    return bbox_iou(bbox, lm_bbox) > 0.35


def get_roi(img, bbox, *, scale=1, padding=0):
    size = max(bbox[2], bbox[3])
    x_c = bbox[0] + bbox[2] / 2
    y_c = bbox[1] + bbox[3] / 2
    new_size = (size * scale) + 2 * padding

    xmin = max(0, x_c - new_size / 2)
    ymin = max(0, y_c - new_size / 2)
    xmax = min(1, x_c + new_size / 2)
    ymax = min(1, y_c + new_size / 2)
    return xmin, ymin, xmax, ymax


def crop_roi(img, bbox, *, scale=1, padding=0):
    xmin, ymin, xmax, ymax = get_roi(img, bbox, scale=scale, padding=padding)
    return img[ymin:ymax, xmin:xmax], [xmin, ymin, xmin - xmax, ymin - ymax]


def gradient(y1, y2, t=1):
    return (y2 - y1) / t


def get_angle(*, bbox=None, x=None, y=None):
    if bbox is not None:
        x = bbox[0] + bbox[2]/2
        y = bbox[1] + bbox[3]/2
    return np.degrees(np.arctan2(0.5 - y, x - 0.5))


def rotate_coord(x, y, angle):
    '''
    Rotate the coordonates of a bounding box
    :param bbox:
    :param img:
    :param angle: 90, 180 or -90
    :return:
    '''
    angle = angle % 360
    if angle == 0:
        return x, y
    elif angle == 90:
        return y, 1 - x
    elif angle == 180:
        return 1 - x, 1 - y
    elif angle == 270:
        return 1 - y, x
    else:
        raise Exception("angle not conform")


def build_suffix(param_list):
    string_char = ""
    count = 1
    last_value = ""
    for idx, value in enumerate(param_list):
        if value == last_value:
            count += 1
        else:
            if len(str(last_value)) > 0:
                string_char += "." + (str(last_value) if count == 1 else "%sx%d" % (last_value, count))
            count = 1
            last_value = value
    string_char += "." + (str(last_value) if count == 1 else "%sx%d" % (last_value, count))

    return string_char[1:]


def get_eye_image_from_video(video, frame, landmarks, roll):
    img = Video(os.path.join(config.video_root, "%s.MP4" % video))[frame]
    img_size = img.shape[0]
    landmarks = (np.array(landmarks) / 640) * img_size
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    for i in range(len(landmarks[0])):
        landmarks[0, i], landmarks[1, i] = rotate_point((landmarks[0, i], landmarks[1, i]), -roll, center_point=image_center)
    rotated_img = rotate_image(img, roll)

    # print(roll)
    # r = rotated_img.copy()
    # r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
    # for i in range(len(landmarks[0])):
    #     cv2.circle(r, (int(landmarks[0, i]), int(landmarks[1, i])), 1, (255, 255, 255))
    # cv2.imshow("img", r)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    eye_bbox = [min(landmarks[0, 36:48]),
                min(landmarks[1, 36:48]),
                max(landmarks[0, 36:48]) - min(landmarks[0, 36:48]),
                max(landmarks[1, 36:48]) - min(landmarks[1, 36:48])]

    eye_img, _ = crop_roi(rotated_img, eye_bbox, padding=5)
    if 0 in eye_img.shape:
        eye_img = np.zeros((224, 224, 3))
    else:
        eye_img = cv2.resize(eye_img, (224, 224))
    return eye_img


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def rotate_point(point, angle, center_point=(0.5, 0.5)):
    """Rotates a point around another centerPoint. Angle is in degrees.
    Rotation is counter-clockwise"""
    angle = math.radians(angle)
    new_point = point[0] - center_point[0], point[1] - center_point[1]
    new_point = (new_point[0] * math.cos(angle) - new_point[1] * math.sin(angle),
                  new_point[0] * math.sin(angle) + new_point[1] * math.cos(angle))
    new_point = new_point[0] + center_point[0], new_point[1] + center_point[1]
    return new_point[0], new_point[1]


def save_results(file, videos, frames, names, results):
    if os.path.exists(file):
        df = pd.read_csv(file)
    else:
        print("creating dump file", file)
        df = pd.DataFrame(columns=["video", "frame", "name", "target"])

    for i in range(len(results)):
        df = df.append(pd.Series([videos[i], int(frames[i]), names[i], results[i]], index=df.columns), ignore_index=True)
    df.to_csv(file, index=False)


def output_processing(outputs, names_list):
    r = []
    _, predicted = torch.max(outputs.data, 1)
    for j in range(predicted.size(0)):
        if predicted[j] == 0:
            r.append('z')
        else:
            r.append(names_list[predicted[j]][j])
    return r
