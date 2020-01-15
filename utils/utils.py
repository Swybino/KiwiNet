import numpy as np


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


def gradient(y1, y2, t=1):
    return (y2 - y1) / t


def build_suffix(param_list):
    string_char = ""
    count = 1
    last_value = ""
    for idx, value in enumerate(param_list):
        if value == last_value:
            count += 1
            last_value = "%sx%d" %(last_value, count)
        else:
            count = 1
            string_char += "." + str(last_value)
            last_value = value
    string_char += "." + str(last_value)
    return string_char[1:]



