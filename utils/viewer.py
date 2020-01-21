import cv2
import numpy as np
from math import cos, sin

LANDMARK_COLOR = (255, 255, 255)
SCALING = 1.5

class Viewer:
    def __init__(self, img_path):
        self.img = cv2.imread(img_path)
        self.img = cv2.resize(self.img, (int(self.img.shape[0] * SCALING), int(self.img.shape[1] *SCALING)))
        return

    def plt_frame_idx(self, frame_idx):
        cv2.putText(self.img, "Frame %d" % frame_idx, (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

    def plt_bbox(self, bbox, label="", color=(0, 255, 0), **kwargs):
        bbox = [i*SCALING for i in bbox]
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(self.img, p1, p2, color, **kwargs)
        p1 = (p1[0], p1[1] - 10)
        cv2.putText(self.img, str(label), p1, cv2.FONT_HERSHEY_DUPLEX, 1, color)
        return

    def plt_landmarks(self, landmarks, color=LANDMARK_COLOR):
        for i in range(len(landmarks[0])):
            final_color = color
            if 36 <= i < 48 or 17 <= i < 27:
                final_color = (0, color[1], color[2])
            elif 27 <= i < 36:
                final_color = (80, color[1], color[2])
            elif 48 <= i < 68:
                final_color = (180, color[1], color[2])
            elif i < 17:
                final_color = (255, color[1], color[2])

            # cv2.putText(self.img, str(i), (int(landmarks[0][i]*SCALING)-5, int(landmarks[1][i]*SCALING)-5), cv2.FONT_HERSHEY_DUPLEX, 0.2, color=(255, 255, 255))
            cv2.circle(self.img, (int(landmarks[0][i]*SCALING), int(landmarks[1][i]*SCALING)), 1, color=final_color)

    def plt_axis(self, yaw, pitch, roll, tdx=None, tdy=None, size=50):
        pitch = pitch * np.pi / 180
        yaw = -(yaw * np.pi / 180)
        roll = roll * np.pi / 180

        if tdx is None and tdy is None:
            height, width = self.img.shape[:2]
            tdx = width / 2
            tdy = height / 2

        else:
            tdx = tdx * SCALING
            tdy = tdy * SCALING

        # X-Axis pointing to right. drawn in red
        x1 = size * (cos(yaw) * cos(roll)) + tdx
        y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

        # Y-Axis | drawn in green
        #        v
        x2 = size * (-cos(yaw) * sin(roll)) + tdx
        y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

        # Z-Axis (out of the screen) drawn in blue
        x3 = size * (sin(yaw)) + tdx
        y3 = size * (-cos(yaw) * sin(pitch)) + tdy

        cv2.line(self.img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 2)
        cv2.line(self.img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.line(self.img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 56, 0), 2)

    def plt_results(self, results, positions_list):
        """

        :param results:
        :type results: dict
        :param positions_list:
        :type positions_list:
        :return:
        :rtype:
        """
        for key, value in results.items():
            if value is None:
                pt1 = positions_list[key]
                pt1 = np.array([pt1[0] + 0.5 * pt1[2], pt1[1] + 0.5 * pt1[3]])
                cv2.circle(self.img, pt1, 1, (0, 0, 255))
            else:
                pt1, pt2 = positions_list[key], positions_list[value]
                pt1 = np.array([pt1[0] + 0.5*pt1[2], pt1[1] + 0.5*pt1[3]])
                pt2 = np.array([pt2[0] + 0.5*pt2[2], pt2[1] + 0.5*pt2[3]])
                v = pt2 - pt1
                pt1, pt2 = pt1 + 0.1 * v, pt1 + 0.8 * v

                cv2.arrowedLine(self.img, pt1, pt2, (0, 0, 255), 5)

    def show(self):
        cv2.imshow("img", self.img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def landmarks_evaluation(self, bboxes=None):
        eval = []
        cv2.imshow("img", self.img)
        while True:
            key = cv2.waitKey()
            if key == 8:
                return None
            elif key == 32:
                print("ok")
                self.plt_bbox(bboxes[len(eval)], color=(0, 255, 0))
                cv2.imshow("img", self.img)
                eval.append(1)
            elif key == ord('n'):
                print("not ok")
                self.plt_bbox(bboxes[len(eval)], color=(0, 0, 255))
                cv2.imshow("img", self.img)
                eval.append(0)
            elif key == ord('b'):
                print("back")
                eval.pop()
                self.plt_bbox(bboxes[len(eval)], color=(128, 128, 128))
                cv2.imshow("img", self.img)
            else:
                break
            if len(eval) == len(bboxes):
                break
        cv2.destroyAllWindows()
        return eval