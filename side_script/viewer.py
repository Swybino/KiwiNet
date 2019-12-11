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

    # def plot_pose_box(image, Ps, pts68s, color=(40, 255, 0), line_width=2):
    #     ''' Draw a 3D box as annotation of pose. Ref:https://github.com/yinguobing/head-pose-estimation/blob/master/pose_estimator.py
    #     Args:
    #         image: the input image
    #         P: (3, 4). Affine Camera Matrix.
    #         kpt: (2, 68) or (3, 68)
    #     '''
    #     image = image.copy()
    #     if not isinstance(pts68s, list):
    #         pts68s = [pts68s]
    #     if not isinstance(Ps, list):
    #         Ps = [Ps]
    #     for i in range(len(pts68s)):
    #         pts68 = pts68s[i]
    #         llength = calc_hypotenuse(pts68)
    #         point_3d = build_camera_box(llength)
    #         P = Ps[i]
    #
    #         # Map to 2d image points
    #         point_3d_homo = np.hstack((point_3d, np.ones([point_3d.shape[0], 1])))  # n x 4
    #         point_2d = point_3d_homo.dot(P.T)[:, :2]
    #
    #         point_2d[:, 1] = - point_2d[:, 1]
    #         point_2d[:, :2] = point_2d[:, :2] - np.mean(point_2d[:4, :2], 0) + np.mean(pts68[:2, :27], 1)
    #         point_2d = np.int32(point_2d.reshape(-1, 2))
    #
    #         # Draw all the lines
    #         cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
    #         cv2.line(image, tuple(point_2d[1]), tuple(
    #             point_2d[6]), color, line_width, cv2.LINE_AA)
    #         cv2.line(image, tuple(point_2d[2]), tuple(
    #             point_2d[7]), color, line_width, cv2.LINE_AA)
    #         cv2.line(image, tuple(point_2d[3]), tuple(
    #             point_2d[8]), color, line_width, cv2.LINE_AA)
    #
    #     return image

    def show(self):
        cv2.imshow("img", self.img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def evaluation(self, bboxes=None):
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
