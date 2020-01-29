import cv2
import numpy as np
import os
import os.path as osp
import math


def solve_head_pose(landmarks):
    if type(landmarks) == list:
        landmarks = np.array(landmarks)
    if landmarks.shape == (3, 68):
        landmarks = np.transpose(landmarks)
    assert landmarks.shape == (68, 3), "bad landmark shape"
    l_eye_center = np.mean(landmarks[36:42, :], axis=0)
    r_eye_center = np.mean(landmarks[42:48, :], axis=0)
    mouth_center = np.mean(landmarks[48:68, :], axis=0)

    dY = r_eye_center[1] - l_eye_center[1]
    dX = r_eye_center[0] - l_eye_center[0]

    roll = 90 if dX == 0 else np.degrees(np.arctan(dY / dX))
    if dX < 0 < dY:
        roll = roll + 180
    elif dX < 0 and dY < 0:
        roll = roll - 180

    p1 = np.array(r_eye_center)
    p2 = np.array(l_eye_center)
    p3 = np.array(mouth_center)

    # These two vectors are in the plane
    v1 = p2 - p1
    v1 = v1 / np.linalg.norm(v1)
    v2 = p3 - p1
    v2 = v2 / np.linalg.norm(v2)

    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    cp_norm = np.linalg.norm(cp)
    cp = cp / cp_norm
    yaw = np.degrees(np.arcsin(cp[0]))
    pitch = np.degrees(np.arcsin(cp[1]))

    # print("roll: %d, pitch %d, yaw %d" % (roll, pitch, yaw))

    return int(roll), int(pitch), int(yaw)


def face_orientation(frame, landmarks):
    size = frame.shape  # (height, width, color_channel)

    image_points = np.array([
        (landmarks[33, 0], landmarks[33, 1]),  # Nose tip
        (landmarks[8, 0], landmarks[8, 1]),  # Chin
        (landmarks[36, 0], landmarks[36, 1]),  # Left eye left corner
        (landmarks[45, 0], landmarks[45, 1]),  # Right eye right corner
        (landmarks[48, 0], landmarks[48, 1]),  # Left Mouth corner
        (landmarks[54, 0], landmarks[54, 1])  # Right mouth corner
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-165.0, 170.0, -135.0),  # Left eye left corner
        (165.0, 170.0, -135.0),  # Right eye right corne
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ])

    # Camera internals

    center = (size[1] / 2, size[0] / 2)
    focal_length = center[0] / np.tan(60 / 2 * np.pi / 180)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    axis = np.float32([[500, 0, 0],
                       [0, 500, 0],
                       [0, 0, 500]])

    imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]

    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))

    # nose = (landmarks[33, 0], landmarks[33, 1])
    # cv2.line(frame, nose, tuple(imgpts[1].ravel()), (0, 255, 0), 3)  # GREEN
    # cv2.line(frame, nose, tuple(imgpts[0].ravel()), (255, 0,), 3)  # BLUE
    # cv2.line(frame, nose, tuple(imgpts[2].ravel()), (0, 0, 255), 3)  # RED

    return imgpts, modelpts, (int(roll), int(pitch), int(yaw)), (landmarks[4], landmarks[5])

#
# # class PoseEstimator:
# #     def __init__(self):
# #         return
# #
# #     def get_face_orientation(self, img, landmarks):
# #         size = img.shape
# #         focal_length = size[1]
# #         center = (size[1] / 2, size[0] / 2)
# #         camera_matrix = np.array(
# #             [[focal_length, 0, center[0]],
# #              [0, focal_length, center[1]],
# #              [0, 0, 1]], dtype="double"
# #         )
# #
# #         dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
# #         return cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.CV_ITERATIVE)
# #
# #         # (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
# #         #                                                               dist_coeffs,
# #         #                                                               flags=cv2.CV_ITERATIVE)
# #
# #     def get_headpose(self, im, landmarks_2d, verbose=False):
# #         h, w, c = im.shape
# #         f = w  # column size = x axis length (focal length)
# #         u0, v0 = w / 2, h / 2  # center of image plane
# #         camera_matrix = np.array(
# #             [[f, 0, u0],
# #              [0, f, v0],
# #              [0, 0, 1]], dtype=np.double
# #         )
# #
# #         # Assuming no lens distortion
# #         dist_coeffs = np.zeros((4, 1))
# #
# #         # Find rotation, translation
# #         (success, rotation_vector, translation_vector) = cv2.solvePnP(self.landmarks_3d, landmarks_2d, camera_matrix,
# #                                                                       dist_coeffs)
# #
# #         if verbose:
# #             print("Camera Matrix:\n {0}".format(camera_matrix))
# #             print("Distortion Coefficients:\n {0}".format(dist_coeffs))
# #             print("Rotation Vector:\n {0}".format(rotation_vector))
# #             print("Translation Vector:\n {0}".format(translation_vector))
# #
# #         return rotation_vector, translation_vector, camera_matrix, dist_coeffs
# #
# #
# #     def get_angles(self, rvec, tvec):
# #         rmat = cv2.Rodrigues(rvec)[0]
# #         P = np.hstack((rmat, tvec))  # projection matrix [R | t]
# #         degrees = -cv2.decomposeProjectionMatrix(P)[6]
# #         rx, ry, rz = degrees[:, 0]
# #         return [rx, ry, rz]
#
#
#
# class HeadposeDetection():
#     # 3D facial model coordinates
#     landmarks_3d_list = [
#         np.array([
#             [0.000, 0.000, 0.000],  # Nose tip
#             [0.000, -8.250, -1.625],  # Chin
#             [-5.625, 4.250, -3.375],  # Left eye left corner
#             [5.625, 4.250, -3.375],  # Right eye right corner
#             [-3.750, -3.750, -3.125],  # Left Mouth corner
#             [3.750, -3.750, -3.125]  # Right mouth corner
#         ], dtype=np.double),
#         np.array([
#             [0.000000, 0.000000, 6.763430],  # 52 nose bottom edge
#             [6.825897, 6.760612, 4.402142],  # 33 left brow left corner
#             [1.330353, 7.122144, 6.903745],  # 29 left brow right corner
#             [-1.330353, 7.122144, 6.903745],  # 34 right brow left corner
#             [-6.825897, 6.760612, 4.402142],  # 38 right brow right corner
#             [5.311432, 5.485328, 3.987654],  # 13 left eye left corner
#             [1.789930, 5.393625, 4.413414],  # 17 left eye right corner
#             [-1.789930, 5.393625, 4.413414],  # 25 right eye left corner
#             [-5.311432, 5.485328, 3.987654],  # 21 right eye right corner
#             [2.005628, 1.409845, 6.165652],  # 55 nose left corner
#             [-2.005628, 1.409845, 6.165652],  # 49 nose right corner
#             [2.774015, -2.080775, 5.048531],  # 43 mouth left corner
#             [-2.774015, -2.080775, 5.048531],  # 39 mouth right corner
#             [0.000000, -3.116408, 6.097667],  # 45 mouth central bottom corner
#             [0.000000, -7.415691, 4.070434]  # 6 chin corner
#         ], dtype=np.double),
#         np.array([
#             [0.000000, 0.000000, 6.763430],  # 52 nose bottom edge
#             [5.311432, 5.485328, 3.987654],  # 13 left eye left corner
#             [1.789930, 5.393625, 4.413414],  # 17 left eye right corner
#             [-1.789930, 5.393625, 4.413414],  # 25 right eye left corner
#             [-5.311432, 5.485328, 3.987654]  # 21 right eye right corner
#         ], dtype=np.double)
#     ]
#
#     # 2d facial landmark list
#     lm_2d_index_list = [
#         [30, 8, 36, 45, 48, 54],
#         [33, 17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8],  # 14 points
#         [33, 36, 39, 42, 45]  # 5 points
#     ]
#
#     def __init__(self, lm_type=1, verbose=True):
#
#         self.lm_2d_index = self.lm_2d_index_list[lm_type]
#         self.landmarks_3d = self.landmarks_3d_list[lm_type]
#
#         self.v = verbose
#
#     def to_numpy(self, landmarks):
#         coords = []
#         for i in self.lm_2d_index:
#             coords += [[landmarks.part(i).x, landmarks.part(i).y]]
#         return np.array(coords).astype(np.int)
#
#     def get_headpose(self, im, landmarks_2d, verbose=False):
#         h, w, c = im.shape
#         f = w  # column size = x axis length (focal length)
#         u0, v0 = w / 2, h / 2  # center of image plane
#         camera_matrix = np.array(
#             [[f, 0, u0],
#              [0, f, v0],
#              [0, 0, 1]], dtype=np.double
#         )
#
#         # Assuming no lens distortion
#         dist_coeffs = np.zeros((4, 1))
#
#         # Find rotation, translation
#         (success, rotation_vector, translation_vector) = cv2.solvePnP(self.landmarks_3d, landmarks_2d, camera_matrix,
#                                                                       dist_coeffs)
#
#         if verbose:
#             print("Camera Matrix:\n {0}".format(camera_matrix))
#             print("Distortion Coefficients:\n {0}".format(dist_coeffs))
#             print("Rotation Vector:\n {0}".format(rotation_vector))
#             print("Translation Vector:\n {0}".format(translation_vector))
#
#         return rotation_vector, translation_vector, camera_matrix, dist_coeffs
#
#     # rotation vector to euler angles
#     def get_angles(self, rvec, tvec):
#         rmat = cv2.Rodrigues(rvec)[0]
#         P = np.hstack((rmat, tvec))  # projection matrix [R | t]
#         degrees = -cv2.decomposeProjectionMatrix(P)[6]
#         rx, ry, rz = degrees[:, 0]
#         return [rx, ry, rz]
#
#     # moving average history
#     history = {'lm': [], 'bbox': [], 'rvec': [], 'tvec': [], 'cm': [], 'dc': []}
#
#     def add_history(self, values):
#         for (key, value) in zip(self.history, values):
#             self.history[key] += [value]
#
#     def pop_history(self):
#         for key in self.history:
#             self.history[key].pop(0)
#
#     def get_history_len(self):
#         return len(self.history['lm'])
#
#     def get_ma(self):
#         res = []
#         for key in self.history:
#             res += [np.mean(self.history[key], axis=0)]
#         return res
#
#     # return image and angles
#     def process_image(self, im, draw=True, ma=3):
#         # landmark Detection
#         im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#
#         rvec, tvec, cm, dc = self.get_headpose(im, landmarks_2d)
#
#         if ma > 1:
#             self.add_history([landmarks_2d, bbox, rvec, tvec, cm, dc])
#             if self.get_history_len() > ma:
#                 self.pop_history()
#             landmarks_2d, bbox, rvec, tvec, cm, dc = self.get_ma()
#
#
#         angles = self.get_angles(rvec, tvec)
#         if self.v:
#             print(', ga: %.2f' % t.toc('ga'), end='ms')
#
#         if draw:
#
#             im = annotator.draw_all()
#             if self.v:
#                 print(', draw: %.2f' % t.toc('draw'), end='ms' + ' ' * 10)
#
#         return im, angles
#
#
# def main(args):
#     in_dir = args["input_dir"]
#     out_dir = args["output_dir"]
#
#     # Initialize head pose detection
#     hpd = HeadposeDetection(args["landmark_type"], args["landmark_predictor"])
#
#     for filename in os.listdir(in_dir):
#         name, ext = osp.splitext(filename)
#         if ext in ['.jpg', '.png', '.gif']:
#             print("> image:", filename, end='')
#             image = cv2.imread(in_dir + filename)
#             res, angles = hpd.process_image(image)
#             cv2.imwrite(out_dir + name + '_out.png', res)
#         else:
#             print("> skip:", filename, end='')
#         print('')
