BBOX_KEY = "bbox"
LANDMARKS_KEY = "landmarks"
POSE_KEY = "pose"
ROTATION_MATRIX_KEY = "rotation_matrix"
YAW_KEY = "yaw"
PITCH_KEY = "pitch"
ROLL_KEY = "roll"
CONFIDENCE_KEY = "confidence"

## NETWORK PARAMETERS
nb_kids = 6
video_root = "data/videos"
eye_img_root = "data/eyes_img"
inputs_dir = "data/inputs_new"
model_folder = "model/flat_models"
kids_color = {"a": (218, 9, 0),
              "b": (62, 52, 229),
              "c": (29, 225, 115),
              "d": (72, 229, 220),
              "e": (67, 89, 105),
              "f": (25, 112, 190),
              "g": (116, 72, 54),

              "h": (24, 18, 168),
              "i": (114, 173, 219),
              "j": (214, 14, 80),
              "k": (170, 66, 241),
              "l": (126, 20, 74)}
