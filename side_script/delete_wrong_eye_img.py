import os

dir = "data/eyes_img"
for i in os.listdir(dir):
    if i[:6] == "171218":
        os.remove(os.path.join(dir, i))

dir = "data/inputs"
for i in os.listdir(dir):
    if i[:6] == "171218":
        os.remove(os.path.join(dir, i))
