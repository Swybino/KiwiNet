import matplotlib.pyplot as plt
import numpy as np
import pickle


history = pickle.load(open("model/history.512x3.1024x2_2020-01-21.p", "rb"))

y = [i[2] for i in history]
epochs = [i[0] for i in history]
epoch_size = epochs.count(0)
x = [i/epoch_size for i in range(len(history))]
mean_y = []
for i in range(1, epochs[-1]):
    mean_y.append(np.mean(y[(i-1)*epoch_size:min(i*epoch_size, len(y))]))
mean_x = [i for i in range(1, epochs[-1])]

print(mean_y)
plt.title("")
plt.plot(mean_x, mean_y, 'b-')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
