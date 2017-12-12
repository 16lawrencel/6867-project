import pickle
import matplotlib.pyplot as plt
import numpy as np

num = 39347193

with open('R_data', 'rb') as f:
    R_list = pickle.load(f)

x = np.linspace(0, num, len(R_list))

plt.plot(x, R_list)
plt.title('Score over time for Pong')
plt.xlabel('Frames')
plt.ylabel('Score')
plt.show()
