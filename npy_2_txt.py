import numpy as np

file = np.load('./labels_test.npy')
print(file)
np.savetxt('./labels_test.txt', file)
