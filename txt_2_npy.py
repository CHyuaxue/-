import numpy as np

a = np.loadtxt(fname='./labels_test.txt')

# print(a)
np.save('./labels_test_3.npy', a)
