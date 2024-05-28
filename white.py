import cv2
import numpy as np
import scipy.io
import glob

img_path = 'D:/Lab/dataset/CCD/train/'
img_list = sorted(glob.glob(img_path + '*.png'))

mat = scipy.io.loadmat('real_illum_568..mat')['real_rgb']

for i in range(len(img_list)):
    img = cv2.imread(img_list[i], cv2.IMREAD_UNCHANGED)

    img = (img / ((2**12)-1)) * 100
    img = np.clip(img * (255.0 / np.percentile(img, 100-2.5, keepdims=True)), 0, 255)
    image = img.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ground_truth = mat[i, :]

    Gain_R = float(np.max(ground_truth)) / float(ground_truth[0])
    Gain_G = float(np.max(ground_truth)) / float(ground_truth[1])
    Gain_B = float(np.max(ground_truth)) / float(ground_truth[2])

    image[:, :, 0] = np.minimum(Gain_R * image[:, :, 0], 255)
    image[:, :, 1] = np.minimum(Gain_G * image[:, :, 1], 255)
    image[:, :, 2] = np.minimum(Gain_B * image[:, :, 2], 255)

    gamma = 1/2.2
    image = pow(image, gamma) * (255.0 / pow(255, gamma))
    image = np.array(image, dtype=np.uint8)

    image8 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite('white_all/' + str(i+1) + '.png', image8)

    # if i % 10 == 0:
    print(str(i+1) + ' Success!')