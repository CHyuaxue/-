import cv2
import numpy as np
import glob

img_path = 'D:/Lab/dataset/CCD/train/'
img_list = sorted(glob.glob(img_path + '*.png'))
print(len(img_list))

for i in range(len(img_list)):
    img = cv2.imread(img_list[i], cv2.IMREAD_UNCHANGED)
    # img = np.array(img, dtype=np.float32)
    img = img.astype(np.float32)

    img = (img / (2**12 - 1)) * 100
    img = np.clip(img * (255.0 / np.percentile(img, 100-2.5, keepdims=True)), 0, 255)

    # image = np.array(img, dtype=np.uint8)
    image = img.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # =============================================
    # 白平衡操作
    # =============================================

    gamma = 1/2.2
    image = pow(image, gamma) * (255.0 / pow(255, gamma))
    image = np.array(image, dtype=np.uint8)

    image8 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite('gamma_all/' + str(i+1) + '.png', image8)

    if i % 25 == 0:
        print(str(i) + ' Success!')