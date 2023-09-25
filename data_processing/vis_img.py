import argparse
import cv2
import matplotlib.pyplot as plt

args = argparse.ArgumentParser()
args.add_argument("--img_fn", type=str)
cfg = args.parse_args()

img = cv2.imread(cfg.img_fn)[:,:,::-1]
plt.subplot(1,2,1), plt.imshow(img[:,:,0])
plt.subplot(1,2,2), plt.imshow(img[:,:,1])
plt.show()
