import os
import argparse
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
from distinctipy import distinctipy
from tqdm import tqdm

COLORS = np.array(distinctipy.get_colors(100))
COLORS[0] = [0, 0, 0]

args = argparse.ArgumentParser()
args.add_argument("--img_fn", type=str, default=None)
args.add_argument("--img_seq_folder", type=str, default=None)
args.add_argument("--img_lbl_folder", type=str, default=None)


cfg = args.parse_args()

if cfg.img_fn is not None:
    img = cv2.imread(cfg.img_fn)[:,:,::-1]
    plt.subplot(1,2,1), plt.imshow(img[:,:,0])
    plt.subplot(1,2,2), plt.imshow(img[:,:,1])
    plt.show()

if cfg.img_seq_folder is not None:
    assert cfg.img_lbl_folder is not None, "Please provide a label folder"
    img_fns = natsorted(glob.glob(os.path.join(cfg.img_seq_folder, "**", '*.png')))

    for i, img_fn in tqdm(enumerate(img_fns), total=len(img_fns)):
        img_name, img_ext = os.path.splitext(os.path.basename(img_fn))
        lbl_fn = os.path.join(cfg.img_lbl_folder, img_name.rsplit("_",1)[0], img_name + '_all.png')
        freq_num = lbl_fn.rsplit("_")[-2]

        if int(freq_num) % 5 != 0:
            continue
        if os.path.exists(lbl_fn):
            # pass
            img_bgr = cv2.imread(img_fn)
            lbl_bgr = cv2.imread(lbl_fn)
            print(img_bgr.shape, lbl_bgr.shape)
            part_lbls = (COLORS[lbl_bgr[:,:,1]]*255).astype(np.uint8)

            alpha = 0.5
            beta = (1.0 - alpha)
            img_lbl_bgr = cv2.addWeighted(img_bgr, alpha, part_lbls, beta, 0.0)

            cv2.imshow("img", img_bgr)
            cv2.imshow("lbl", part_lbls)
            cv2.imshow("img_lbl", img_lbl_bgr)
            cv2.waitKey(1)
        else:
            print(lbl_fn, "DOESN'T EXIST")


