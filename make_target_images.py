import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from pathlib import Path
import sys

# openpose
from network.rtpose_vgg import get_model
from evaluate.coco_eval import get_multiplier, get_outputs

# utils
from openpose_utils import remove_noise, get_pose


# set dirs
save_dir = Path('./data/target/')
save_dir.mkdir(exist_ok=True)

img_dir = save_dir.joinpath('images')
img_dir.mkdir(exist_ok=True)

# create images from target video
#########################################
#########################################
# vc = cv2.VideoCapture(str(save_dir.joinpath('target.mp4')))
#
# cnt = 0
# while vc.isOpened():
#     is_capturing, img = vc.read()
#     # img = img[:, 80:80+480]
#     cv2.imwrite('./data/target/images/img_%4d.png' % cnt, img)
#     cv2.imshow('video', img)
#     if is_capturing == False or cnt == 5000:
#         break
#     cnt += 1
#########################################
#########################################


openpose_dir = Path('./src/pytorch_Realtime_Multi-Person_Pose_Estimation/')


# build VGG19
# weight_name = openpose_dir.joinpath('network/weight/pose_model.pth')
weight_name = 'src/pose_model.pth'
model = get_model('vgg19')
model.load_state_dict(torch.load(weight_name))
model = torch.nn.DataParallel(model).cuda()
model.float()
model.eval()

# check
#########################################
#########################################
# img_path = sorted(img_dir.iterdir())[0]
# img = cv2.imread(str(img_path))
# shape_dst = np.min(img.shape[:2])
# # offset
# oh = (img.shape[0] - shape_dst) // 2
# ow = (img.shape[1] - shape_dst) // 2
#
# img = img[oh:oh + shape_dst, ow:ow + shape_dst]
# img = cv2.resize(img, (512, 512))
#
# plt.imshow(img[:, :, [2, 1, 0]])  # BGR -> RGB
#
# multiplier = get_multiplier(img)
# with torch.no_grad():
#     paf, heatmap = get_outputs(multiplier, img, model, 'rtpose')
#
# r_heatmap = np.array([remove_noise(ht)
#                       for ht in heatmap.transpose(2, 0, 1)[:-1]]) \
#     .transpose(1, 2, 0)
# heatmap[:, :, :-1] = r_heatmap
# param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
# label = get_pose(param, heatmap, paf)
#
# plt.imshow(label)
#########################################
#########################################

# make label images for pix2pix
#########################################
#########################################
train_dir = save_dir.joinpath('train')
train_dir.mkdir(exist_ok=True)

train_img_dir = train_dir.joinpath('train_img')
train_img_dir.mkdir(exist_ok=True)
train_label_dir = train_dir.joinpath('train_label')
train_label_dir.mkdir(exist_ok=True)

for idx in tqdm(range(100, 700)):
    img_path = img_dir.joinpath('img_%4d.png' % idx)
    img = cv2.imread(str(img_path))
    shape_dst = np.min(img.shape[:2])
    oh = (img.shape[0] - shape_dst) // 2
    ow = (img.shape[1] - shape_dst) // 2

    img = img[oh:oh + shape_dst, ow:ow + shape_dst]
    img = cv2.resize(img, (512, 512))
    multiplier = get_multiplier(img)
    with torch.no_grad():
        paf, heatmap = get_outputs(multiplier, img, model, 'rtpose')
    r_heatmap = np.array([remove_noise(ht)
                          for ht in heatmap.transpose(2, 0, 1)[:-1]]) \
        .transpose(1, 2, 0)
    heatmap[:, :, :-1] = r_heatmap
    param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
    label = get_pose(param, heatmap, paf)

    cv2.imwrite(str(train_img_dir.joinpath('img_%04d.png' % idx)), img)
    cv2.imwrite(str(train_label_dir.joinpath('label_%04d.png' % idx)), label)

torch.cuda.empty_cache()
#########################################
#########################################