import cv2
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


# openpose
from network.rtpose_vgg import get_model
from evaluate.coco_eval import get_multiplier, get_outputs

# utils
from openpose_utils import remove_noise, get_pose

save_dir = Path('./data/source/')
save_dir.mkdir(exist_ok=True)

img_dir = save_dir.joinpath('images')
img_dir.mkdir(exist_ok=True)

# convert video to images
#########################################
#########################################
# cap = cv2.VideoCapture(str(save_dir.joinpath('mv.mp4')))
# i = 0
# while cap.isOpened():
#     flag, frame = cap.read()
#     if flag == False or i == 1000:
#         break
#     cv2.imwrite(str(img_dir.joinpath('img_%04d.png' % i)), frame)
#     i += 1
#########################################
#########################################

openpose_dir = Path('src/pytorch_Realtime_Multi-Person_Pose_Estimation/')


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
# img_path = sorted(img_dir.iterdir())[137]
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
#########################################
#########################################

# make label images for pix2pix
#########################################
#########################################
test_img_dir = save_dir.joinpath('test_img')
test_img_dir.mkdir(exist_ok=True)
test_label_dir = save_dir.joinpath('test_label')
test_label_dir.mkdir(exist_ok=True)

for idx in tqdm(range(100, 100 + 300)):
    img_path = img_dir.joinpath('img_%04d.png' % idx)
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
    cv2.imwrite(str(test_img_dir.joinpath('img_%04d.png' % idx)), img)

    cv2.imwrite(str(test_label_dir.joinpath('label_%04d.png' % idx)), label)

torch.cuda.empty_cache()
#########################################
#########################################