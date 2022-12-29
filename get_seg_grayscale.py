import cv2
from PIL import Image
import pickle
import json
import numpy as np

img = Image.open('./resized_segmentation_img.png')
img_w ,img_h = img.size

img = np.array(img)
gray_img=np.zeros((img_h,img_w))

for y_idx in range(img.shape[0]):
    for x_idx in range(img.shape[1]):    
        tmp = img[y_idx][x_idx]
        if np.array_equal(tmp, [0,0,0]):
            gray_img[y_idx][x_idx] = 0
        if np.array_equal(tmp, [255,0,0]):
            gray_img[y_idx][x_idx] = 2 #머리카락
        elif np.array_equal(tmp, [0,0,255]):
            gray_img[y_idx][x_idx] = 13 #머리
        elif np.array_equal(tmp, [85, 51, 0]):
            gray_img[y_idx][x_idx] = 10 #목
        elif np.array_equal(tmp, [255, 85, 0]):
            gray_img[y_idx][x_idx] = 5 #몸통
        elif np.array_equal(tmp, [0, 255, 255]):
            gray_img[y_idx][x_idx] = 15 #왼팔
        elif np.array_equal(tmp, [51, 170, 221]):
            gray_img[y_idx][x_idx] = 14 #오른팔
        elif np.array_equal(tmp, [0, 85, 85]):
            gray_img[y_idx][x_idx] = 9 #바지
        elif np.array_equal(tmp, [0, 0, 85]):
            gray_img[y_idx][x_idx] = 6 #원피스
        elif np.array_equal(tmp, [0, 128, 0]):
            gray_img[y_idx][x_idx] = 12 #치마
        elif np.array_equal(tmp, [177, 255, 85]):
            gray_img[y_idx][x_idx] = 17 #왼다리
        elif np.array_equal(tmp, [85, 255, 170]):
            gray_img[y_idx][x_idx] = 16 #오른다리
        elif np.array_equal(tmp, [0, 119, 221]):
            gray_img[y_idx][x_idx] = 5 #외투
        else:
            gray_img[y_idx][x_idx] = 0
            
img=cv2.resize(gray_img,(768,1024),interpolation=cv2.INTER_NEAREST)
bg_img = Image.fromarray(np.uint8(img),"L")
bg_img.save("./HR-VITON-main/test/test/image-parse-v3/00001_00.png")