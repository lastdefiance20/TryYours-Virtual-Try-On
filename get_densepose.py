import cv2
from PIL import Image
import pickle
import json
import numpy as np

colormap = {
    2 : [20, 80, 194],
    3 : [4, 98, 224],
    4 : [8, 110, 221],
    9 : [6, 166, 198],
    10 : [22, 173, 184],
    15 : [145, 191, 116],
    16 : [170, 190, 105],
    17 : [191, 188, 97],
    18 : [216, 187, 87],
    19 : [228, 191, 74],
    20 : [240, 198, 60],
    21 : [252, 205, 47],
    22 : [250, 220, 36],
    23 : [251, 235, 25],
    24 : [248, 251, 14],
}

img = Image.open('./origin.jpg')
img_w ,img_h = img.size
with open('./data.json', 'r') as f:
    json_data = json.load(f)
i = np.array(json_data[0])

seg_img=np.zeros((i.shape[0],i.shape[1],3))

for y_idx in range(i.shape[0]):
    for x_idx in range(i.shape[1]):
        if i[y_idx][x_idx] in colormap:
            seg_img[y_idx][x_idx] = colormap[i[y_idx][x_idx]]
        else:
            seg_img[y_idx][x_idx] = [0, 0, 0]
            
box = json_data[2]
box[2]=box[2]-box[0]
box[3]=box[3]-box[1]
x,y,w,h=[int(v) for v in box]
bg=np.zeros((img_h,img_w,3))
bg[y:y+h,x:x+w,:] = seg_img
bg_img = Image.fromarray(np.uint8(bg),"RGB")
bg_img.save("./HR-VITON-main/test/test/image-densepose/00001_00.jpg")