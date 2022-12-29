import torch
from posenet.constants import *
from posenet.decode_multi import decode_multiple_poses
from posenet.models.model_factory import load_model
from posenet.utils import *
import json
name="origin"
testfile = "origin"+".jpg"
net = load_model(101)
net = net.cuda()
output_stride = net.output_stride
scale_factor = 1.0

input_image, draw_image, output_scale = posenet.read_imgfile(testfile, scale_factor=scale_factor, output_stride=output_stride)
#print(input_image)
with torch.no_grad():
    input_image = torch.Tensor(input_image).cuda()

    heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = net(input_image)

    pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
        heatmaps_result.squeeze(0),
        offsets_result.squeeze(0),
        displacement_fwd_result.squeeze(0),
        displacement_bwd_result.squeeze(0),
        output_stride=output_stride,
        max_pose_detections=20,
        min_pose_score=0.1)
poses = []
# find face keypoints & detect face mask
for pi in range(len(pose_scores)):
    if pose_scores[pi] != 0.:
        #print('Pose #%d, score = %f' % (pi, pose_scores[pi]))       
        keypoints = keypoint_coords.astype(np.int32) # convert float to integer
        #print(keypoints[pi])
        poses.append(keypoints[pi])
# map rccpose-to-openpose mapping
indices = [0, (5,6), 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
i=0
pose = poses[np.argmax(pose_scores)]
openpose = []
for ix in indices:
    if ix==(5,6):
        openpose.append([int((pose[5][1]+pose[6][1])/2), int((pose[5][0]+pose[6][0])/2), 1])   
    else:
        openpose.append([int(pose[ix][1]),int(pose[ix][0]),1])        
    i+=1
coords = []
for x,y,z in openpose:
    coords.append(float(x))
    coords.append(float(y))
    coords.append(float(z))

data = {"version": 1.0}
pose_dic = {}
pose_dic['pose_keypoints_2d'] = coords
tmp = []
tmp.append(pose_dic)
data["people"]=tmp


# VITON's .json is in ACGPN_TestData/test_pose/000001_0_keypoints.json
pose_name = './HR-VITON-main/test/test/openpose_json/00001_00_keypoints.json' 
with open(pose_name,'w') as f:
        json.dump(data, f)