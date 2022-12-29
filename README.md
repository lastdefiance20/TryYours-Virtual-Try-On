# TryYours

1. INPUT: ./origin.jpg
2. origin.jpg를 1024*768로 변경 (7~10 line)
3. openpose 좌표 추출 및 ./HR-VITON-main/test/test/openpose_json/00001_00_keypoints.json 에 좌표 저장(11~12 line)
4. 512*384 size 변경 (14~16 line)
5. segmentation 이미지 생성 (라이브러리) (20~25 line)
6. 5를 이용해서 배경 제거 (30~45 line)

* 7. 새로운 segmentation label 이미지 생성 및 ./HR-VITON-main/test/test/ image-parse-v3/00001_00.png 에 저장 (input= ./origin.png) (unet) (49~50 line) ()

* 8. densepose 이미지 생성 및 ./HR-VITON-main/test/test/densepose/00001_00.jpg 에 저장 (unet) (input= ./origin.png) (57~58)
9. HD-VTION 돌리기 




7과 8을 변경하면 됨