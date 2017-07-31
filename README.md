# FaceDetectNet
Caffe Implementation of Face Detection using DetectNet

## Model files
./models contains two caffemodel files: face_detectnet_1024.caffemodel (*input image: 1024x640x3*) and face_detectnet_600.caffemodel (*input_image: 600x480x3*)

## Main file usage
run ./vis_res.sh 
OR 
python ./visualize_result.py --deploy_file *./models/deploy_600.prototxt* --size *600,480* --mode *reshape* --threshold *4.* --model_file *./models/face_detectnet_600.caffemodel* --input *path_to_folder_with_images* --gpu gpu 

to do face detection for the input image

replace *path_to_folder_with_images* with the target folder contains image file 
and choose mode for detector: size_image => deploy_file and model_file 
(for 1024x640x3 input image - *deploy_1024.prototxt* and *face_detectnet_1024.caffemodel*; 
for 600x480x3 input image - *deploy_600.prototxt* and *face_detectnet_600.caffemodel*), 
threshold (default: 4.) and mode: reshape net (--mode reshape) or resize image (--mode resize)

## Requirements
Caffe, pycaffe

## Example:
<p align="left">  <img src="https://github.com/NIIAS3050/FaceDetectNet/blob/master/examples/result_img.png"></p>
