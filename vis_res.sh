#!/usr/bin/env sh
python ./visualize_result.py --deploy_file ./models/deploy_1024.prototxt --size 600,480 --mode reshape --threshold 4. --model_file ./models/face_detectnet_1024.caffemodel --input *path_to_folder_with_images* --gpu gpu
#python ./visualize_result.py --deploy_file ./models/deploy_600.prototxt --size 600,480 --mode resize --threshold 4. --model_file ./models/face_detectnet_600.caffemodel --input *path_to_folder_with_images* --gpu gpu
