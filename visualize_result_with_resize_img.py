#!/usr/bin/env python
import csv

import numpy as np
import matplotlib.pyplot as plt
import os

caffe = '/home/user/caffe_nvidia/python'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe)
import caffe
import argparse

from skimage import io, img_as_float, transform

import PIL
from PIL import Image

def IoU(gt_rect, dt_rect):
	dt_area = float(abs((dt_rect[2] - dt_rect[0])*(dt_rect[3] - dt_rect[1])))
	gt_area = float(abs((gt_rect[2] - gt_rect[0])*(gt_rect[3] - gt_rect[1])))

	x_overlap = max(0, min(dt_rect[2], gt_rect[2]) - max(dt_rect[0], gt_rect[0]))
	y_overlap = max(0, min(dt_rect[3], gt_rect[3]) - max(dt_rect[1], gt_rect[1])) #

	intersect = x_overlap * y_overlap    

	union = gt_area + dt_area - intersect
	if (intersect == 0.0 or union == 0.0):
		return 0.0

	if (union == 0.0):
		return 0.0

	IoU = intersect / union
	return IoU    

def getFileName(pathOfFile):
	base = os.path.basename(pathOfFile)
	return os.path.splitext(base)[0]    

def clamp(z, min_v, max_v):
	if(z < min_v):
		return min_v
	if(z > max_v):
		return max_v
	return z

def process_boxes(net_cvg, net_boxes, im_sz_x,im_sz_y,stride,thr,itCNT = 1,thr_IOU = 0.4):
	grid_sz_x = int(im_sz_x / stride)
	grid_sz_y = int(im_sz_y / stride)

	cell_width = im_sz_x / grid_sz_x
	cell_height = im_sz_y / grid_sz_y

	cvg_val = net_cvg[0][0:grid_sz_y][0:grid_sz_x]
	rectz = net_boxes

	x1ref = np.zeros([grid_sz_x, grid_sz_y])
	x2ref = np.zeros([grid_sz_x, grid_sz_y])
	y1ref = np.zeros([grid_sz_x, grid_sz_y])
	y2ref = np.zeros([grid_sz_x, grid_sz_y])
	cvgref = np.zeros([grid_sz_x, grid_sz_y])

	for x in range(0,grid_sz_x):
		for y in range(0, grid_sz_y):
			x1ref[x][y] = rectz[0][y][x] + x * cell_width
			y1ref[x][y] = rectz[1][y][x] + y * cell_height
			x2ref[x][y] = rectz[2][y][x] + x * cell_width
			y2ref[x][y] = rectz[3][y][x] + y * cell_height
			cvgref[x][y] = cvg_val[y][x]

   
	for n in range(0, itCNT):
		x1s = np.zeros([grid_sz_x, grid_sz_y])
		y1s = np.zeros([grid_sz_x, grid_sz_y])
		x2s = np.zeros([grid_sz_x, grid_sz_y])
		y2s = np.zeros([grid_sz_x, grid_sz_y])
		Cnts = np.ones([grid_sz_x, grid_sz_y])
		Cnts2 = np.ones([grid_sz_x, grid_sz_y])
		for x in range(0, grid_sz_x):
			for y in range(0, grid_sz_y):
				if cvgref[x][y]>thr:
					x_beg = x1ref[x][y]/cell_width
					y_beg = y1ref[x][y]/cell_height
					x_end = x2ref[x][y]/cell_width
					y_end = y2ref[x][y]/cell_height
					x_beg = int(clamp(x_beg,0,grid_sz_x-1))
					y_beg = int(clamp(y_beg,0,grid_sz_y-1))
					x_end = int(clamp(x_end,0,grid_sz_x-1))
					y_end = int(clamp(y_end,0,grid_sz_y-1))
					x1s[x][y] += x1ref[x][y]*cvgref[x][y]
					y1s[x][y] += y1ref[x][y]*cvgref[x][y]
					x2s[x][y] += x2ref[x][y]*cvgref[x][y]
					y2s[x][y] += y2ref[x][y]*cvgref[x][y]
					Cnts[x][y] = cvgref[x][y]
        
					for x_i in range(x_beg,x_end+1):
						for y_i in range(y_beg, y_end+1):
							if cvgref[x_i][y_i]>thr:
								IOU = IoU([x1ref[x][y],y1ref[x][y],x2ref[x][y],y2ref[x][y]],[x1ref[x_i][y_i],y1ref[x_i][y_i],x2ref[x_i][y_i],y2ref[x_i][y_i]])
								if(IOU>thr_IOU):
									x1s[x][y] += x1ref[x_i][y_i]*cvgref[x_i][y_i]
									y1s[x][y] += y1ref[x_i][y_i]*cvgref[x_i][y_i]
									x2s[x][y] += x2ref[x_i][y_i]*cvgref[x_i][y_i]
									y2s[x][y] += y2ref[x_i][y_i]*cvgref[x_i][y_i]

									Cnts[x][y] += cvgref[x_i][y_i]
									Cnts2[x][y] +=1.0


	for x in range(0, grid_sz_x):
		for y in range(0, grid_sz_y):
			x1ref[x][y] = x1s[x][y]/Cnts[x][y]
			y1ref[x][y] = y1s[x][y]/Cnts[x][y]
			x2ref[x][y] = x2s[x][y]/Cnts[x][y]
			y2ref[x][y] = y2s[x][y]/Cnts[x][y]
			cvgref[x][y] = Cnts[x][y]/Cnts2[x][y]

	coord = np.where(cvgref > thr)

	y = np.asarray(coord[1])
	x = np.asarray(coord[0])

	boxes = []
	for i in range(x.size):
		boxes.append([x1ref[x[i],y[i]], y1ref[x[i],y[i]], x2ref[x[i],y[i]], y2ref[x[i],y[i]], cvgref[x[i],y[i]]])

	return boxes

def neighbours(b, M1, M2):
    lst = []
    try:
      for b_j_2 in range(M1.shape[1]):
          if M1[b][b_j_2] == 1:
            lst.append(M2[b_j_2])
            M1[b][b_j_2] = 0
            M1[b_j_2][b] = 0
            neighbour_ = neighbours(b_j_2, M1, M2)
            for nb in neighbour_:
                lst.append(nb)

    except RuntimeError: 
	#print "Runtime Error"
	return lst
    return lst

def union_rects(rects):

	B = np.zeros([len(rects),len(rects)])
	i = 0
	j = 0
	for b_dt in rects:
    		j = 0
    		for b_dt_2 in rects:
        		if b_dt[0] != b_dt_2[0] and b_dt[1] != b_dt_2[1] and b_dt[2] != b_dt_2[2] and b_dt[3] != b_dt_2[3]:
                		if IoU(b_dt, b_dt_2) >= 0.5:
                        		B[i][j] = 1
        		j = j+1
    		i = i + 1

	cov_iou = []
	for b_i in range(B.shape[0]):
    		tmp_lst = [rects[b_i]]
    		for b_j in range(B.shape[1]):
        		if B[b_i][b_j] == 1:
	       			tmp_lst.append(rects[b_j])
            			B[b_i][b_j] = 0
           			B[b_j][b_i] = 0
            			neighbour = neighbours(b_j, B, rects)
            			for n in neighbour:
                			tmp_lst.append(n)

        		else: pass 
    		if len(tmp_lst) > 1:
        		s_st_0 = 0
        		s_st_1 = 0
        		s_st_2 = 0
        		s_st_3 = 0
        		s_st_cov = 0
        		tmp_lst_clean = []

        		for el in tmp_lst:
            			if el not in tmp_lst_clean:
                			tmp_lst_clean.append(el)
            			else:
                			pass
        		for s_st in tmp_lst_clean:
            			s_st_0 = s_st_0 + s_st[0]
            			s_st_1 = s_st_1 + s_st[1]
            			s_st_2 = s_st_2 + s_st[2]
            			s_st_3 = s_st_3 + s_st[3]

            			s_st_cov = s_st_cov + s_st[4]
        		cov_iou.append([s_st_0/len(tmp_lst_clean), s_st_1/len(tmp_lst_clean), s_st_2/len(tmp_lst_clean), s_st_3/len(tmp_lst_clean), 
			s_st_cov, len(tmp_lst_clean)])
	return cov_iou

PRJ_MODEL = "./models/1024_640" #"./models/600_480"

PATH_TO_MODEL = PRJ_MODEL + "/face_detectnet_1024.caffemodel"

PATH_TO_DEPLOY = PRJ_MODEL + "/deploy.prototxt"

img_ext = ("jpg","jpeg","JPG","png","bmp")    
     
caffe.set_mode_gpu()  
#caffe.set_device(0)

INPUT_PATH = "/storage/wider-face/WIDER/WIDER_val"

if os.path.exists(PATH_TO_DEPLOY) and os.path.exists(PATH_TO_MODEL):
	net = caffe.Net(PATH_TO_DEPLOY, PATH_TO_MODEL, caffe.TEST)    
else:
	raise "Can\'t load file {} or {} for create net".format(PATH_TO_DEPLOY, PATH_TO_MODEL)

'''
INPUT_PATH = "/storage/wider-face/WIDER_val/images"
for p in os.listdir(INPUT_PATH):
	input_path_img = "/storage/wider-face/WIDER_val/images/" + str(p)
	
	file_for_work = [] 

	for image in os.listdir(input_path_img):
		if image.endswith(img_ext):
			path_img = os.path.join(input_path_img, image)
			name_file = getFileName(image)
			file_for_work.append([path_img, name_file])'''
 		      
file_for_work = []
for img in os.listdir(INPUT_PATH):

	if img.endswith(img_ext):
		path_img = os.path.join(INPUT_PATH, img)
		name_file = getFileName(img)
		file_for_work.append([path_img, name_file]) 
              
#load input and configure preprocessing
scale = 255.0
mean = 127.0

for i, file_set in enumerate(file_for_work):
	name = file_set[1]
	print "image", i+1, ' - ', name
	print "Wait..."

	im = Image.open(file_set[0])
		
	w = im.size[0]
	h = im.size[1]

	if w > h:
		k = 600.0/w
		im_res = im.resize((600, int(k*h)), PIL.Image.ANTIALIAS)
		w_res = im_res.size[0]
		h_res = im_res.size[1]	
	
	else:
		k = 480.0/h
		im_res = im.resize((int(k*w), 480), PIL.Image.ANTIALIAS)
		w_res = im_res.size[0]
		h_res = im_res.size[1]

	new_im = Image.new(im_res.mode, (600,480), (255,255,255))
	new_im.paste(im_res, (0,0,w_res,h_res))

	image_without_transformed =  img_as_float(new_im)

	image = image_without_transformed * scale - mean
	swap_rb = True
	if swap_rb:
		temp = np.array(image[:,:,0])
		image[:,:,0] = image[:,:,2]
		image[:,:,2] = temp
	net.blobs['data'].data[0] = image.transpose((2,0,1))
           
	out = net.forward()
	boxes = net.blobs['bboxes'].data[0]
	cvgs = net.blobs['coverage'].data[0]
		
	final_rects = process_boxes(cvgs, boxes, 600, 480, 8, 0.1)#!!!!! with clustering pickles

	bboxes = np.array(final_rects)
	bboxes = filter(lambda b: b.any() > 0, bboxes)
	bbox_dt = []

	for box in bboxes:
		if box.any() != 0:
			bbox_dt.append([float(box[0]/k), float(box[1]/k), float(box[2]/k), float(box[3]/k), float(box[4])])
	bbox_dt_f = filter(lambda b: b[0] > 0 and b[1] > 0 and b[2] > 0 and b[3] > 0, bbox_dt)

	final_rects = union_rects(bbox_dt_f)

	'''plt.ion()
	fig = plt.figure(i+1)
	plt.title(getFileName(name))
	plt.imshow(im) 
	currentAxis = plt.gca()
	for box_dt in final_rects:
		coords2 = (float(box_dt[0]), float(box_dt[1])), \
		float(box_dt[2]) - float(box_dt[0]), float(box_dt[3]) - float(box_dt[1]) 
		currentAxis.add_patch(plt.Rectangle(*coords2, fill = False, edgecolor = 'r', linewidth = 1))
	plt.draw()
	plt.waitforbuttonpress()
	plt.show()
	plt.close()'''

	fig = plt.figure(i+1)
	plt.title(getFileName(name))
	plt.imshow(im) 
	currentAxis = plt.gca()
	for box_dt in final_rects:
		coords2 = (float(box_dt[0]), float(box_dt[1])), \
		float(box_dt[2]) - float(box_dt[0]), float(box_dt[3]) - float(box_dt[1]) 
		currentAxis.add_patch(plt.Rectangle(*coords2, fill = False, edgecolor = 'r', linewidth = 1))
	plt.show()
	plt.close()
