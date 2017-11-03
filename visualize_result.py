#!/usr/bin/env python
import argparse

import numpy as np
import matplotlib.pyplot as plt

import os

caffe = "_"  # this file is expected to be in {caffe_root}/python
import sys
sys.path.insert(0, caffe)
import caffe

from skimage import io, img_as_float, transform

import PIL
from PIL import Image

import Tkinter as tk

def getFileName(pathOfFile):
	base = os.path.basename(pathOfFile)
	return os.path.splitext(base)[0]    

def clamp(z, min_v, max_v):
	if(z < min_v):
		return min_v
	if(z > max_v):
		return max_v
	return z

def IoU(gt_rect, dt_rect):
	dt_area = float(abs((dt_rect[2] - dt_rect[0])*(dt_rect[3] - dt_rect[1])))
	gt_area = float(abs((gt_rect[2] - gt_rect[0])*(gt_rect[3] - gt_rect[1])))

	x_overlap = max(0, min(dt_rect[2], gt_rect[2]) - max(dt_rect[0], gt_rect[0]))
	y_overlap = max(0, min(dt_rect[3], gt_rect[3]) - max(dt_rect[1], gt_rect[1]))

	intersect = x_overlap * y_overlap    

	union = gt_area + dt_area - intersect
	if (intersect == 0.0 or union == 0.0):
		return 0.0

	if (union == 0.0):
		return 0.0

	IoU = intersect / union
	return IoU    

def process_boxes(net_cvg, net_boxes, im_sz_x, im_sz_y, stride, thr, itCNT = 3, thr_IOU = 0.4):
	grid_sz_x = int(im_sz_x / stride)
	grid_sz_y = int(im_sz_y / stride)

	cell_width = im_sz_x / grid_sz_x
	cell_height = im_sz_y / grid_sz_y

	rectz = net_boxes

	x1ref = np.zeros([grid_sz_x, grid_sz_y])
	x2ref = np.zeros([grid_sz_x, grid_sz_y])
	y1ref = np.zeros([grid_sz_x, grid_sz_y])
	y2ref = np.zeros([grid_sz_x, grid_sz_y])
	cvgref = np.zeros([grid_sz_x, grid_sz_y])
	cvg_val = np.zeros([grid_sz_y, grid_sz_x])

	for x in range(0, grid_sz_x):
		for y in range(0, grid_sz_y):
			cvg_val[y][x] = net_cvg[0][y][x]
			x1ref[x][y] = rectz[0][y][x] + x * cell_width
			y1ref[x][y] = rectz[1][y][x] + y * cell_height
			x2ref[x][y] = rectz[2][y][x] + x * cell_width
			y2ref[x][y] = rectz[3][y][x] + y * cell_height
			cvgref[x][y] = cvg_val[y][x] #y x

	for n in range(0, itCNT):
		x1s = np.zeros([grid_sz_x, grid_sz_y])
		y1s = np.zeros([grid_sz_x, grid_sz_y])
		x2s = np.zeros([grid_sz_x, grid_sz_y])
		y2s = np.zeros([grid_sz_x, grid_sz_y])
		Cnts = np.ones([grid_sz_x, grid_sz_y])
		Cnts2 = np.ones([grid_sz_x, grid_sz_y])

		for x in range(0, grid_sz_x):
			for y in range(0, grid_sz_y):
				if cvgref[x][y] > thr:
					x_beg = x1ref[x][y]/cell_width
					y_beg = y1ref[x][y]/cell_height
					x_end = x2ref[x][y]/cell_width
					y_end = y2ref[x][y]/cell_height
					x_beg = int(clamp(x_beg, 0, grid_sz_x - 1))
					y_beg = int(clamp(y_beg, 0, grid_sz_y - 1))
					x_end = int(clamp(x_end, 0, grid_sz_x - 1))
					y_end = int(clamp(y_end, 0, grid_sz_y - 1))
					x1s[x][y] += x1ref[x][y]*cvgref[x][y]
					y1s[x][y] += y1ref[x][y]*cvgref[x][y]
					x2s[x][y] += x2ref[x][y]*cvgref[x][y]
					y2s[x][y] += y2ref[x][y]*cvgref[x][y]
					Cnts[x][y] = cvgref[x][y]

					for x_i in range(x_beg, x_end+1):
						for y_i in range(y_beg, y_end+1):
							if cvgref[x_i][y_i] > thr:
								IOU = IoU([x1ref[x][y], y1ref[x][y], x2ref[x][y], y2ref[x][y]], 
								[x1ref[x_i][y_i], y1ref[x_i][y_i], x2ref[x_i][y_i], y2ref[x_i][y_i]])
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
		boxes.append([x1ref[x[i], y[i]], y1ref[x[i],y[i]], x2ref[x[i],y[i]], y2ref[x[i], y[i]], cvgref[x[i],y[i]]])

	return boxes

def neighbours(neighb):
	def findRoot(node, root):
		while node != root[node][0]:
			node = root[node][0]
		return (node, root[node][1])
	root_dict = {} 
	for nd in neighb.keys():
		root_dict[nd] = (nd, 0)  
	for i in neighb: 
		for j in neighb[i]: 
			(root_i, depth_i) = findRoot(i, root_dict) 
			(root_j, depth_j) = findRoot(j, root_dict) 
			if root_i != root_j: 
				min_ = root_i
				max_ = root_j 
				if  depth_i > depth_j: 
					min_ = root_j
					max_ = root_i
				root_dict[max_] = (max_, max(root_dict[min_][1] + 1, root_dict[max_][1]))
				root_dict[min_] = (root_dict[max_][0], -1) 
	dict_result = {}
	for i in neighb: 
		if root_dict[i][0] == i:
			dict_result[i] = []
	for i in neighb: 
		dict_result[findRoot(i, root_dict)[0]].append(i) 
	return dict_result

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



	###############
	dict_ind = {}
	#
	for b_i in range(B.shape[0]):
		lst_index = []
		for b_j in range(B.shape[1]):
			if B[b_i][b_j] == 1:
				lst_index.append(b_j)
				B[b_i][b_j] = 0
           			B[b_j][b_i] = 0
		dict_ind.update({b_i: lst_index})

	cov_iou = []
	
	f_rect = neighbours(dict_ind)
	for l in f_rect.keys(): 
		tmp_lst = []
		for iii in dict_ind.get(l): 
			
			tmp_lst.append(rects[iii])
	
		
	##############
    		if len(tmp_lst) > 1:
			avrg_x1 = []
        		avrg_y1 = []
        		avrg_x2 = []
        		avrg_y2 = []
        		avrg_cvg = 0
        		tmp_lst_clean = []

        		for el in tmp_lst:
            			if el not in tmp_lst_clean:
                			tmp_lst_clean.append(el)
            			else:
                			pass
        		for tlc in tmp_lst_clean:
				avrg_x1.append(tlc[0])
            			avrg_y1.append(tlc[1])
            			avrg_x2.append(tlc[2])
            			avrg_y2.append(tlc[3])

            			avrg_cvg = avrg_cvg + tlc[4]

			cov_iou.append([min(avrg_x1), min(avrg_y1), max(avrg_x2), max(avrg_y2), avrg_cvg, len(tmp_lst_clean)])
	return cov_iou

def main(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument(
	"--deploy_file",
	default = "",
	help = "Filename of deploy archictecture net *.prototxt."
	)
	parser.add_argument(
	"--model_file",
	default = "",
	help = "Model definition file *.caffemodel."
	)
	parser.add_argument(
	"--size",
	default = "600,480",
	help = "Canonical 'height,width' dimensions of input images."
	)
	parser.add_argument(
	"--mode",
	default = "reshape",
	help = "Mode to choose algorithm with reshape net or resize image: 'reshape'/'resize'."
	)
	parser.add_argument(
	"--threshold",
	default = 4.,
	help = "Threshold to separate false face."
	)
	parser.add_argument(
	"--input",
	help = "Folder for make forward proccess images and detect faces."
	)
	parser.add_argument(
	"--gpu",
	default = "",
	help = "Switch for gpu computation."
	)

	args = parser.parse_args()

	if args.deploy_file:
		PATH_TO_DEPLOY = args.deploy_file

	if args.model_file:
		PATH_TO_MODEL = args.model_file

	if args.size:
		IMAGE_SIZE = args.size
		IMAGE_SIZE = IMAGE_SIZE.split(",")

		IMAGE_WIDTH = int(IMAGE_SIZE[0])
		IMAGE_HEIGHT = int(IMAGE_SIZE[1])

	if args.mode:
		MODE_DETECTOR = args.mode

	if args.threshold:
		THRSHLD = args.threshold

	if args.input:
		INPUT_PATH = args.input

	if args.gpu:
		caffe.set_mode_gpu() 
		#caffe.set_device(1) #

	img_ext = ("jpg","jpeg","JPG","png","bmp")
	
	if os.path.exists(PATH_TO_DEPLOY) and os.path.exists(PATH_TO_MODEL):
		net = caffe.Net(PATH_TO_DEPLOY, PATH_TO_MODEL, caffe.TEST) 
	else:
		raise "Can\'t load file {} or {} for create net".format(PATH_TO_DEPLOY, PATH_TO_MODEL)
      
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
		path_to_img = file_set[0]
		name = file_set[1]
		print "Image", path_to_img, "processing..."
	
		im = Image.open(file_set[0])
		w = im.size[0]
		h = im.size[1]

		if MODE_DETECTOR == "resize":
			if w > h:
				k = float(IMAGE_WIDTH)/w
				im_res = im.resize((IMAGE_WIDTH, int(k*h)), PIL.Image.ANTIALIAS)
				w_res = im_res.size[0]
				h_res = im_res.size[1]	
	
			else:
				k = float(IMAGE_HEIGHT)/h
				im_res = im.resize((int(k*w), IMAGE_HEIGHT), PIL.Image.ANTIALIAS)
				w_res = im_res.size[0]
				h_res = im_res.size[1]

			new_im = Image.new(im_res.mode, (IMAGE_WIDTH,IMAGE_HEIGHT), (255,255,255))
			new_im.paste(im_res, (0,0,w_res,h_res))

			image_without_transformed =  img_as_float(new_im)
		else:
			
			#for control out_of_memory_error
			if h > w and h >= 1200:
				k = 1200.0/h
				im = im.resize((int(k*w), 1200), PIL.Image.ANTIALIAS)
				w = im.size[0]
				h = im.size[1]

			image_without_transformed =  img_as_float(im)

		image = image_without_transformed * scale - mean
		swap_rb = True
		if swap_rb:
			temp = np.array(image[:,:,0])
			image[:,:,0] = image[:,:,2]
			image[:,:,2] = temp

		if MODE_DETECTOR == "resize":
			net.blobs['data'].data[0] = image.transpose((2, 0, 1))
		else:
			net.blobs['data'].reshape(1, 3, h, w)
			net.blobs['data'].data[0] = image.transpose((2, 0, 1))
           
		out = net.forward()
		boxes = net.blobs['bboxes'].data[0]
		cvgs = net.blobs['coverage'].data[0]

		if MODE_DETECTOR == "resize":
			final_rects = process_boxes(cvgs, boxes, IMAGE_WIDTH, IMAGE_HEIGHT, 8, 0.1)
		else:
			final_rects = process_boxes(cvgs, boxes, w, h, 8, 0.1, 1)

		bboxes = np.array(final_rects)
	
		bboxes = filter(lambda b: b.any() > 0, bboxes)
		bbox_dt = []

		for box in bboxes:
			if box.any() != 0:
				if MODE_DETECTOR == "resize":
					bbox_dt.append([float(box[0]/k), float(box[1]/k), float(box[2]/k), float(box[3]/k), float(box[4])])
				else:

					bbox_dt.append([float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(box[4])])
		bbox_dt_f = filter(lambda b: b[0] > 0 and b[1] > 0 and b[2] > 0 and b[3] > 0, bbox_dt)
		
		final_rects = union_rects(bbox_dt_f)

		try:
			plt.ion()
			plt.figure(i+1)
			plt.title("Image " + path_to_img)
			plt.imshow(im) 
			currentAxis = plt.gca()
			for box_dt in final_rects:
				if box_dt[4] > float(THRSHLD):
					coords2 = (float(box_dt[0]), float(box_dt[1])), \
					float(box_dt[2]) - float(box_dt[0]), float(box_dt[3]) - float(box_dt[1]) 
					currentAxis.add_patch(plt.Rectangle(*coords2, fill = False, edgecolor = 'r', linewidth = 1))
			plt.draw()
			plt.waitforbuttonpress()
			plt.close(i+1)
		except tk.TclError: break


if __name__ == '__main__':
	main(sys.argv)
