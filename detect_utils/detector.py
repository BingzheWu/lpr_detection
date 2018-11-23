from __future__ import absolute_import
import numpy as np
import cv2
import skimage.io
import skimage.transform
import torch
import time
from os.path import splitext, basename
import sys
sys.path.append('./')

from detect_utils.label_utils import Label, Shape, writeShapes
from detect_utils.utils import getWH, nms, im2single, load_config
from detect_utils.projection_utils import getRectPts, find_T_matrix


class DLabel (Label):

	def __init__(self,cl,pts,prob):
		self.pts = pts
		tl = np.amin(pts,1)
		br = np.amax(pts,1)
		Label.__init__(self,cl,tl,br,prob)

def save_model(model,path,verbose=0):
	path = splitext(path)[0]
	model_json = model.to_json()
	with open('%s.json' % path,'w') as json_file:
		json_file.write(model_json)
	model.save_weights('%s.h5' % path)
	if verbose: print('Saved to %s' % path)

def load_keras_model(path,custom_objects={},verbose=0):
	from keras.models import model_from_json

	path = splitext(path)[0]
	with open('%s.json' % path,'r') as json_file:
		model_json = json_file.read()
	model = model_from_json(model_json, custom_objects=custom_objects)
	model.load_weights('%s.h5' % path)
	if verbose: print('Loaded from %s' % path)
	return model


def reconstruct(Iorig,I,Y,out_size,threshold=.9):

	net_stride 	= 2**4
	side 		= ((208. + 40.)/2.)/net_stride # 7.75

	Probs = Y[...,0]
	Affines = Y[...,2:]
	rx,ry = Y.shape[:2]
	ywh = Y.shape[1::-1]
	iwh = np.array(I.shape[1::-1],dtype=float).reshape((2,1))

	xx,yy = np.where(Probs>threshold)

	WH = getWH(I.shape)
	MN = WH/net_stride

	vxx = vyy = 0.5 #alpha

	base = lambda vx,vy: np.matrix([[-vx,-vy,1.],[vx,-vy,1.],[vx,vy,1.],[-vx,vy,1.]]).T
	labels = []

	for i in range(len(xx)):
		y,x = xx[i],yy[i]
		affine = Affines[y,x]
		prob = Probs[y,x]

		mn = np.array([float(x) + .5,float(y) + .5])

		A = np.reshape(affine,(2,3))
		A[0,0] = max(A[0,0],0.)
		A[1,1] = max(A[1,1],0.)

		pts = np.array(A*base(vxx,vyy)) #*alpha
		pts_MN_center_mn = pts*side
		pts_MN = pts_MN_center_mn + mn.reshape((2,1))

		pts_prop = pts_MN/MN.reshape((2,1))

		labels.append(DLabel(0,pts_prop,prob))
	final_labels = nms(labels,.1)
	TLps = []
	if len(final_labels):
		final_labels.sort(key=lambda x: x.prob(), reverse=True)
		for i,label in enumerate(final_labels):

			t_ptsh 	= getRectPts(0,0,out_size[0],out_size[1])
			ptsh 	= np.concatenate((label.pts*getWH(Iorig.shape).reshape((2,1)),np.ones((1,4))))
			H 		= find_T_matrix(ptsh,t_ptsh)
			Ilp 	= cv2.warpPerspective(Iorig,H,out_size,borderValue=.0)
			#affine = skimage.transform.AffineTransform(H)
			#Ilp = skimage.transform.warp(Iorig, affine, out_size)
			TLps.append(Ilp)

	return final_labels,TLps
	

def detect_lp(model,I,max_dim,net_step,out_size,threshold, mode = 'keras'):

	min_dim_img = min(I.shape[:2])
	factor 		= float(max_dim)/min_dim_img

	w,h = (np.array(I.shape[1::-1],dtype=float)*factor).astype(int).tolist()
	w += (w%net_step!=0)*(net_step - w%net_step)
	h += (h%net_step!=0)*(net_step - h%net_step)
	w,h = 428, 428
	Iresized = cv2.resize(I,(w,h))
	#Iresized = skimage.transform.resize(I, (w,h))
	#print(Iresized.shape)
	T = Iresized.copy()
	T = T.reshape((1,T.shape[0],T.shape[1],T.shape[2]))

	start 	= time.time()
	if mode == 'keras':
		print(T.shape)
		#model.input = T
		Yr = model.predict(T)
	elif mode == 'caffe':
		print(T.shape)
		T = T.transpose((0,3,1,2))
		print(T.shape)
		model.blobs['blob0'].data[...] = T
		print('start forward')
		Yr = model.forward()['cat_blob91']
		'''
		conv0 = model.blobs["batch_norm_blob2"].data
		print(conv0)
		'''
		Yr = Yr.transpose((0,2,3, 1))
	elif mode == 'torch':
		model.eval()
		T = torch.Tensor(T)
		T = T.permute((0,3,1,2))
		Yr = model(T)
		Yr = Yr.detach().numpy()
	Yr 		= np.squeeze(Yr)
	print(Yr.shape)
	print(Yr[0][0])
	elapsed = time.time() - start

	L,TLps = reconstruct(I,Iresized,Yr,out_size,threshold)
	return L,TLps,elapsed
def main():
    import sys
    import os
    import glob
    cfg_file = sys.argv[1]
    cfg  = load_config(cfg_file)
    mode = cfg.mode
    test_path = cfg.test_root
    results_path = os.path.join(test_path, 'results')
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    if mode == 'caffe':
        import caffe
        caffe_proto = cfg.proto_file
        caffe_model = cfg.weight_file
        model = caffe.Net(caffe_proto, caffe_model, caffe.TEST)
    if mode == 'torch':
        from models.lpr_net import lpr_model
        import torch
        model = lpr_model(16)
        state_dict = torch.load(os.path.join(cfg.log_dir, 'checkpoint.pth.tar'))['state_dict']
        model.load_state_dict(state_dict)
    imgs_path = glob.glob('%s/*.jpg'%test_path)
    for i, img_path in enumerate(imgs_path):
        print('\t Processing %s' % img_path)
        bname = splitext(basename(img_path))[0]
        img = cv2.imread(img_path)
        ratio = float(max(img.shape[:2])) / min(img.shape[:2])
        side = int(ratio*288.)
        bound_dim = min(side + (side%(2**4)), 608)
        Llp, LipImgs, _  = detect_lp(model, im2single(img), bound_dim, 2**4, (240, 80), 0.5, mode = mode)
        #Llp, LipImgs, _  = detect_lp(model, img, bound_dim, 2**4, (240, 80), 0.5, mode = mode)
        if len(LipImgs):
            Ilp = LipImgs[0]
            #Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
            #Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)
            s = Shape(Llp[0].pts)
            cv2.imwrite('%s/%s_lp.png' % (results_path,bname),Ilp*255.)
            writeShapes('%s/%s_lp.txt' % (results_path,bname),[s])
if __name__ == '__main__':
    main()