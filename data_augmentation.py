# data argumentation
import pdb
import numpy as np
import tensorflow as tf
import skimage
import cv2
from skimage.transform import rotate
from skimage import color

keras = tf.keras
image_process = keras.preprocessing.image

def add_impulse_noise(img, height, weight):
	randint = int(height*weight/5)
	for j in range(randint):
		x = np.random.randint(0,height)
		y = np.random.randint(0,weight)
		img[i, x ,y ,:] = 255
	return img

def add_gaussian_noise(img, mean, var, shape):
	noise = np.random.normal(mean, var, shape)
	return img + noise

def vertical_flip(img):
	img = img[::-1,:,:]
	return img

def horizontal_flip(img):
	img = img[:,::-1,:]
	return img

# rotate_limit=(-30, 30)
# theta = np.pi / 180 * np.random.uniform(rotate_limit[0], rotate_limit[1]) #逆时针旋转角度
# img_rot = rotate(img, theta)
# imshow(img_rot)
def rotatation(img, theta, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0):
	rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
								[np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
	h, w = img.shape[row_axis], img.shape[col_axis]
	transform_matrix = image_process.transform_matrix_offset_center(rotation_matrix, h, w)
	img = image_process.apply_transform(img, transform_matrix, channel_axis, fill_mode, cval)
	return img

# w_limit=(-0.2, 0.2)
# h_limit=(-0.2, 0.2)
# wshift = np.random.uniform(w_limit[0], w_limit[1])
# hshift = np.random.uniform(h_limit[0], h_limit[1])
# img_shift = shift(img, wshift, hshift)
# imshow(img_shift)
def shift(img, wshift, hshift, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
	h, w = img.shape[row_axis], img.shape[col_axis] #读取图片的高和宽
	tx = hshift * h #高偏移大小，若不偏移可设为0，若向上偏移设为正数
	ty = wshift * w #宽偏移大小，若不偏移可设为0，若向左偏移设为正数
	translation_matrix = np.array([[1, 0, tx],
								  [0, 1, ty],
								  [0, 0, 1]])
	transform_matrix = translation_matrix  
	img = image_process.apply_transform(img, transform_matrix, channel_axis, fill_mode, cval)
	return img

# zoom_range=(0.7, 1)
# zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
# img_zoom = zoom(img, zx, zy)
# imshow(img_zoom)
def zoom(img, zx, zy, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
	zoom_matrix = np.array([[zx, 0, 0],
							[0, zy, 0],
							[0, 0, 1]])
	h, w = img.shape[row_axis], img.shape[col_axis]
	transform_matrix = image_process.transform_matrix_offset_center(zoom_matrix, h, w) #保持中心坐标不改变
	img = image_process.apply_transform(img, transform_matrix, channel_axis, fill_mode, cval)
	return img

# intensity = 0.5
# sh = np.random.uniform(-intensity, intensity) #逆时针方向剪切强度为正
# img_shear = shear(img, sh)
# imshow(img_shear)
def shear(img, shear, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
	shear_matrix = np.array([[1, -np.sin(shear), 0],
							[0, np.cos(shear), 0],
							[0, 0, 1]])
	h, w = img.shape[row_axis], img.shape[col_axis]
	transform_matrix = image_process.transform_matrix_offset_center(shear_matrix, h, w)
	img = image_process.apply_transform(img, transform_matrix, channel_axis, fill_mode, cval)
	return img

def contrast(image, hue_shift_limit=(-180, 180),
			sat_shift_limit=(-255, 255),
			val_shift_limit=(-255, 255), u=0.5):
	if np.random.random() < u:
		img = color.rgb2hsv(image)
		h, s ,v = img[:,:,0],img[:,:,1],img[:,:,2]
		hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])

		h = h + hue_shift

		sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
		s = s + sat_shift

		val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
		v = v + val_shift

		img[:,:,0],img[:,:,1],img[:,:,2] = h, s ,v

		image = color.hsv2rgb(img)
	return image

def channel_shift(img, intensity=0.05, channel_index=0):
	x = np.rollaxis(channel_shift[i], channel_index, 0)
	min_x, max_x = np.min(x), np.max(x)
	channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x)
					for x_channel in x]
	x = np.stack(channel_images, axis=0)
	x = np.rollaxis(x, 0, channel_index+1)
	return x

def rgb_pca(img):
	
	return img

def rgb_mean(batch_data):
	per_img_r = []
	per_img_g = []
	per_img_b = []
	batch_size = np.shape(batch_img)[0]
	for i in range(batch_size):
		per_img_r.append(np.mean(batch_img[i,:,:,0]))
		per_img_g.append(np.mean(batch_img[i,:,:,1]))
		per_img_b.append(np.mean(batch_img[i,:,:,2]))
	r_mean = np.mean(per_img_r)
	g_mean = np.mean(per_img_g)
	b_mean = np.mean(per_img_b)
	batch_img[:,:,:,0] -= r_mean
	batch_img[:,:,:,1] -= g_mean
	batch_img[:,:,:,2] -= b_mean
	return batch_img

def random_erase(img, prob=0.5, min=0.1, max=0.3):
	# cv2.imshow('original', img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	flag = np.random.rand(1)
	height, width, depth = img.shape
	if flag>=prob:
		rand_h = np.random.rand(1)*(max-min)+min
		rand_h = int(rand_h*height)
		rand_w = np.random.rand(1)*(max-min)+min
		rand_w = int(rand_w*width)
		h_begin = int(np.random.randint(low=0, high=height-rand_h+1, dtype=np.uint8))
		w_begin = int(np.random.randint(low=0, high=width-rand_w+1, dtype=np.uint8))
		value = np.random.randint(low=0, high=256, dtype=np.uint8)
		img[h_begin:h_begin+rand_h,w_begin:w_begin+rand_w,:] = value
		# cv2.imshow('erase', img)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		return img
	else:
		return img

if __name__ == '__main__':
	img = np.random.randint(low=0, high=256, size=(100,100,3), dtype=np.uint8)
	random_erase(img, prob=0., min=0.2, max=0.5)
