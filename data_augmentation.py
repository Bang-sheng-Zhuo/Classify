# data argumentation
import pdb
import numpy as np
import tensorflow as tf
import skimage
import cv2
from skimage.transform import rotate
from skimage import color, io

keras = tf.keras
image_process = keras.preprocessing.image

def img_show(img, name='test'):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def one_hot_encode(label, nums_classes):
    onehot = np.eye(nums_classes)[label]
    return onehot.astype(np.float32)

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

def cv2_clipped_zoom(img, zoom_factor):
    """
    Center zoom in/out of the given image and returning an enlarged/shrinked view of 
    the image without changing dimensions
    Args:
        img : Image array
        zoom_factor : amount of zoom as a ratio (0 to Inf)
    """
    # pdb.set_trace()
    height, width = img.shape[:2] # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1,x1,y2,x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(np.int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0,0)] * (img.ndim - 2)

    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.expand_dims(result, axis=-1)
    result = np.pad(result, pad_spec, mode='constant')
    assert result.shape[0] == height and result.shape[1] == width
    return result

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

def random_crop(img, padding=4, is_flip=True, prob=0.5, is_crop=True):
    height, width, depth = img.shape
    pad_shape = ((padding,padding),(padding,padding),(0,0))
    flag = np.random.rand(1)
    if is_flip and flag>=prob:
        flip = horizontal_flip(img)
    else:
        flip = img
    if is_crop:
        [left, top] = np.random.randint(low=0, high=2*padding, size=2, dtype=np.uint8)
        flip_pad = np.pad(flip, pad_shape, 'constant', constant_values=0)
        crop = flip_pad[left:left+height, top:top+width, :]
        return crop
    else:
        return flip

    
# data augmentation for conv_lstm 
def img_padding(X, max_len):
    batch_x = []
    _, h, w, d = X[0].shape
    for data in X:
        tmp_x = np.zeros([max_len-data.shape[0], h, w, d], dtype=np.int32)
        batch_x.append(np.concatenate([data, tmp_x], axis=0))
    batch_x = np.asarray(batch_x)
    return batch_x

def label_padding(Y, max_len, classes):
    batch_y = []
    for y in Y:
        tmp_y = one_hot_encode(y, classes)
        y_pad = np.zeros([max_len-tmp_y.shape[0], classes], dtype=np.float32)
        batch_y.append(np.concatenate([tmp_y, y_pad], axis=0))
    return np.concatenate(batch_y, axis=0)
    
def next_batch(X, Y, nums_frames, classes, batch_size=6, padding=True):
    idx = np.random.choice(len(nums_frames), batch_size)
    tmp_x = [X[i] for i in idx]
    batch_y = [Y[i] for i in idx]
    seq = [nums_frames[i] for i in idx]
    seq = np.array(seq).astype(np.int32)
    max_len = np.max(seq)
    if padding:
        batch_x = img_padding(tmp_x, max_len)
    else:
        batch_x = np.concatenate(tmp_x, axis=0)
    batch_y = label_padding(batch_y, max_len, classes)
    # pdb.set_trace()
    return batch_x, batch_y, seq, max_len

def batch_augmentation_1(X, seq,
                         rotation_range=None,
                         flip_prob=None,
                         shift_range=None,
                         zoom_range=None
                        ):
    for i in range(X.shape[0]):
        # if rotation_range is not None:
        theta = np.random.uniform(-rotation_range, rotation_range)
        # if flip_prob is not None and np.random.rand(1) > flip_prob:
        flag = np.random.rand(1)
        # if shift_range is not None:
        wshift = np.random.uniform(-shift_range, shift_range)
        hshift = np.random.uniform(-shift_range, shift_range)
        # if zoom_range is not None:
        zoom_factor = np.random.uniform(zoom_range[0], zoom_range[1])
        for j in range(seq[i]):
            X[i,j,:,:,:] = rotate(X[i,j,:,:,:], theta)
            if flag > flip_prob:
                X[i,j,:,:,:] = X[i,j,:,::-1,:]
            X[i,j,:,:,:] = shift(X[i,j,:,:,:], wshift, hshift)
            cv2_clipped_zoom(X[i,j,:,:,:], zoom_factor)
    return X
            
if __name__ == '__main__':
    theta = np.pi / 180 * np.random.uniform(-50, 50)
    img = io.imread('/home/zbs/jpg_data/all_data/1/1_93.jpg')
    wshift = 0.2
    hshift = 0.2
    pdb.set_trace()
