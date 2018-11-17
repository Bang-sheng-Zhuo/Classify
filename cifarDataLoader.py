import numpy as np
import pickle
import os

class cifarDataLoader():
    def __init__(self):
        self.fine_label_names = None
        self.coarse_label_names = None
        self.X = None
        self.Y = None
        self.YY = None
        self.fine_classes = None
        self.coarse_classes = None

    def load_CIFAR_Data(self, filename):
        with open(filename, 'rb') as file:
            dataset = pickle.load(file, encoding='bytes')
            X = dataset[b'data']            # img data
            Y = dataset[b'fine_labels']     # fine label
            YY = dataset[b'coarse_labels']  # coarse label
            self.X = np.reshape(X, (-1, 3, 32, 32)).transpose(0, 2, 3, 1).astype("uint8")
            self.Y = np.array(Y)
            self.YY = np.array(YY)

    def load_CIFAR_labels(self, filename):
        with open(filename, 'rb') as file:
            labels = pickle.load(file, encoding='bytes')
            self.fine_label_names = labels[b'fine_label_names']
            self.fine_classes = len(self.fine_label_names)
            self.coarse_label_names = labels[b'coarse_label_names']
            self.coarse_classes = len(self.coarse_label_names)

    def load(self, imgfile, labelfile):
        self.load_CIFAR_Data(imgfile)
        self.load_CIFAR_labels(labelfile)

    def get_next_batch(self, batch_size, one_hot=True, is_fine=True):
        idx = np.random.choice(len(self.X), batch_size)
        # idx = range(batch_size)
        batch_x = self.X[idx]
        if one_hot:
            if is_fine:
                batch_y = self.Y[idx]
                batch_onehot = np.eye(self.fine_classes)[batch_y]
            else:
                batch_y = self.YY[idx]
                batch_onehot = np.eye(self.coarse_classes)[batch_y]
            return batch_x.astype(np.float32), batch_onehot.astype(np.float32)
        else:
            if is_fine:
                batch_y = batch_y = self.Y[idx]
            else:
                batch_y = self.YY[idx]
        return (batch_x.astype(np.float32), batch_y.astype(np.float32))

    def data_argumentation(self):
        batch_size, height, weight, channel = self.X.shape
        X = [self.X]
        Y = [self.Y]
        impulse_noise = self.X.copy()
        for i in range(batch_size):
            for j in range(150):
                x = np.random.randint(0,height)
                y = np.random.randint(0,weight)
                impulse_noise[i, x ,y ,:] = 255
        gaussian_noise = self.X.copy()
        for i in range(batch_size):
            noise = np.random.normal(0, 0.5, (height, weight, channel))
            gaussian_noise[i] = gaussian_noise[i] + noise
        # vertical flip
        vertical_flip = self.X[:,::-1,:,:]
        # horizontal flip
        horizontal_flip = self.X[:,:,::-1,:]
        # channel_shift
        channel_shift = self.X.copy()
        for i in range(batch_size):
            intensity = 0.05
            channel_index = 0
            x = np.rollaxis(channel_shift[i], channel_index, 0)
            min_x, max_x = np.min(x), np.max(x)
            channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x)
                            for x_channel in x]
            x = np.stack(channel_images, axis=0)
            x = np.rollaxis(x, 0, channel_index+1)
            channel_shift[i] = x
        # concat
        X.append(impulse_noise)
        Y.append(self.Y)
        X.append(gaussian_noise)
        Y.append(self.Y)
        X.append(vertical_flip)
        Y.append(self.Y)
        X.append(horizontal_flip)
        Y.append(self.Y)
        X.append(channel_shift)
        Y.append(self.Y)
        X = np.concatenate(X, axis=0)
        Y = np.concatenate(Y, axis=0)
        self.X = X
        self.Y = Y

    def show(self):
        example_nums = self.X.shape[0]
        for i in range(example_nums):
            img = self.X[i]
            print("fine_label:", self.fine_label_names[self.Y[i]],
                " coarse_label:", self.coarse_label_names[self.YY[i]])
            plt.imshow(img)
            plt.show()