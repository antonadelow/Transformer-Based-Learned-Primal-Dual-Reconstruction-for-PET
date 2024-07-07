import numpy as np
import torch
import odl
from utils.torchProjectionLayer import LinearSingleChannelOperator 
from torchvision import transforms
from torch.utils.data import Dataset	

class RandomEllipsoids(Dataset):
    def __init__(self, proj, num_images, dim=2, diag=200, num_ellipsoids = 10, axes_scale = 0.4, use_negatives=True, negative_scale=0.5):
        self.proj = proj
        self.num_ellipsoids = num_ellipsoids
        self.diag = diag
        self.num_images = num_images
        self.axes_scale = axes_scale
        self.use_negatives = use_negatives
        self.negative_scale = negative_scale
        self.dim = dim

        diag_ceil = int(np.ceil(self.diag))
        if self.dim==2:
            self.space = odl.uniform_discr([0]*2, [diag_ceil]*2, [diag_ceil]*2, dtype='float32')
            self.axes_len = (proj.in_shape[0], proj.in_shape[2])
        elif self.dim==3:
            self.space = odl.uniform_discr([0]*3, [diag_ceil]*3, [diag_ceil]*3, dtype='float32')
            self.axes_len = proj.in_shape
        else:
            raise ValueError('Only 2 or 3 dimensions is suppoerted')

    def __getitem__(self, _=None):
        if self.dim==2:
            item = torch.zeros(self.proj.in_shape)
            for i in range(self.proj.in_shape[1]):
                item[:,i,:] = self.generate_ellipsoids()
        else:
            item = self.generate_ellipsoids()

        return item

    def __len__(self):
        return self.num_images

    def generate_ellipsoids(self):
        num = np.random.poisson(self.num_ellipsoids)
        value = np.random.rand(num)
        angles = np.random.rand(1 if self.dim==2 else 3, num)*2*np.pi
        centers = np.random.rand(self.dim, num) - 0.5
        axes = np.random.exponential(scale=self.axes_scale, size=(self.dim, num))

        ellipsoid_params = np.vstack((value, axes, centers, angles)).T
        ellipsoids = odl.phantom.geometric.ellipsoid_phantom(self.space, ellipsoid_params).asarray()

        if self.use_negatives:
            value = np.random.rand(num)
            angles = np.random.rand(1 if self.dim==2 else 3, num)*2*np.pi
            centers = np.random.rand(self.dim, num) - 0.5
            axes = np.random.exponential(scale=self.axes_scale*self.negative_scale, size=(self.dim, num))

            ellipsoid_params = np.vstack((value, axes, centers, angles)).T
            negative_ellipsoids = -odl.phantom.geometric.ellipsoid_phantom(self.space, ellipsoid_params).asarray()
            ellipsoids = np.maximum(ellipsoids + negative_ellipsoids, 0)

        start_idx = [(ellipsoids.shape[i] - self.axes_len[i]) // 2 for i in range(self.dim)]
        end_idx = [start_idx[i] + self.axes_len[i] for i in range(self.dim)]

        center = torch.tensor(ellipsoids[tuple(slice(start_idx[i], end_idx[i]) for i in range(self.dim))])
        
        if torch.max(center) != 0:
            center = center/torch.max(center)
        
        return center

def generate_data(images, operator, noise_level=1.):
    fwd_op_layer = LinearSingleChannelOperator.apply
    data = fwd_op_layer(images.unsqueeze(1),operator)
    noisy_data = torch.poisson(data/noise_level)*noise_level
    return noisy_data


def blur_image(img, kernel_size=3, sigma=1.7):
    blurred_img = torch.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[2]):
            blurred_img[i,:,j,:] = transforms.functional.gaussian_blur(img[i,:,j,:].unsqueeze(0), kernel_size=kernel_size, sigma=sigma).squeeze(0)

    return blurred_img