import parallelproj
import torch
import numpy as np
import array_api_compat.torch as xp
from array_api_compat import to_device
from torchProjectionLayer import *
import torch.nn.functional as F


def get_minipet_projector(dev,num_rings = 35):
    scanner = parallelproj.RegularPolygonPETScannerGeometry(
        xp,
        dev,
        radius= 211/2 - 2,
        num_sides=12,
        num_lor_endpoints_per_side= 35,
        lor_spacing= 211*np.pi/(35*12),
        ring_positions=torch.linspace(-20, 20, num_rings),
        symmetry_axis=1,
    )

    lor_desc = parallelproj.RegularPolygonPETLORDescriptor(
        scanner,
        radial_trim=155,
        max_ring_difference=0,
        sinogram_order=parallelproj.SinogramSpatialAxisOrder.RVP,
    )


    proj = parallelproj.RegularPolygonPETProjector(
        lor_desc, img_shape=(147, num_rings, 147), voxel_size=(80/147, 40/num_rings, 80/147)
    )

    return proj

def to_2D(proj):
    scanner = parallelproj.RegularPolygonPETScannerGeometry(
    xp,
    'cuda',
    radius=proj.lor_descriptor.scanner.__getattribute__('radius'),
    num_sides=proj.lor_descriptor.scanner.__getattribute__('num_sides'),
    num_lor_endpoints_per_side= proj.lor_descriptor.scanner.__getattribute__('num_lor_endpoints_per_side'),
    lor_spacing=proj.lor_descriptor.scanner.__getattribute__('lor_spacing'),
    ring_positions=torch.linspace(0, 0, 1),
    symmetry_axis=proj.lor_descriptor.scanner.__getattribute__('symmetry_axis'),
    )

    lor_desc = parallelproj.RegularPolygonPETLORDescriptor(
        scanner,
        radial_trim=proj.lor_descriptor.__getattribute__('radial_trim'),
        max_ring_difference=proj.lor_descriptor.__getattribute__('max_ring_difference'),
        sinogram_order=proj.lor_descriptor.__getattribute__('sinogram_order'),
    )
    
    proj = parallelproj.RegularPolygonPETProjector(
        lor_desc, img_shape=(proj.in_shape[0], 1, proj.in_shape[2]), voxel_size=proj.__getattribute__('voxel_size')
    )
    return proj

def get_indices(proj,s=0):
    proj = to_2D(proj)
    image = to_device(torch.ones(proj.in_shape), 'cuda')
    indices = (proj(image) > s).squeeze(-1).nonzero()

    return indices

def patch_sinogram(proj, patch_size=(3,3), s=0, return_values=False):
    proj = to_2D(proj)
    image_shape = proj.in_shape

    patch_dim = (int(np.ceil(image_shape[0]/patch_size[0])), int(np.ceil(image_shape[2]/patch_size[1])))
    images = to_device(torch.zeros(patch_dim[0]*patch_dim[1], image_shape[0], 1, image_shape[2]), 'cuda')
    indices = []
    values = []
    num_patches = patch_dim[0]*patch_dim[1]
    for k in range(num_patches):
        i = k // patch_dim[1]
        j = k % patch_dim[1]
        images[k, i*patch_size[0]:(i+1)*patch_size[0], 0, j*patch_size[1]:(j+1)*patch_size[1]] = 1
        sino = proj(images[k])
        index = (sino > s).squeeze(-1).nonzero()
        indices.append(index)

    lengths = [len(tensor) for tensor in indices]
    max_length = max(lengths)

    padded_tensors = []
    for tensor in indices:
        pad_size = max_length - len(tensor)
        padded_tensor = F.pad(tensor, (0, 0, pad_size, 0))
        padded_tensors.append(padded_tensor)
        values.append(sino[padded_tensor[:,0],padded_tensor[:,1],0]/torch.sum(sino[padded_tensor[:,0],padded_tensor[:,1],0]))

    stacked_tensor = torch.stack(padded_tensors)
    stacked_values = torch.stack(values)
    if return_values:
        stacked_tensor = torch.cat((stacked_tensor,stacked_values.unsqueeze(-1)),dim=2)
        
    return stacked_tensor