import torch
import numpy as np
from odl.contrib import fom
import numpy as np
from array_api_compat import to_device
from data import *
from einops import rearrange
from geometry import *
from odl.contrib import fom
import torch.nn as nn
import matplotlib.pyplot as plt
import nibabel
import matplotlib.gridspec as gridspec


def get_SLP(dev):
    rs=odl.uniform_discr([  0. , -21.9, -21.9], [  0.675,  22.2  ,  22.2  ], (147, 147, 60), dtype='float32')
    SLP = odl.phantom.shepp_logan(rs,modified=True)

    SLP_slices = torch.flip(rearrange(to_device(
        torch.tensor(SLP),
        dev), 'w h d -> h d w')[:,13:48,:].unsqueeze(0),[1])

    SLP_slices[SLP_slices < 0] = 0

    return SLP_slices

def get_derenzo_sources(dev):
    rs=odl.uniform_discr([  0. , -21.9, -21.9], [  0.675,  22.2  ,  22.2  ], (147, 147, 60), dtype='float32')
    SLP = odl.phantom.emission.derenzo_sources(rs)

    SLP_slices = torch.flip(rearrange(to_device(
        torch.tensor(SLP),
        dev), 'w h d -> h d w')[:,13:48,:].unsqueeze(0),[1])

    SLP_slices[SLP_slices < 0] = 0

    return SLP_slices

def evaluate(model, proj, dev, noise_level, n=3, loss_function = nn.MSELoss(), images=None):
    ssim_avg_list = []
    psnr_avg_list = []
    loss_avg_list = []

    d = proj.in_shape[1]

    if images is None:
        images = get_SLP(dev)

    for i in range(n):
        blurred_slices = blur_image(images, kernel_size=5, sigma=2.0)
        blurred_slices = rearrange(blurred_slices, 'b h (d1 d2) w -> (b d1) h d2 w', d2=d)
        images = rearrange(images, 'b h (d1 d2) w -> (b d1) h d2 w', d2=d)

        loganSin = generate_data(blurred_slices,proj,noise_level=noise_level)

        ssim_list = []
        psnr_list = []
        loss = 0

        with torch.no_grad():
            sino, images = to_device(loganSin,dev), to_device(images,dev)
            outputs = model(sino)
            loss += loss_function(outputs, images).item()
            for b in range(images.shape[0]):
                for i in range(images.shape[2]):
                    psnr_list.append(fom.psnr(outputs[b,:,i,:].cpu().numpy(),images[b,:,i,:].cpu().numpy()))
                    ssim_list.append(fom.ssim(outputs[b,:,i,:].cpu(),images[b,:,i,:].cpu()))

        psnr_avg_list.append(np.mean(psnr_list))
        ssim_avg_list.append(np.mean(ssim_list))
        loss_avg_list.append(loss)
    
    mean_ssim = np.mean(ssim_avg_list)
    mean_psnr = np.mean(psnr_avg_list)
    mean_loss = np.mean(loss)

    return mean_ssim, mean_psnr, mean_loss

def plot_reconstructed_images(model, proj, dev, noise_level, images=None, idx=16, legend=True):

    d = proj.in_shape[1]

    if images is None:
        images = get_SLP(dev)

    blurred_slices = blur_image(images, kernel_size=5, sigma=2.0)
    blurred_slices = rearrange(blurred_slices, 'b h (d1 d2) w -> (b d1) h d2 w', d2=d)
    
    data = generate_data(blurred_slices,proj,noise_level=noise_level)
    with torch.no_grad():
        data = to_device(data, dev)
        outputs = model(data)

    outputs = rearrange(outputs, '(b d1) h d2 w -> b h (d1 d2) w', b=1)
    
    fig = plt.figure()

    im = plt.imshow(outputs.cpu().numpy()[0,:,idx,:], cmap='bone',  clim=[0,1.2])
    plt.axis('off')

    if legend:
        cbar_ax = fig.add_axes([0.85, 0.17, 0.02, 0.65])
        fig.colorbar(im, cax=cbar_ax) 

    plt.show()

def plot_miniPET(model, proj, dev, path='/home/mamo_alegua/anaconda3/ThesisCodeminiPET/Mouse4/sino40min.sino.mnc', indices= [70,80,16], all=False, legend=False):
    data = torch.Tensor(nibabel.load(path).get_fdata())
    
    transform=transforms.Resize((111, 210))

    sino = transform(data)

    d = proj.in_shape[1]

    sino = to_device(rearrange(sino,'(d1 d2) h w -> d1 h w d2', d2=d).unsqueeze(1), dev)
    
    sino[sino < 0] = 0
    with torch.no_grad():
        outputs = model(sino)

    outputs = rearrange(outputs, '(b d1) h d2 w -> b h (d1 d2) w', b=1)

    if not all:
        fig = plt.figure()

        gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 4.2, 0.1])

        ax0 = plt.subplot(gs[0])
        im0 = ax0.imshow(outputs[0,:,:,indices[0]].cpu().numpy(), cmap='bone',clim=[0,1.5])
        ax0.axis('off')

        ax1 = plt.subplot(gs[1])
        im1 = ax1.imshow(np.flip(outputs[0,indices[1],:,:].cpu().numpy().T), cmap='bone',clim=[0,1.5])
        ax1.axis('off')

        ax2 = plt.subplot(gs[2])
        im2 = ax2.imshow(outputs[0,:,indices[2],:].cpu().numpy(), cmap='bone', clim=[0,1.5])
        ax2.axis('off')

        if legend:
            cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.36])
            fig.colorbar(im0, cax=cbar_ax)

        plt.subplots_adjust(wspace=0.02)
        plt.show()

    else:
        rows = 5
        cols = 7
        fig, axs = plt.subplots(rows, cols, figsize=(20, 20))

        for i in range(35):
            row = i // cols
            col = i % cols
            axs[row, col].imshow(outputs[0,:,i,:].cpu().numpy(), cmap='bone', clim=[0,1])
            axs[row, col].set_title(str(i))
            axs[row, col].axis('off')
        plt.subplots_adjust(wspace=0.02, hspace=-0.55)
        plt.show()



def plot_var(model, proj, dev, noise_level, n=10, images=None):
    if images is None:
        images = get_SLP(dev)

    outputs=torch.zeros(n,*images.shape)
    blurred_slices = blur_image(images, kernel_size=5, sigma=2.0)
    with torch.no_grad():
        for i in range(n):
            sino, images = to_device(generate_data(blurred_slices,proj,noise_level=noise_level),dev), to_device(images,dev)
            outputs[i] = model(sino)
    
    var = torch.var(outputs,dim=0)
    
    plt.contourf(torch.flip(var[0,:,17,:].cpu(),[0]).numpy(), cmap='magma',levels=50)
    plt.axis('off')
    plt.show()

def visualize_unrolling(model, proj, dev, noise_level=0.3, images=None, idx=16):
    if images is None:
        images = get_SLP(dev)
    blurred_slices = blur_image(images, kernel_size=5, sigma=2.0)

    loganSin = generate_data(blurred_slices,proj,noise_level=noise_level)

    with torch.no_grad():
        sino, images = to_device(loganSin,dev), to_device(images,dev)
        outputs = model(sino)

    fig, axs = plt.subplots(1, len(outputs), figsize=(10, 5))
    for i in range(len(outputs)):
        axs[i].imshow(outputs[i][0,0,:,idx,:].cpu().numpy(), cmap='bone')
        axs[i].axis('off')

    plt.subplots_adjust(wspace=0.02)
    plt.show()