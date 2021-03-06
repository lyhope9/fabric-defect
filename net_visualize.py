from __future__ import print_function

#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import math

def get_grid_dim(num_filters):
    s = math.sqrt(num_filters)
    r = int(s + 0.5)
    c = int(num_filters /s  + 0.5)
    return r,c


def plot_conv_weights(weights, plot_dir, name, channels_all=True, filters_all=True, channels=[0], filters=[0]):
    """
    Plots convolutional filters
    :param weights: numpy array of rank 4
    :param name: string, name of convolutional layer
    :param channels_all: boolean, optional
    :return: nothing, plots are saved on the disk
    """

    w_min = np.min(weights)
    w_max = np.max(weights)

    # make a list of channels if all are plotted
    if channels_all:
        num_channels = weights.shape[2]
        channels = range(weights.shape[2])
    else:
        num_channels = len(channels)

    # get number of convolutional filters
    if filters_all:
        num_filters = weights.shape[3]
        filters = range(weights.shape[3])
    else:
        num_filters = len(filters)

    # get number of grid rows and columns
    grid_r, grid_c = get_grid_dim(num_channels * num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))
    plt.suptitle(name)
    # iterate channels
    for channel_ID in channels:
        # iterate filters inside every channel
        if num_filters == 1:
            img = weights[:, :, channel_ID, filters[0]]
            axes.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')
            # remove any labels from the axes
            axes.set_xticks([])
            axes.set_yticks([])
        else:
            for l, ax in enumerate(axes.flat):
                if(l >= num_filters * num_channels):
                    break
                # get a single filter
                img = weights[:, :, channel_ID, filters[l%num_filters]]
                # put it on the grid
                ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')
                # remove any labels from the axes
                ax.set_xticks([])
                ax.set_yticks([])
        # save figure
        plt.savefig(os.path.join(plot_dir, '{}-{}.png'.format(name, channel_ID)), bbox_inches='tight')

def plot_conv_output(conv_img, plot_dir, name, filters_all=True, filters=[0]):
    w_min = np.min(conv_img)
    w_max = np.max(conv_img)

    # get number of convolutional filters
    if filters_all:
        num_filters = conv_img.shape[3]
        filters = range(conv_img.shape[3])
    else:
        num_filters = len(filters)

    # get number of grid rows and columns
    grid_r, grid_c = get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))
    plt.suptitle(name)
    # iterate filters
    if num_filters == 1:
        img = conv_img[0, :, :, filters[0]]
        axes.imshow(img, vmin=w_min, vmax=w_max, interpolation='bicubic', cmap=cm.hot)
        # remove any labels from the axes
        axes.set_xticks([])
        axes.set_yticks([])
    else:
        for l, ax in enumerate(axes.flat):
            if (l >= num_filters):
                break
            # get a single image
            img = conv_img[0, :, :, filters[l]]
            # put it on the grid
            ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='bicubic', cmap=cm.hot)
            # remove any labels from the axes
            ax.set_xticks([])
            ax.set_yticks([])
    # save figure
    plt.savefig(os.path.join(plot_dir, '{}.png'.format(name)), bbox_inches='tight')