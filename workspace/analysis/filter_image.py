#!/usr/bin/env python3
from collections.abc import Callable
from functools import partial
from itertools import groupby, product, starmap
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import List

import cv2
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import zarr
from jump_portrait.fetch import get_jump_image
from jump_portrait.utils import batch_processing, parallel
from skimage.feature import (  # blob_dog is a faster approximation of blob_log
    blob_dog)
from skimage.filters import threshold_otsu
from skimage.util import img_as_float32, img_as_ubyte
from skimage.util.shape import view_as_windows

fig_directory = Path("/home/hhakem/projects/counterfactuals_projects/workspace/analysis/figures")


"""
# plot the mapping between moa_id to  #####################################################################
# perturbation into a figure to keep this mapping in mind #####################################################################
"""

def plot_table_id_to_pert(metadata: pl.DataFrame,
                          fig_name: str="moa_id_to_pert",
                          fig_directory: str=Path("./figures")):
    moa_to_pert = (metadata.select(pl.col(["moa", "moa_id", "pert_iname", "Metadata_InChIKey", "inchi_id"]))
                 .unique()
                 .sort(by=["moa_id", "pert_iname"])
                 .group_by("pert_iname", maintain_order=True)
                 .agg(pl.col(["Metadata_InChIKey", "inchi_id", "moa", "moa_id"
                              ]).unique(maintain_order=True))
                 .with_columns(pl.col(["Metadata_InChIKey", "inchi_id", "moa", "moa_id"
                              ]).list.explode())
                 ).to_pandas()

    # Assign a unique color to each moa_id
    unique_moa_ids = moa_to_pert["moa_id"].unique()
    base_palette = sns.color_palette("hsv", len(unique_moa_ids))  # Use a distinctive palette

    # Adjust the alpha (transparency) for each color
    alpha = 0.5  # Set desired transparency
    moa_id_to_color = {moa_id: mcolors.to_rgba(color, alpha=alpha) for moa_id, color in zip(unique_moa_ids, base_palette)}

    # Plot as a colored table
    fig, ax = plt.subplots(figsize=(15, 10))  # Bigger figure size
    ax.axis('tight')
    ax.axis('off')

    # Create the table
    table = ax.table(
        cellText=moa_to_pert.values,
        colLabels=moa_to_pert.columns,
        cellLoc='center',
        loc='center',
    )

    # Apply row colors based on moa_id
    for i, key in enumerate(table.get_celld().keys()):
        cell = table.get_celld()[key]
        if key[0] > 0:  # Skip the header row
            moa_id = moa_to_pert.iloc[key[0] - 1]["moa_id"]  # Find moa_id for the row
            cell.set_facecolor(moa_id_to_color[moa_id])  # Set color with adjusted alpha
        else:
            cell.set_text_props(weight='bold')  # Bold header row
        cell.set_height(0.15)  # Increase cell height
        cell.set_fontsize("large")  # Set font size

    # Increase column width to make bigger boxes
    table.auto_set_column_width(col=list(range(len(moa_to_pert.columns))))

    fig.suptitle("Map table from moa_id to perturbation",
                 fontsize="x-large",
                 weight="bold",
                 y=0.9)
    plt.tight_layout()
    fig.savefig(fig_directory / fig_name)
    plt.close()


metadata = pl.read_csv("target2_eq_moa2_active_metadata")
table = plot_table_id_to_pert(metadata, fig_name="moa_id_to_pert", fig_directory=fig_directory)

"""
# Fetching images while waiting for jump_portrait to be merged #####################################################################
"""

def try_function(f: Callable):
    '''
    Wrap a function into an instance which will Try to call the function:
        If it success, return the output of the function.
        If it fails, return None

    Parameters
    ----------
    f : Callable

    Returns
    -------
    tryed_fn : Callable
    '''
    # This assumes parameters are packed in a tuple
    def tryed_fn(*item, **kwargs):
        try:
            result = f(*item, **kwargs)
        except:
            result = None

        return result
    return tryed_fn


# ### ii) function to overcome turn get_jump_image_iter compatible with list and load data in a threaded fashion

def get_jump_image_batch(
        metadata: pl.DataFrame,
        channel: list[str],
        site: list[str],
        correction: str='Orig',
        # verbose: bool=True,
) -> tuple[list[tuple], list[np.ndarray]]:
    '''
    Load jump image associated to metadata in a threaded fashion.

    Parameters:
    ----------
    metadata : pl.DataFrame
        must have the column in this specific order ("Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well")
    channel : list of string
        list of channel desired
        Must be in ['DNA', 'ER', 'AGP', 'Mito', 'RNA']
    site : list of string
        list of site desired
        - For compound, must be in ['1' - '6']
        - For ORF, CRISPR, must be in ['1' - '9']
    correction : str
        Must be 'Illum' or 'Orig'
    verbose : bool
        Whether to enable tqdm or not.

    Return:
    ----------
    iterable : list of tuple
        list containing the metadata, channel, site and correction
    img_list : list of array
        list containing the images

    '''
    iterable = list(starmap(lambda *x: (*x[0], *x[1:]), product(metadata.rows(), channel, site, [correction])))
    img_list = parallel(iterable, batch_processing(try_function(get_jump_image)))
                        # verbose=verbose)

    return iterable, img_list
"""
# utility for image - cropping - clipping  ####################################
"""
def window_step(img, window_length, window_num):
    length = min(*img.shape[-2:])
    return (length - window_length) // (window_num -1)

def window_img(img_list, wanted_crop=128, num_crop=8):
    with ThreadPool(cpu_count()) as thread:
        img_window = np.vstack(list(
            thread.map(lambda img: view_as_windows(img,
                                            (img.shape[-3], wanted_crop, wanted_crop),
                                            window_step(img, wanted_crop, num_crop)),
                        img_list)))

    pixel_window =  sum(list(map(np.size,  img_window)))
    pixel_origin = sum(list(map(np.size,  img_list)))
    print(f"pixel increase due to overlapping window: {(pixel_window - pixel_origin) / pixel_origin:.2%}")
    return img_window

def clip_norm_img(img, clip=(1, 99), axis=(1, 2, 4, 5)):
    min_clip_img, max_clip_img = np.percentile(img, clip,
                                               axis=axis,
                                               keepdims=True)
    img_norm = np.clip(img,
                        min_clip_img,
                        max_clip_img)
    img_norm = (img_norm - min_clip_img) / (max_clip_img - min_clip_img)
    return img_norm

"""
# Plotting function to plot img in rgb - plot blob and otsu #####################################################################
"""

def channel_to_rgb(img_array: np.ndarray,
                   channel: List[str]):
    if img_array.ndim not in {3, 4}:
        raise Exception(
            f"input array should have shape (sample, C, H, W) or (C, H, W).\n"
            f"Here input shape: {img_array.shape}")
    if img_array.shape[-3] > len(channel):
        raise Exception(f"input array should have shape (sample, C, H, W) or (C, H, W) with C <= 5.\n"
                        f"Here input shape: {img_array.shape}\n"
                        f"And C: {img_array.shape[-3]}")

    map_channel_color = {
        "AGP": "#FF7F00", #"#FFA500", # Orange
        "DNA": "#0000FF", # Blue
        "ER": "#00FF00", #"#65fe08" # Green
        "Mito": "#FF0000", # Red
        "RNA": "#FFFF00", # Yellow
                         }
    channel_rgb_weight = np.vstack(list(map(lambda ch: list(mcolors.to_rgb(map_channel_color[ch])), channel)))
    img_array_rgb = np.tensordot(img_array, channel_rgb_weight, axes=[[-3], [0]])
    # tensordot give img in (sample, H, W, C) or (H, W, C)
    # we normalize and rotate for the sake of consistency with input
    norm_rgb = np.maximum(channel_rgb_weight.sum(axis=0), np.ones(3))
    img_array_rgb = np.moveaxis(img_array_rgb / norm_rgb, -1, -3)
    return img_array_rgb.clip(0, 1) # to ensure correct normalisation

def plot_img(img_array, label_array, channel, size=4, fig_name="multiple_cells_small_crop",
             fig_directory=Path("./figures"), seed=42):
    rng = np.random.default_rng(seed)
    choice = rng.choice(img_array.shape[0], size=size*size, replace=False)
    fig, axis = plt.subplots(size, size, figsize=(5 * size, 5 * size))
    axis = axis.flatten()
    for ax, i in zip(axis, choice):
        img_to_plot = channel_to_rgb(img_array[i], channel)
        ax.imshow(img_to_plot.transpose(1, 2 ,0))
        ax.set_axis_off()
        ax.set_title(f"class: {label_array[i]}")
    fig.savefig(fig_directory / fig_name)
    plt.close()

def plot_img_blob(img_array, blob_list, label_array, channel,
                  blob_thr=None, size=4,
                  fig_name="multiple_cells_small_crop",
                  fig_directory=Path("./figures"), seed=42):
    rng = np.random.default_rng(seed)
    choice = rng.choice(img_array.shape[0], size=size*size, replace=False)
    fig, axis = plt.subplots(size, size, figsize=(5 * size, 5 * size))
    axis = axis.flatten()
    for ax, i in zip(axis, choice):
        img_to_plot = channel_to_rgb(img_array[i], channel)
        ax.imshow(img_to_plot.transpose(1, 2 ,0))
        blob_area = cumulative_circle_area_numpy(blob_list[i])
        if blob_thr is not None:
            if blob_area < blob_thr:
                ax.add_patch(mpatches.Rectangle((0,0), img_to_plot.shape[-2], img_to_plot.shape[-2], color="red", alpha=0.3))
        for blob in blob_list[i]:
            y, x, r = blob
            c = plt.Circle((x, y), r, color="red", linewidth=2, fill=False)
            ax.add_patch(c)
        ax.set_axis_off()
        ax.set_title(f"class: {label_array[i]} - blob_area: {blob_area:.1%}")
    fig.savefig(fig_directory / fig_name)
    plt.close()

def plot_img_otsu(img_array, img_binary, label_array, channel, size=4,
                  otsu_thr=0.05, alpha=0.2, fig_name="multiple_cells_small_crop",
                  fig_directory=Path("./figures"),
                  seed=42):
    rng = np.random.default_rng(seed)
    choice = rng.choice(img_array.shape[0], size=size*size, replace=False)
    fig, axis = plt.subplots(size, size, figsize=(5 * size, 5 * size))
    axis = axis.flatten()
    for ax, i in zip(axis, choice):
        img_to_plot = channel_to_rgb(img_array[i], channel)
        ax.imshow(img_to_plot.transpose(1, 2 ,0))
        ax.imshow(img_binary[i], cmap="gray", alpha=alpha)
        area = (img_binary[i].sum(axis=(0,1)) / img_binary[i].size)
        if otsu_thr is not None:
            if area < otsu_thr:
                ax.add_patch(mpatches.Rectangle((0,0), img_to_plot.shape[-2], img_to_plot.shape[-2], color="red", alpha=0.3))
            ax.set_axis_off()
        ax.set_title(f"class: {label_array[i]} - otsu_area: {area:.2%} ")
    fig.savefig(fig_directory / fig_name)
    plt.close()

def plot_feat_distrib_per_class(img_feat, label_array, channel,
                                bins=30,
                                thr=None,
                                fig_name="feat_distribution_per_class_per_channel",
                                fig_directory=Path("./figures")):
    img_feat_per_class = list(
        map(lambda y: img_feat[label_array == y].T,
            np.unique(label_array)))

    map_channel_color = {
        "AGP": "#FF7F00", #"#FFA500", # Orange
        "DNA": "#0000FF", # Blue
        "ER": "#00FF00", #"#65fe08" # Green
        "Mito": "#FF0000", # Red
        "RNA": "#FFFF00", # Yellow
                         }
    # Plotting parameters
    n_arrays = len(img_feat_per_class)
    n_channels = len(channel)
    fig, axes = plt.subplots(n_arrays, n_channels, figsize=(15, 10),
                             sharex='col',  # Shared x-axis per column
                             sharey='row',  # Shared y-axis per row
                             constrained_layout=True,
                             squeeze=False)

    # Iterate over arrays and channels
    for array_idx, array in enumerate(img_feat_per_class):
        for channel_idx in range(n_channels):
            ax = axes[array_idx, channel_idx]

            # Extract data for the specific channel
            data = array[channel_idx]

            # Plot histogram using Seaborn
            sns.histplot(data, bins=50, kde=True, ax=ax,
                         color=map_channel_color[channel[channel_idx]],
                         alpha=0.7)

            if thr is not None:
                ax.axvline(x=thr, color='red', linestyle='--', linewidth=1.5, label=f"Threshold = {thr}")

                # Add bold x-tick for `thr`
                xticks = ax.get_xticks()
                if thr not in xticks:
                    xticks = np.append(xticks, thr)
                ax.set_xticks(xticks)
                ax.set_xticklabels(
                    [f"{tick:.1f}" if tick != thr else f"$\\bf{{{tick}}}$" for tick in xticks]
                )
            # Titles for first row and first column
            if array_idx == 0:
                ax.set_title(f"Channel {channel[channel_idx]}", fontsize=10)
            if channel_idx == 0:
                ax.set_ylabel(f"Class {array_idx}", fontsize=10)

    # Overall title for the figure
    fig.suptitle(fig_name, fontsize=16)
    fig.savefig(fig_directory / fig_name)

def plot_joint_distribution(x, y, categories,
                            x_name="X", y_name="Y",
                            fig_name="joint_distribution",
                            fig_directory=Path("./figures"),
                            kind='scatter', height=10):

    # Create the jointplot
    g = sns.jointplot(
        x=x,
        y=y,
        hue=categories,
        kind=kind,
        alpha=0.6,
        palette=sns.color_palette()[:len(np.unique(categories))],
        height=height
    )

    # Customize the axis labels
    g.ax_joint.set_xlabel(x_name, fontsize=12)
    g.ax_joint.set_ylabel(y_name, fontsize=12)

    # Customize the legend title
    g.ax_joint.legend_.set_title("Class")

    # Save the figure
    plt.title(fig_name)
    plt.tight_layout()
    plt.savefig(fig_directory / fig_name)
    plt.close()

"""
# Filter utility  #####################################################################
"""

def blob_compute(img, func=blob_dog, **kwargs):
    blob = func(img, **kwargs)
    blob[:, 2] = blob[: , 2] * np.sqrt(2)
    return blob

def cumulative_circle_area_numpy(circle_data, return_mask=False, square_size=128):
    """
    Compute the cumulative area of circles intersecting with a square using NumPy.

    Parameters:
    - circle_data: 2D array where each row is [x, y, radius].
    - square_size: Size of the square (default is 128x128).

    Returns:
    - Cumulative area of all circles intersecting the square.
    """
    # Create a grid of points representing the square
    x_grid, y_grid = np.ogrid[:square_size, :square_size]

    # Expand circle parameters for vectorized operations
    x_centers = circle_data[:, 0][:, np.newaxis, np.newaxis]
    y_centers = circle_data[:, 1][:, np.newaxis, np.newaxis]
    radii = circle_data[:, 2][:, np.newaxis, np.newaxis]

    # Compute squared distances from the grid to each circle center
    distance_squared = (x_grid - x_centers)**2 + (y_grid - y_centers)**2

    # Create masks for points within each circle
    circle_masks = distance_squared <= radii**2

    # Sum up areas for all circles
    # Each `True` in the mask corresponds to 1 unit of area
    mask = np.sum(circle_masks, axis=0).clip(0, 1)
    if return_mask:
        return mask
    else:
        total_area = np.sum(mask) / (square_size ** 2)
        return total_area

def get_filter_thr(feat, num_bin=1):
    """
    feature is a descriptor of our images.
    compute bin width of the histogram (distribution) of feature using the
    ‘fd’ (Freedman Diaconis Estimator). The threshold above which we keep every imgages
    is given by this estimator * num_bin (higher the num_bin, less images are retrieved).
    """
    iqr = np.subtract(*np.percentile(feat, [75, 25]))
    fd = 2.0 * iqr * feat.size ** (-1.0 / 3.0)
    return num_bin * fd

def get_mask_thr(feat, labels, num_bin=1):
    """
    get threshold for feature grouped by label. Then retrieve mask of feature above the threshold for each group
    """
    mask_labels = np.vstack(list(map(lambda i : labels == i, np.unique(labels))))
    thr_array = np.hstack(list(map(lambda mask: get_filter_thr(feat[mask], num_bin), mask_labels)))
    mask_above_thr = (feat >= thr_array).T
    return (mask_above_thr * mask_labels).sum(axis=0).astype(bool)

"""
# Execute above code  #####################################################################
"""

#### Fetch image and stack each channel
metadata = pl.read_csv("target2_eq_moa2_active_metadata")
plot_table_id_to_pert(metadata, fig_name="moa_id_to_pert", fig_directory=fig_directory)


channel = ['AGP', 'DNA', 'ER', 'Mito', 'RNA']
channel = sorted(channel) # just to make sure there is consistency across data. This should be the default.
iterable, img_list = get_jump_image_batch(metadata.select(pl.col(["Metadata_Source", "Metadata_Batch",
                                                                  "Metadata_Plate", "Metadata_Well"])),
                                          channel=channel,#, 'ER', 'AGP', 'Mito', 'RNA'],
                                          site=[str(i) for i in range(1, 7)],
                                          correction=None) #None, 'Illum'
mask = [x is not None for x in img_list]
iterable_mask, img_list_mask = [it for i, it in enumerate(iterable) if mask[i]],  [img for i, img in enumerate(img_list) if mask[i]]

zip_iter_img = sorted(zip(iterable_mask, img_list_mask),
                      key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3], x[0][5], x[0][4]))
# grouped image are returned as the common key, and then the zip of param and img, so we retrieve the img then we stack
iterable_stack, img_list_stack = map(lambda tup: list(tup),
                                zip(*starmap(lambda key, param_img: (key, np.stack(list(map(lambda x: x[1], param_img)))),
                                             groupby(zip_iter_img,
                                                     key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3], x[0][5])))))

#### Retrieve moa_id and InChIKey
labels, groups = tuple((metadata
        .select(pl.col(["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well", "moa_id", "inchi_id"]))
        .join(pl.DataFrame([t[:4] for t in iterable_stack],
                           schema=["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well"]),
              on=["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well"],
              how="left")
        .select(pl.col(["moa_id", "inchi_id"]))
        .to_dict(as_series=False)
        .values()))

#### Crop img in small square, clip normalize them and flatten window - plot example of images
wanted_crop = 128 #256
num_crop = 8 #4
img_stack_window = window_img(img_list_stack, wanted_crop=wanted_crop, num_crop=num_crop) # axis = (sample, row_grid, column_grid, channel, H, W)
# NB: (1, 2, 4, 5) : normalise per image of origin and (4, 5) : normalise per tile
img_stack_window_norm = clip_norm_img(img_stack_window, clip=(1, 99), axis=(1, 2, 4, 5)) # clip on big image to reduce noise


img_flat_crop = img_stack_window_norm.reshape(-1, *img_stack_window_norm.shape[-3:])
labels_flat_crop = np.repeat(labels, num_crop**2)
groups_flat_crop = np.repeat(groups, num_crop**2)
iterable_flat_crop = list(
    map(lambda tup: (*tup[0], tup[1]), zip(
    [t for tup in iterable_stack for t in (num_crop**2 * [tup])], np.tile(np.arange(num_crop**2), len(labels)))))

# plot rgb image
plot_img(img_flat_crop, labels_flat_crop, channel=channel, size=12,
         fig_name="multiple_cells_crop_norm_big_img", fig_directory=fig_directory)

#### Filter out low quality image
##### Blob detection on dna channel (dna channel is relevant to nucleus so is a proxi for precise cell detection)
img_gray_dna = img_as_ubyte(img_flat_crop[:, 1])

with ThreadPool(cpu_count()) as thread:
    blobs_dna = list(thread.map(partial(blob_compute, func=blob_dog, min_sigma=5, max_sigma=25, threshold=0.1), img_gray_dna))

with ThreadPool(cpu_count()) as thread:
    blobs_dna_area = np.stack(list(thread.map(partial(cumulative_circle_area_numpy, square_size=wanted_crop), blobs_dna)))[:, None]

plot_feat_distrib_per_class(blobs_dna_area, labels_flat_crop, channel[1:2],
                            bins="auto",
                            thr=0.03,
                            fig_name="blob_area_dna_distribution",
                            fig_directory=fig_directory)

plot_img_blob(img_flat_crop[:, 1:2], blobs_dna, labels_flat_crop, channel=channel[1:2],
              blob_thr=None, size=12,
              fig_name="multiple_cells_crop_norm_big_img_blob", fig_directory=fig_directory)

##### Otsu filter with morphological opening to remove noise again on dna channel
size_closing = 10
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size_closing, size_closing))

with ThreadPool(cpu_count()) as thread:
    otsu_filt_dna = list(thread.map(threshold_otsu, img_gray_dna))
    otsu_dna = (img_gray_dna >= np.stack(otsu_filt_dna)[:, None, None]).astype(np.float32)
    otsu_dna = np.stack(list(thread.map(partial(cv2.morphologyEx, op=cv2.MORPH_OPEN, kernel=kernel), otsu_dna)))

otsu_dna_area = (otsu_dna.sum(axis=(1,2)) / otsu_dna[0].size)[:, None]
plot_feat_distrib_per_class(otsu_dna_area,
                            labels_flat_crop,
                            channel[1:2],
                            thr=0.02,
                            bins="auto",
                            fig_name="otsu_area_distribution",
                            fig_directory=fig_directory)

plot_img_otsu(img_flat_crop[:, 1:2], otsu_dna, labels_flat_crop, channel=channel[1:2], size=12,
              otsu_thr=None,
              alpha=0.2,
              fig_name="multiple_cells_crop_norm_big_img_otsu", fig_directory=fig_directory)

##### Combining otsu filter and blob filter
with ThreadPool(cpu_count()) as thread:
    blobs_dna_mask = np.stack(list(thread.map(partial(cumulative_circle_area_numpy, return_mask=True, square_size=wanted_crop), blobs_dna)))

blob_otsu_dna = (blobs_dna_mask * otsu_dna).clip(0, 1)
blob_otsu_dna_area = (blob_otsu_dna.sum(axis=(1,2)) / blob_otsu_dna[0].size)[:, None]

plot_feat_distrib_per_class(blob_otsu_dna_area,
                            labels_flat_crop,
                            channel[1:2],
                            thr=0.02,
                            bins="auto",
                            fig_name="blob_otsu_area_distribution",
                            fig_directory=fig_directory)

plot_img_otsu(img_flat_crop[:, 1:2], blob_otsu_dna, labels_flat_crop, channel=channel[1:2], size=12,
              otsu_thr=None,
              alpha=0.2,
              fig_name="multiple_cells_crop_norm_big_img_blob_otsu", fig_directory=fig_directory)

##### Show joint distribution of each features obtained
plot_joint_distribution(blobs_dna_area[:,0], otsu_dna_area[:,0], labels_flat_crop,
                        x_name="blob area", y_name="otsu area",
                        fig_name="joint_distribution_blobs_vs_otsu_area",
                        fig_directory=Path("./figures"),
                        kind='scatter', height=10)

plot_joint_distribution(blobs_dna_area[:,0], blob_otsu_dna_area[:,0], labels_flat_crop,
                        x_name="blob area", y_name="blob_otsu area",
                        fig_name="joint_distribution_blobs_vs_blob_otsu_area",
                        fig_directory=Path("./figures"),
                        kind='scatter', height=10)

plot_joint_distribution(blob_otsu_dna_area[:,0], otsu_dna_area[:,0], labels_flat_crop,
                        x_name="blob_otsu area", y_name="otsu area",
                        fig_name="joint_distribution_blob_otsu_vs_otsu_area",
                        fig_directory=Path("./figures"),
                        kind='scatter', height=10)

#### Filter out low quality images based on the blob_otsu_dna_area feature.
filter_mask = get_mask_thr(blob_otsu_dna_area, labels_flat_crop, num_bin=2)
img_filt = img_flat_crop[filter_mask]
labels_filt = labels_flat_crop[filter_mask]
groups_filt = groups_flat_crop[filter_mask]
metadata_filt = [it for i, it in enumerate(iterable_flat_crop) if filter_mask[i]]


plot_img(img_filt, labels_filt, channel=channel, size=12,
         fig_name="multiple_cells_crop_norm_big_img_filt",
         fig_directory=fig_directory,
         seed=12)


#### Enhance contrast using CLAHE Histogram equalization
clip_limit = 2
tile_grid_size = (8, 8)
clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

with ThreadPool(cpu_count()) as thread:
    img_filt_contrast = np.stack(list(
        thread.map(lambda img: np.stack(list(map(lambda img_ch: img_as_float32(clahe.apply(img_as_ubyte(img_ch))), img))),
                   img_filt)))

plot_img(img_filt_contrast, labels_filt, channel=channel, size=12,
         fig_name="multiple_cells_crop_norm_big_img_filt_contrasted",
         fig_directory=fig_directory,
         seed=12)

img_filt = img_filt.astype(np.float32)
#### Store dataset
store = zarr.DirectoryStore(Path("image_active_crop_dataset/imgs_labels_groups.zarr"))
root = zarr.group(store=store)
# Save datasets into the group
root.create_dataset('imgs', data=img_filt, overwrite=True, chunks=(1, 1, *img_filt.shape[2:]))
root.create_dataset('imgs_contrast', data=img_filt_contrast, overwrite=True, chunks=(1, 1, *img_filt_contrast.shape[2:]))
root.create_dataset('labels', data=labels_filt, overwrite=True, chunks=1)
root.create_dataset('groups', data=groups_filt, # dtype=object, object_codec=numcodecs.JSON(), ### This encoding was for string inchi
                    overwrite=True, chunks=1)
#### Store metadata
metadata_filt_df = pl.DataFrame(metadata_filt, schema=["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well", "site", "crop_id"])
metadata_filt_df = metadata_filt_df.with_row_count("img_index")
metadata_filt_df.write_csv(Path("image_active_crop_dataset/metadata.csv"))
