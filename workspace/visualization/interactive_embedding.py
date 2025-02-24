#!/usr/bin/env python3

# 1. Visualise image
# 2. Convert to functional argument
# 3. Reproduce marimo example

import marimo
import numpy as np

__generated_with = "0.9.23"
app = marimo.App(width="full")

import polars as pl
from pathlib import Path
@app.cell
def __():
    from pathlib import Path

    import numpy as np
    import polars as pl
    from umap import UMAP
    from sklearn.decomposition import PCA

    ap_tidy_dir = Path("../tidy_ap")

    metadata = pl.read_csv("target2_eq_moa2_active_metadata")
    # contains [img_id, Metadata_Source, Metadata_Batch, Metadata_Plate, Metadata_Well, site, crop_id]
    metadata_img = pl.read_csv(Path("image_active_crop_dataset/metadata"))
    # maybe smarter way to retrieve embedding with regex
    embedding_img = pl.read_csv(Path("image_active_crop_dataset/embedding_VGG_image_crop_active_groups_fold_1epoch=77-train_acc=0.86-val_acc=0.77.csv"))
    reducer_pca = PCA(100)
    reducer_umap = UMAP()
    embedding_XY = reducer_umap.fit_transform(
        reducer_pca.fit_transform(
            embedding_img.select(pl.all().exclude("pred")).to_numpy()))
    # need to put embedding_XY into a dataframe
    return Path, np, pl, ap_tidy_dir, metadata, metadata_img, embedding_img


@app.cell
def __(Path):
    from functools import cache
    import zarr

    @cache
    def load_image(img_id: np.ndarray, channel_id: np.ndarray) -> np.ndarray:
        img_path = Path("image_active_crop_dataset/imgs_labels_groups.zarr")
        img_dataset = zarr.open(img_path)
        return img_dataset["imgs"].oindex[img_id, channel_id]
    return cache, zarr, load_image


@app.cell
def __(alt):
    def scatter(df):
        return (
            alt.Chart(df)
            .mark_circle()
            .encode(
                x=alt.X("average_precision:Q"),#.scale(domain=(-2.5, 2.5)),
                y=alt.Y("cell_count:Q"),#.scale(domain=(-2.5, 2.5)),
                color=alt.Color("MoA:N"),
                # shape=alt.Shape("Compound:N")
            )
            .properties(width=500, height=500)
        )
    return (scatter,)


@app.cell
def __(df, mo, scatter):
    chart = mo.ui.altair_chart(scatter(df.to_pandas()))
    chart
    return (chart,)


@app.cell
def __(mo):
    import string

    channel = mo.ui.slider(1, 6, value=3, step=1, label="Channel")
    clip_outliers = mo.ui.checkbox(value=True, label="Clip outliers")
    return channel, string, tp, z, clip_outliers


@app.cell
def __(chart, mo):
    table = mo.ui.table(chart.value)
    return (table,)


@app.cell
def __(channel, chart, load_image, mo, table, tp, z, clip_outliers):
    import math

    import matplotlib.pyplot as plt
    mo.stop(not len(chart.value))

    import numpy as np

    def clip_outliers_(image: np.ndarray, lower_percentile=1, upper_percentile=99):
        """
        Clips pixel values in a numpy image array to the specified percentile range.

        Parameters:
            image (np.ndarray): Input image array.
            lower_percentile (float): Lower bound percentile for clipping (default 2.5 for 95% range).
            upper_percentile (float): Upper bound percentile for clipping (default 97.5 for 95% range).

        Returns:
            np.ndarray: Image with values clipped to the specified percentile range.
        """
        # Calculate the percentile thresholds
        lower_bound = np.percentile(image, lower_percentile)
        upper_bound = np.percentile(image, upper_percentile)

        # Clip the image values to the computed thresholds
        clipped_image = np.clip(image, lower_bound, upper_bound)

        return clipped_image

    def show_images(indices, max_images=14):

        indices = indices[:max_images]

        # TODO Preload images or make threading work
        images = [load_image(*x.split("_"), tp.value, z.value, channel.value) for x in indices]

        fig, axes = plt.subplots(min(2, len(indices)), math.ceil(len(indices)/2))
        fig.set_size_inches(20, 5)
        if len(indices) > 1:
            for im, ax in zip(images, axes.T.flat):
                if clip_outliers.value:
                    im = clip_outliers_(im)

                ax.imshow(im, cmap="gray")
                ax.set_yticks([])
                ax.set_xticks([])
        else:
            axes.imshow(images[0], cmap="gray")
            axes.set_yticks([])
            axes.set_xticks([])

        plt.tight_layout()
        return fig

    selected_images = (
        show_images(list(chart.value["site"]))
        if not len(table.value)
        else show_images(list(table.value["site"]))
    )

    mo.md(
        f"""
        **Here's a preview of the images you've selected**:

        {channel}
        {clip_outliers}

        {mo.as_html(selected_images)}

        Here's all the data you've selected.

        {table}
        """
    )
    return clip_outliers, math, np, plt, selected_images, show_images


@app.cell
async def __():
    import sys

    if "pyodide" in sys.modules:
        import micropip

        await micropip.install("altair")

    import altair as alt
    return alt, micropip, sys


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
