# 1. Visualise image
# 2. Convert to functional argument
# 3. Reproduce marimo example

import marimo

__generated_with = "0.9.27"
app = marimo.App(width="full")


@app.cell
def __():
    from pathlib import Path

    import numpy
    import polars as pl


    # df = pl.read_parquet("/datastore/shared/attribution/data/main_data.parquet")
    df = pl.read_parquet("/home/hhakem/projects/counterfactuals_projects/workspace/analysis/image_active_crop_dataset/embedding_UMAP.parquet")
    return Path, df, numpy, pl


@app.cell
def __(Path):
    from functools import cache

    import zarr

    @cache
    def load_image(image_id: int, channel: int):
        zarr_path = Path("/home/hhakem/projects/counterfactuals_projects/workspace/analysis/image_active_crop_dataset/imgs_labels_groups.zarr")
        with zarr.open(zarr_path) as imgs:
            return imgs["imgs"].oindex[image_id, channel]
    return cache, load_image, zarr


@app.cell
def __(alt):
    def scatter(df):
        return (
            alt.Chart(df)
            .mark_circle()
            .encode(
                x=alt.X("UMAP1:Q"),#"PC1:Q"),#.scale(domain=(-2.5, 2.5)),
                y=alt.Y("UMAP2:Q"),#"PC2:Q"),#.scale(domain=(-2.5, 2.5)),
                color=alt.Color("Metadata_Batch:N"),
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

    channel = mo.ui.slider(0, 5, value=3, step=1, label="Channel")
    clip_outliers = mo.ui.checkbox(value=False, label="Clip outliers")
    return channel, clip_outliers, string


@app.cell
def __(chart, mo):
    table = mo.ui.table(chart.value)
    return (table,)


@app.cell
def __(channel, chart, clip_outliers, load_image, mo, table):
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

    def show_images(indices, max_images=8):

        indices = indices[:max_images]

        images = [load_image(x, channel.value) for x in indices]

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
        show_images(list(chart.value["img_id"]))
        if not len(table.value)
        else show_images(list(table.value["img_id"]))
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
    return clip_outliers_, math, np, plt, selected_images, show_images


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


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
