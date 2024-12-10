# 1. Visualise image
# 2. Convert to functional argument
# 3. Reproduce marimo example

import marimo

__generated_with = "0.9.27"
app = marimo.App(width="full")


@app.cell
def __():
    from pathlib import Path
    import polars as pl

    # df = pl.read_parquet("/datastore/shared/attribution/data/main_data.parquet")
    df = pl.read_parquet("/home/hhakem/projects/counterfactuals_projects/workspace/analysis/image_active_crop_dataset/embedding_UMAP.parquet")
    df = df.with_columns(pl.when(pl.col("Metadata_Source").is_in(("source_5","source_10"))).then(True).otherwise(False).alias("flag"))
    return Path, df, pl


@app.cell
def __(Path):
    from functools import cache

    import zarr
    import numpy as np

    @cache
    def load_image(image_id: int, channel: int):
        zarr_path = Path("/home/hhakem/projects/counterfactuals_projects/workspace/analysis/image_active_crop_dataset/imgs_labels_groups.zarr")
        imgs = zarr.open(zarr_path)
        return imgs["imgs"].oindex[image_id, channel]

    def load_image_channel(image_id: int, channel: tuple(int)):
        return np.stack([load_image(image_id, ch) for ch in channel], axis=-3)
    return cache, load_image, load_image_channel, np, zarr


@app.cell
def __(mo):
    # Get all possible metadata fields for dropdown
    metadata_fields = [
        "Metadata_Source",
        "Metadata_Batch",
        "moa",
        "moa_id",
        "pert_iname",
        "Metadata_InChIKey",
        "inchi_id",
        "pred",
    ]

    # Create a dropdown for selecting the metadata field
    color_by = mo.ui.dropdown(
        metadata_fields,
        value="Metadata_Batch",
        label="**Color by metadata**"
    )

    mo.md(f"""
    {color_by}
    """)
    return color_by, metadata_fields


@app.cell
def __(alt, color_by):
    def scatter(df):
        return (
            alt.Chart(df)
            .mark_circle(size=10, opacity=0.2)
            .encode(
                x=alt.X("UMAP1:Q"),#"PC1:Q"),#.scale(domain=(-2.5, 2.5)),
                y=alt.Y("UMAP2:Q"),#"PC2:Q"),#.scale(domain=(-2.5, 2.5)),
                color=alt.Color(f"{color_by.value}:N")
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
    # Max images
    max_images_ = mo.ui.number(start=4, stop=32, value=8, step=1)

    # Checkbox for selecting channels
    get_channel, set_channel = mo.state(["DNA"])
    channel_to_select = ["AGP", "DNA", "ER", "Mito", "RNA"]
    channel_to_indices = {
        "AGP": 0,
        "DNA": 1,
        "ER": 2,
        "Mito": 3,
        "RNA": 4
    }
    indices_to_channel = {v: k for k, v in channel_to_indices.items()}
    channel_multiselect = mo.ui.multiselect(options=channel_to_select, value=["DNA"], on_change=lambda v: set_channel(v) if len(v) > 0 else None)
    # Switch to clip outliers
    clip_outliers = mo.ui.switch(value=False, label="**Clip**")
    low_high_clip = mo.ui.range_slider(0, 100, value=(1, 99))
    return (
        channel_multiselect,
        channel_to_indices,
        channel_to_select,
        clip_outliers,
        get_channel,
        indices_to_channel,
        low_high_clip,
        max_images_,
        set_channel,
    )


@app.cell
def __(clip_outliers, mo):
    # Normalize button available if image clipped
    normalize = mo.ui.button(value=False, kind="neutral", label="**Normalize**", on_click=lambda v: not v, disabled=not clip_outliers.value)
    return (normalize,)


@app.cell
def __(np):
    # Define channel to rgb mapping
    import matplotlib.colors as mcolors

    def channel_to_rgb(img_array: np.ndarray, channel: list):
        # Ensure the image array has the correct shape
        if img_array.ndim not in {3, 4}:
            raise Exception(f"Input array should have shape (sample, C, H, W) or (C, H, W).\n"
                             f"Here input shape: {img_array.shape}")

        if img_array.shape[-3] > len(channel):
            raise Exception(f"Input array should have shape (sample, C, H, W) or (C, H, W) with C <= 5.\n"
                             f"Here input shape: {img_array.shape}\n"
                             f"And C: {img_array.shape[-3]}")

        # Channel to RGB color map
        map_channel_color = {
            "AGP": "#FF7F00",  # Orange
            "DNA": "#0000FF",  # Blue
            "ER": "#00FF00",   # Green
            "Mito": "#FF0000",  # Red
            "RNA": "#FFFF00",   # Yellow
        }

        # Create RGB weights for the selected channels
        channel_rgb_weight = np.vstack([list(mcolors.to_rgb(map_channel_color[ch])) for ch in channel])

        # Combine the selected channels using tensordot
        img_array_rgb = np.tensordot(img_array, channel_rgb_weight, axes=[[-3], [0]])

        # Normalize the result
        norm_rgb = np.maximum(channel_rgb_weight.sum(axis=0), np.ones(3))
        img_array_rgb = np.moveaxis(img_array_rgb / norm_rgb, -1, -3)

        return img_array_rgb.clip(0, 1)  # Ensure correct normalization
    return channel_to_rgb, mcolors


@app.cell
def __(chart, mo):
    table = mo.ui.table(chart.value)
    return (table,)


@app.cell
def __(chart, load_image_channel, np):
    # load images
    channel_id_selected = list(range(5))
    raw_images = np.stack([load_image_channel(x, tuple(channel_id_selected)) for x in list(chart.value["img_id"])], axis=0)
    pixels = raw_images.reshape(len(chart.value), 5, -1)
    return channel_id_selected, pixels, raw_images


@app.cell
def __(channel_id_selected, chart, indices_to_channel, np, pixels, pl):
    import operator
    # compute pixel_count according bins
    num_bins = 80
    bins = np.round(np.linspace(0, 1, num_bins + 1), 5)
    pixel_count = np.stack(list(map(lambda id: np.stack(list(map(lambda ch: np.histogram(pixels[id][ch], bins=bins)[0], channel_id_selected))), range(pixels.shape[0]))))
    pixel_count_df = pl.DataFrame({
        "img_id": np.repeat(list(chart.value["img_id"]), pixel_count[0].size),
        "channel": np.tile(np.repeat([indices_to_channel[i] for i in np.arange(5)], pixel_count.shape[-1]), pixel_count.shape[0]),
        "pixel_count": pixel_count.reshape(-1),
        "bin_start": np.tile(bins[:-1], operator.mul(*pixel_count.shape[:-1])),  # Start of bins
        "bin_end": np.tile(bins[1:], operator.mul(*pixel_count.shape[:-1]))
    })
    return bins, num_bins, operator, pixel_count, pixel_count_df


@app.cell
def __(df, pixel_count_df, pl):
    # group per metadata
    metadata = ["Metadata_Source"]
    pixel_count_df_grouped = (pixel_count_df
                              .join(
                                  other=df.select(pl.col(["img_id"] + metadata)),
                                  on="img_id",
                                  how="left"
                              )
                              .group_by(pl.col(metadata + ["channel", "bin_start", "bin_end"]))
                              .agg(pl.col("pixel_count").sum())
                              .with_columns(
                                  (pl.col("pixel_count") / pl.col("pixel_count").sum().over(metadata + ["channel"])).alias("pixel_freq")
                              ))
    return metadata, pixel_count_df_grouped


@app.cell
def __(alt, metadata, pixel_count_df_grouped):
    ax = (alt.Chart(pixel_count_df_grouped)
     .mark_bar(
         opacity=0.3,
         binSpacing=0
    ).encode(
        x=alt.X('bin_start:Q', bin='binned', title='Value Range'),
        x2='bin_end:Q',  # Use x2 for the bin range
        y=alt.Y('pixel_freq:Q', title='Frequency').stack(None),
        column=alt.Column('channel:N', title='Channel'),
        color=alt.Color(f"{metadata[0]}:N", title=f"{metadata[0]}")
        # row=alt.Row(f"{metadata[0]}:N", title=f"{metadata[0]}")
    ).properties(
        title='Histogram'
        # width=600,
        # height=400
    ))
    ax
    return (ax,)


@app.cell
def __():
    # import seaborn as sns
    # col_to_plot = "Metadata_Source"
    # # Modify here
    # data_type = table.data.loc[:,col_to_plot].iloc[:8].values
    # pixels = raw_images.reshape(8,raw_images.shape[1], -1).swapaxes(0,1)
    # d = {f"{i}_{j}":[] for i in range(len(pixels)) for j in range(pixels.shape[1])}
    # for i in range(len(pixels)):
    #     for j in range(pixels.shape[1]):
    #         d[f"{i}_{j}"] += pixels[i,j].reshape(-1).tolist()
    # map_channel_id = {i:ch for i, ch in enumerate(map_channel_default.keys()) if channel_checkboxes[i].value}
    # map_activated_label = {i:v for i,v in enumerate(map_channel_id.values())}
    # selected_df=(pl.DataFrame(d).unpivot()
    #                      .with_columns(pl.col("variable")
    #                                    .str.split("_").list.to_struct(n_field_strategy="max_width"))
    #                      .unnest("variable").with_columns(pl.col("field_0").replace(map_activated_label).alias("Channel"), 
    #                                                       pl.col("field_1").replace({i:v for i,v in enumerate(data_type)}).alias("Source")))

    # ax = sns.displot(selected_df.to_pandas(), x="value", col="Channel", hue="Source")
    # plt.ylim(0,2000)
    # ax
    return


@app.cell
def __(
    channel_multiselect,
    channel_to_indices,
    channel_to_rgb,
    chart,
    clip_outliers,
    get_channel,
    load_image_channel,
    low_high_clip,
    max_images_,
    mo,
    normalize,
    np,
    table,
):
    import math
    import matplotlib.pyplot as plt

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
        lower_percentile, upper_percentile = low_high_clip.value
        lower_bound = np.percentile(image, lower_percentile, axis=(-2, -1), keepdims=True)
        upper_bound = np.percentile(image, upper_percentile, axis=(-2, -1), keepdims=True)

        # Clip the image values to the computed thresholds
        clipped_image = np.clip(image, lower_bound, upper_bound)
        if normalize.value:
            clipped_image = (clipped_image - lower_bound) / (upper_bound - lower_bound)
        return clipped_image

    # Function to show images with the selected channels
    def show_images(indices, max_images=8):
        indices = indices[:max_images]

        selected_channel = get_channel()
        selected_channel_indices = [channel_to_indices[ch] for ch in selected_channel]

        # Load the images and process them
        raw_images = np.stack([load_image_channel(x, tuple(selected_channel_indices)) for x in indices], axis=0)

        if clip_outliers.value:
            images = channel_to_rgb(clip_outliers_(raw_images), selected_channel)
        else:
            images = channel_to_rgb(raw_images, selected_channel)

        fig, axes = plt.subplots(min(max(2, int(np.ceil(len(indices) / 8))), len(indices)), math.ceil(len(indices)/ int(max(2, np.ceil(len(indices) / 8)))))
        fig.set_size_inches(20, 5)

        if len(indices) > 1:
            for im, ax in zip(images, axes.T.flat):
                ax.imshow(im.transpose(1, 2 ,0))  # Display the image with the combined RGB channels
            for ax in axes.T.flat:
                ax.set_axis_off()
        else:
            axes.imshow(images[0].transpose(1, 2, 0))  # If there's only one image
            axes.set_axis_off()

        fig.tight_layout()
        return fig

    indices = (list(chart.value["img_id"]) if not len(table.value) else list(table.value["img_id"]))

    if len(indices) > 0:
        selected_images = show_images(indices, max_images=max_images_.value)
    else:
        selected_images = None
    mo.md(
        (
        f"""
        **Here's a preview of the images you've selected**:

        **Max Image:**
        {max_images_}

        **Select Channels:**
        {channel_multiselect}

        **Clip Outlier Pixels:**

        {mo.vstack([
            mo.hstack([mo.md(f"**Percentile range:**"), low_high_clip, mo.md(f"**{low_high_clip.value}**")], justify="start"), 
            mo.hstack([clip_outliers, normalize, mo.md(f"**{normalize.value}**")], justify="start")
        ])}

        {mo.as_html(selected_images)}

        Here's all the data you've selected.

        {table}
        """
    ) if len(indices) > 0 else "**If you select embedding, you will see the corresponding images.**")
    return clip_outliers_, indices, math, plt, selected_images, show_images


@app.cell
async def __():
    import sys

    if "pyodide" in sys.modules:
        import micropip

        await micropip.install("altair")

    import altair as alt
    # alt.data_transformers.enable("vegafusion")
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
