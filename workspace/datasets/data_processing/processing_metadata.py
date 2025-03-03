import polars as pl

def same_plate_trt_control(metadata: pl.DataFrame) -> pl.DataFrame:
    """
    Filters the metadata to retain only samples that share the 
    same plate and batch as treatment (trt) samples.

    This function ensures that only metadata entries corresponding to the same 
    `Metadata_Plate` and `Metadata_Batch` as treatment samples (`pert_type == "trt"`) 
    are included in the final dataset.

    Args:
        metadata (pl.DataFrame): A Polars DataFrame containing experimental metadata.

    Returns:
        pl.DataFrame: A filtered Polars DataFrame containing only entries that share 
                      a plate and batch with treatment samples.
    """
    return (
        metadata.join(
            metadata
            .filter(pl.col("pert_type") == "trt")
            .select("Metadata_Plate")
            .unique(),
            on="Metadata_Plate",
            how="inner"
        )
    )
