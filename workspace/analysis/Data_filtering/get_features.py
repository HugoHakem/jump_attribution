import polars as pl
from pyarrow.dataset import dataset
from s3fs import S3FileSystem


_PREFIX = (
    "s3://cellpainting-gallery/cpg0016-jump-assembled/source_all/workspace/profiles"
)
_RECIPE = "jump-profiling-recipe_2024_a917fa7"

transforms = (
    (
        "CRISPR",
        "profiles_wellpos_cc_var_mad_outlier_featselect_sphering_harmony_PCA_corrected",
    ),
    ("ORF", "profiles_wellpos_cc_var_mad_outlier_featselect_sphering_harmony"),
    ("COMPOUND", "profiles_var_mad_int_featselect_harmony"),
)

filepaths = {
    dset: f"{_PREFIX}/{_RECIPE}/{dset}/{transform}/profiles.parquet"
    for dset, transform in transforms
}


def lazy_load(path: str) -> pl.LazyFrame:
    fs = S3FileSystem(anon=True)
    myds = dataset(path, filesystem=fs)
    df = pl.scan_pyarrow_dataset(myds)
    #source 13 and 7 are the same. 
    #source 9 and 1 are not comparable to the others due to different plate type with more wells
    df = df.with_columns(
        pl.when(pl.col("Metadata_Source").str.contains("_13$"))
        .then(pl.lit("source_7"))
        .otherwise(pl.col("Metadata_Source"))
        .alias("Metadata_Source"))
    df = df.filter(pl.col("Metadata_Source").str.contains("_9|_1$") != True)
    return df

def load_features(table_name: str,
                  metadata: pl.DataFrame) -> pl.DataFrame:
    """
    Return the profiles or feature table of the "metadata" dataframe passed
    retirved from the feature table named after "table_name"
    ----------
    Parameters: 
        table_name(str): The wished table from which the features should be retrieved.
          -> must be in {"CRISPR",
                         "ORF",
                         "COMPOUND"
                         }
        metadata(pl.DataFrame): The metadata polar dataframe with at least the 
        following info for the join: [Metadata_Source,
                                      Metadata_Plate,
                                      Metadata_Well,
                                      Metadata_JCP2022
                                      ]
    ----------
    Return:
        pl.DataFrame
    """
    
    data = lazy_load(filepaths[table_name])
    return data.join(metadata.select(pl.col(["Metadata_Source",
                                      "Metadata_Plate",
                                      "Metadata_Well",
                                      "Metadata_JCP2022"
                                      ])),
                     on=["Metadata_Source",
                         "Metadata_Plate",
                         "Metadata_Well",
                         "Metadata_JCP2022"],
                     how="inner")
                              







