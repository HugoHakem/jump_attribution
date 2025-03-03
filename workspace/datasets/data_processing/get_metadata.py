from functools import cache

import polars as pl
import pooch

@cache
def get_table(table_name: str) -> pl.DataFrame:
    """
    Downloads and caches a dataset from the JUMP Cell Painting repository, 
    returning it as a Polars DataFrame.

    This function retrieves metadata tables from the JUMP Cell Painting project 
    hosted on GitHub and other sources. The files are cached for efficiency.

    Args:
        table_name (str): 
            The name of the table to retrieve. Must be one of:
            - "compound"
            - "well"
            - "plate"
            - "moa"
            - "microscope_config"
            - "target2"
            - "target2_plate"

    Returns:
        pl.DataFrame:
            A Polars DataFrame containing the requested dataset.

    Notes:
        - If `table_name` is "moa", the data is fetched from an external S3 source.
        - Other datasets are retrieved from GitHub and verified using precomputed hashes.
        - Data is stored locally in cache using `pooch` to avoid redundant downloads.
    """

    METADATA_LOCATION = "https://github.com/jump-cellpainting/"
    
    if table_name == "moa":
        return (pl.read_csv(
            "https://s3.amazonaws.com/data.clue.io/repurposing/downloads/repurposing_drugs_20200324.txt",
            separator="\t",
            comment_prefix="!")
                .select(pl.col("pert_iname", "moa")).unique().drop_nulls())
        
    elif table_name == "target2":
        METADATA_LOCATION += "JUMP-Target/raw/master/JUMP-Target-2_compound_metadata.tsv"
        
    elif table_name == "target2_plate":
        METADATA_LOCATION += "JUMP-Target/raw/master/JUMP-Target-2_compound_platemap.tsv"
        
    elif table_name == "microscope_config": 
        METADATA_LOCATION += f"datasets/raw/main/metadata/{table_name}.csv"
        
    else:
        METADATA_LOCATION += f"datasets/raw/main/metadata/{table_name}.csv.gz"


    
    METAFILE_HASH = {
        "compound": "03c63e62145f12d7ab253333b7285378989a2f426e7c40e03f92e39554f5d580",
        "crispr": "55e36e6802c6fc5f8e5d5258554368d64601f1847205e0fceb28a2c246c8d1ed",
        "orf": "9c7ec4b0fa460a3a30f270a15f11b5e85cef9dd105c8a0ab8ab50f6cc98894b8",
        "well": "677d3c1386d967f10395e86117927b430dca33e4e35d9607efe3c5c47c186008",
        "plate": "745391d930627474ec6e3083df8b5c108db30408c0d670cdabb3b79f66eaff48",
        "microscope_config": "bf589c5e8cc79b64a3f8ad1436422e32bbf7d746444c638efd156a21ed4af916",
        "target2": "d8e7820746cbc203597b7258c7c3659b46644958e63c81d9a96cb137d8f747ef",
        "target2_plate": "60ac7533b23d2bf94a06f4e1e85ae9f7f6c8c819ca1dc393c46638eab1da0b56"
    }
    
    
    return pl.read_csv(
        pooch.retrieve(
            url=METADATA_LOCATION,
            known_hash=METAFILE_HASH[table_name],
        ),
        use_pyarrow=True,
        separator="\t" if METADATA_LOCATION[-3::] == "tsv" else ","
    )