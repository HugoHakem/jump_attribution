"""
Convert csv to parquet for ease of plotting.
Note on naming: always replace special signs (.=-) before saving files.
"""
from pathlib import Path
import polars as pl
from umap import UMAP
from sklearn.decomposition import PCA

dir_path = Path("/home/hhakem/projects/counterfactuals_projects/workspace/analysis/image_active_crop_dataset")
fname = "embedding_VGG_image_crop_active_groups_fold_1epoch=77-train_acc=0.86-val_acc=0.77.csv"
# shared_working_dir = Path("/datastore/shared/attribution/data/")
data = pl.read_csv(dir_path / fname)

reducer_pca = PCA(100)
reducer_umap = UMAP()
embedding_XY = reducer_umap.fit_transform(
    reducer_pca.fit_transform(
        data.select(pl.all().exclude("pred")).to_numpy()))

metadata = pl.read_csv(dir_path / "metadata.csv")
metadata_moa = pl.read_csv("target2_eq_moa2_active_metadata")

new_data = pl.concat(
    (
        metadata.join(
            metadata_moa.select(
                ["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well",
                 "pert_iname", "target", "moa", "moa_id", "Metadata_InChIKey", "inchi_id"]
            ),
            on=["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well"],
            how="left"),
        data.select(pl.col("pred")),
        pl.DataFrame(embedding_XY, schema=["UMAP1", "UMAP2"])
    ),
    how="horizontal")
# new_data = pl.concat((metadata, data.select(pl.col("^column_[123]$")).rename(lambda x: f"PC{x[-1]}")), how="horizontal")
new_data.write_parquet(dir_path / "embedding_UMAP.parquet")
