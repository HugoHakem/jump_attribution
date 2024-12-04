import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

def show_distribution(metadata: pl.DataFrame):
    """
    Plot 6 graphs: 
        top left: barplot, number of samples per sources
        top right: barplot, number of samples per type of microscope
        bottom left: barplot, number of unique compounds per sources
        bottom right: barplot, number of unique compounds per type of microscope
        big image 1: heatmap, number of samples per well
        big image 2: heatmap, number of unique compounds per well
    ---------------
    Parameters:
        metadata(pl.DataFrame): metadata dataframe with at least the following columns: 
            Metadata_Source
            Metadata_JCP2022
            Micro_id (an id which identify the different configuration of microscope)
            Metadata_Well  
    """

    
    info_per_source = (metadata.group_by("Metadata_Source")
                         .agg(
                             pl.col("Metadata_JCP2022").n_unique().alias("n_compound"),
                             pl.col("Metadata_JCP2022").count().alias("n_sample"))
                       .sort(by=pl.col("Metadata_Source").str.extract("\D+_(\d+)")))
     
    info_per_micro = (metadata.group_by("Micro_id")
                         .agg(
                             pl.col("Metadata_JCP2022").n_unique().alias("n_compound"),
                             pl.col("Metadata_JCP2022").count().alias("n_sample"))
                       .sort(by="Micro_id"))
    

    info_per_well = (metadata.group_by("Metadata_Well")
                     .agg(
                         pl.col("Metadata_JCP2022").n_unique().alias("n_compound"),
                         pl.col("Metadata_JCP2022").count().alias("n_sample"))
                   .sort(by="Metadata_Well").select(
                       pl.col("Metadata_Well").str.extract("(\D+)").alias("Well_letter"),
                       pl.col("Metadata_Well").str.extract("\D+(\d+)").alias("Well_numb"),
                       pl.col("n_compound", "n_sample")))

    
    fig, axes = plt.subplots(2,2, figsize=(20, 10))
    info_per_group = [info_per_source, info_per_micro]
    group = ["Metadata_Source", "Micro_id"]
    info = ["n_sample", "n_compound"]
    for i in range(2):
        for j in range(2):
            sns.barplot(info_per_group[j],
                       x=group[j],
                       y=info[i],
                       ax=axes[i][j])
            axes[i][j].tick_params(axis='x', rotation=90)


    sample_well_map = info_per_well.pivot(index="Well_letter",
                                    columns="Well_numb",
                                    values="n_sample",
                                    sort_columns=True)
    compound_well_map = info_per_well.pivot(index="Well_letter",
                                    columns="Well_numb",
                                    values="n_compound",
                                    sort_columns=True)
    
    
    fig2, ax2 = plt.subplots(1, figsize=(20,10))
    sns.heatmap(sample_well_map.select(pl.all().exclude("Well_letter")),
                ax=ax2)
    ax2.set_xticklabels(sample_well_map.columns[1:])
    ax2.set_yticklabels(sample_well_map.select(pl.col("Well_letter")).to_numpy().reshape(-1))
    ax2.set_title("n_sample")

    
    fig3, ax3 = plt.subplots(1, figsize=(20,10))
    sns.heatmap(compound_well_map.select(pl.all().exclude("Well_letter")),
                ax=ax3)
    ax3.set_xticklabels(compound_well_map.columns[1:])
    ax3.set_yticklabels(compound_well_map.select(pl.col("Well_letter")).to_numpy().reshape(-1))
    ax3.set_title("n_compound")




def compound_info(metadata: pl.DataFrame):
    """
    Plot 2 graphs: 
        top: barplot, count of unique compounds (y axis) 
              who has a certain amount of sample (x axis)
        bottom: barplot, count of unique compounds (y axis) 
                who has a been tested in a certain amount of wells (x axis)

    ---------------
    Parameters:
        metadata(pl.DataFrame): metadata dataframe with at least the following columns: 
            Metadata_Source
            Metadata_JCP2022
            Metadata_InChIKey
            Metadata_Well  
    """
    compounds_info = (metadata.group_by("Metadata_JCP2022")
                      .agg(pl.col("Metadata_InChIKey").count().alias("Sample_count"),
                           pl.col("Metadata_Source", "Metadata_Well", "Micro_id")
                           .n_unique().name.prefix("Unique_"))
                           )
    fig, ax1 = plt.subplots(1, figsize=(16,8))
    
    df = (compounds_info.group_by("Sample_count")
                 .agg(pl.col("Metadata_JCP2022").n_unique()).sort(by="Sample_count"))
    
    sns.barplot(df,
                x="Sample_count",
                y="Metadata_JCP2022",
                ax=ax1)
    ax1.tick_params(axis='x', rotation=90)
    
    fig, ax2 = plt.subplots(1, figsize=(16,8))
    
    df2 = (compounds_info.group_by("Unique_Metadata_Well")
                 .agg(pl.col("Metadata_JCP2022").n_unique()).sort(by="Unique_Metadata_Well"))
    
    sns.barplot(df2,
                x="Unique_Metadata_Well",
                y="Metadata_JCP2022",
                ax=ax2)
    ax2.tick_params(axis='x', rotation=90)


def moa_distribution(metadata: pl.DataFrame, key: str="moa"):
    """
    Plot 2 graphs: 
        left: barplot, number of samples (y axis) per moa (x axis)

        right: barplot, number of unique compound (y axis) per moa (x axis)

    ---------------
    Parameters:
        metadata(pl.DataFrame): metadata dataframe with at least the following columns: 
            Metadata_Source
            Metadata_JCP2022
            Metadata_InChIKey
            Metadata_Well
        key(str): the key to group by with
    """

    moa_info = (metadata.group_by(key).agg(
        pl.col("Metadata_JCP2022").count().alias("Sample_count"),
        pl.col("Metadata_JCP2022").n_unique().alias("Compound_count")))
    
    fig, axes = plt.subplots(2, 1, figsize=(13,13))
    y_axis = ["Sample_count", "Compound_count"]
    for i, y in enumerate(y_axis):
        sns.barplot(moa_info,
                    x=key,
                    y=y,
                    ax=axes[i])
        axes[i].tick_params(axis='x', rotation=90)
        axes[i].set_xticklabels([])


    
def replicate_per_source_per_comp(metadata: pl.DataFrame):
    """
    Plot 1 graphs: 
        heatmap, number of samples (y axis) per moa (x axis)

    ---------------
    Parameters:
        metadata(pl.DataFrame): metadata dataframe with at least the following columns: 
            Metadata_Source
            Metadata_JCP2022
            Metadata_InChIKey
            Metadata_Well
        key(str): the key to group by with
    """
    fig, ax = plt.subplots(1, figsize=(20,10))
    pivot_compound_source = (metadata.group_by("Metadata_InChIKey", "Metadata_Source").agg(
        pl.col("Metadata_JCP2022").count())
        .pivot(index="Metadata_InChIKey", 
               columns="Metadata_Source", 
               values="Metadata_JCP2022"))
    
    sns.heatmap(pivot_compound_source.select(pl.all().exclude("Metadata_InChIKey")),
                ax=ax,
                annot=True)
    ax.set_xticklabels(pivot_compound_source.columns[1:])
    ax.set_yticks(ticks=np.arange(pivot_compound_source.shape[0]),
                  labels=(pivot_compound_source.select(pl.col("Metadata_InChIKey").str.extract(("^(\w+)-")))
                                                     .to_numpy().reshape(-1)))
    ax.tick_params(axis='y', rotation=0, labelsize='xx-small')
    ax.set_title("count replicate per compound per source")

