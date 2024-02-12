import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pytz import timezone
import pytz
from datetime import date


def tp_to_date(df: pd.DataFrame, tp_label: str):
    sources_nation = df.copy()
    if "period" in sources_nation.columns:
        print("period is a column of the dataframe")
    else:
        sources_nation["period"] = [
            pd.to_numeric(x[0]) for x in sources_nation[tp_label].str.split("_")
        ]
    sources_nation["week"] = [
        pd.to_numeric(x[1][1:]) for x in sources_nation[tp_label].str.split("_")
    ]
    sources_nation["hour"] = [
        pd.to_numeric(x[2]) for x in sources_nation[tp_label].str.split("_")
    ]
    #  hours ro days
    def cat(row):
        if row["hour"] <= 23:
            return 1
        elif row["hour"] > 23 and row["hour"] <= 47:
            return 2
        elif row["hour"] > 47 and row["hour"] <= 71:
            return 3
        elif row["hour"] > 71 and row["hour"] <= 95:
            return 4
        elif row["hour"] > 95 and row["hour"] <= 119:
            return 5
        elif row["hour"] > 119 and row["hour"] <= 143:
            return 6
        return 7

    sources_nation["days"] = sources_nation.apply(lambda row: cat(row), axis=1)
    # sources_nation["days"] = pd.to_numeric(sources_nation["days"])
    sources_nation = sources_nation.sort_values(by=["week", "hour"])

    # sources_nation = sources_nation.astype(
    #     {"days": "int64", "period": "int64", "week": "int64"}
    # )
    return sources_nation


def tech_type_group(df: pd.DataFrame):
    df["tech_type"] = "Other"
    df.loc[
        df["resource_name"].str.contains("hydrogen", case=False), "tech_type"
    ] = "Hydrogen"
    df.loc[
        df["resource_name"].str.contains("batter", case=False), "tech_type"
    ] = "Battery"
    df.loc[
        df["resource_name"].str.contains("storage", case=False), "tech_type"
    ] = "Battery"
    df.loc[df["resource_name"].str.contains("coal", case=False), "tech_type"] = "Coal"
    df.loc[
        df["resource_name"].str.contains("solar|pv", case=False), "tech_type"
    ] = "Solar"
    df.loc[df["resource_name"].str.contains("wind", case=False), "tech_type"] = "Wind"
    df.loc[
        df["resource_name"].str.contains("hydro|water", case=False), "tech_type"
    ] = "Hydro"
    df.loc[
        df["resource_name"].str.contains("distributed", case=False), "tech_type"
    ] = "Distributed Solar"
    df.loc[
        df["resource_name"].str.contains("geothermal", case=False), "tech_type"
    ] = "Geothermal"
    df.loc[
        df["resource_name"].str.contains("nuclear", case=False), "tech_type"
    ] = "Nuclear"
    df.loc[
        df["resource_name"].str.contains("natural", case=False), "tech_type"
    ] = "Natural Gas"
    df.loc[df["resource_name"].str.contains("ccs", case=False), "tech_type"] = "CCS"
    df.loc[
        df["resource_name"].str.contains("bio", case=False), "tech_type"
    ] = "Bio Solid"

    return df
