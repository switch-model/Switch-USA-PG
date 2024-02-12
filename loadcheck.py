from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from powergenome.generators import GeneratorClusters
from powergenome.GenX import reduce_time_domain
from powergenome.load_profiles import make_final_load_curves
from powergenome.params import DATA_PATHS
from powergenome.util import (
    build_scenario_settings,
    init_pudl_connection,
    load_settings,
    check_settings,
)
from powergenome.external_data import (
    make_demand_response_profiles,
    make_generator_variability,
)

pd.options.display.max_columns = 200

cwd = Path.cwd()

settings_path = cwd / "settings_TD_east.yml"
settings = load_settings(settings_path)
settings["input_folder"] = settings_path.parent / settings["input_folder"]
scenario_definitions = pd.read_csv(
    settings["input_folder"] / settings["scenario_definitions_fn"]
)
scenario_settings = build_scenario_settings(settings, scenario_definitions)

pudl_engine, pudl_out, pg_engine = init_pudl_connection(
    freq="AS",
    start_year=min(settings.get("data_years")),
    end_year=max(settings.get("data_years")),
)

s = """
        SELECT time_index, region_id_epaipm, load_mw, year
        FROM load_curves_ferc
    """
load_curves_ferc = pd.read_sql_query(s, pg_engine)
load_wecc2012 = load_curves_ferc.loc[
    load_curves_ferc["region_id_epaipm"].str.contains("WEC")
]
load_wecc2012
load2012 = load_curves_ferc.groupby("time_index").agg(
    {"load_mw": "sum", "year": "first"}
)
s = """
        SELECT time_index, region, sector, subsector, load_mw, year
        FROM load_curves_nrel_efs
    """

#     WHERE region in ('WEC_BANC', 'WEC_CALN', 'WEC_LADW', 'WEC_SDGE', 'WECC_AZ',
#    'WECC_CO', 'WECC_ID', 'WECC_IID', 'WECC_MT', 'WECC_NM', 'WECC_NNV',
#    'WECC_PNW', 'WECC_SCE', 'WECC_SNV', 'WECC_UT', 'WECC_WY')
load_curves_efs = pd.read_sql_query(s, pg_engine)
# load_wecc2019 = load_curves_efs.loc[load_curves_efs["region"].str.contains("WEC")]
# load2019 = load_wecc2019.groupby("time_index").agg({"load_mw": "sum", "year": "first"})
load2019 = load_curves_efs.copy()


def get_month_group(hours):
    if hours < 744:
        return "Jan"
    if hours >= 744 and hours < 1416:
        return "Feb"
    if hours >= 1416 and hours < 2160:
        return "Mar"
    if hours >= 2160 and hours < 2880:
        return "Apr"
    if hours >= 2880 and hours < 3624:
        return "May"
    if hours >= 3624 and hours < 4344:
        return "Jun"
    if hours >= 4344 and hours < 5088:
        return "Jul"
    if hours >= 5088 and hours < 5832:
        return "Aug"
    if hours >= 5832 and hours < 6552:
        return "Sep"
    if hours >= 6552 and hours < 7296:
        return "Oct"
    if hours >= 7296 and hours < 8016:
        return "Nov"
    if hours >= 8016 and hours <= 8760:
        return "Dec"


hourly2019 = load2019.groupby("time_index", as_index=False).agg(
    {"load_mw": "sum", "time_index": "first", "year": "first"}
)

hourly2019["month"] = hourly2019["time_index"].apply(get_month_group)
hourly2019["week"] = hourly2019.index.repeat(168)[: len(hourly2019)]
hourly2019["week"] = hourly2019["week"] + 1
hourly2019.to_csv(
    "/Users/rangrang/Desktop/transmission_project/demand2019.csv", index=False
)

w_2019 = hourly2019.groupby("week", as_index=False).agg(
    {"load_mw": "sum", "month": "first", "year": "first"}
)
w_2019.to_csv(
    "/Users/rangrang/Desktop/transmission_project/weekly_demand2019.csv", index=False
)
m_2019 = hourly2019.groupby("month", as_index=False).agg(
    {"load_mw": "sum", "month": "first", "year": "first"}
)
# load_wecc2019 = load_curves_efs.loc[load_curves_efs["region"].str.contains("WEC")]
# load2019 = load_wecc2019.groupby("time_index").agg({"load_mw": "sum", "year": "first"})
m_2019.to_csv(
    "/Users/rangrang/Desktop/transmission_project/montly_demand2019.csv", index=False
)


############# load 2050
import pandas as pd
import os

# import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pytz import timezone
import pytz

loads = pd.read_csv(
    "/Users/rangrang/Dropbox/PowerGenomeSWITCH_documentation r1/Jupyter Notebooks/test_full/p1/loads.csv"
)
timepoints = pd.read_csv(
    "/Users/rangrang/Dropbox/PowerGenomeSWITCH_documentation r1/Jupyter Notebooks/test_full/p1/timepoints.csv"
)
timeseries = pd.read_csv(
    "/Users/rangrang/Dropbox/PowerGenomeSWITCH_documentation r1/Jupyter Notebooks/test_full/p1/timeseries.csv"
)

timepoints_weighted = pd.merge(
    left=timepoints,
    right=timeseries,
    on=["timeseries"],
    validate="many_to_one",
    how="left",
)

timepoints_weighted = timepoints_weighted.rename({"timepoint_id": "TIMEPOINT"}, axis=1)

loads
timepoints
loads_new = pd.merge(
    left=loads,
    right=timepoints_weighted,
    on=["TIMEPOINT"],
    validate="many_to_one",
    how="left",
)

loads_new["demand_mw"] = (
    loads_new["zone_demand_mw"] * loads_new["ts_scale_to_period"] / 10
)
loads_new["period"] = loads_new["timestamp"].astype(str).str[:4]
loads_new = loads_new.loc[loads_new["period"] == "2050"]
loads_new["hour"] = loads_new["TIMEPOINT"] - 17520

toplot = loads_new.groupby("hour", as_index=False).agg(
    {"demand_mw": "sum", "hour": "first"}
)
import pylab as plt

X = range(1, 8761)
Y1 = load2012["load_mw"].iloc[
    0:8760,
]
# Y2 = load2019["load_mw"]
Y3 = toplot["demand_mw"]
plt.xlim([1, 8760])
plt.scatter(X, Y1 / 1000, color="yellow", s=2)
# plt.scatter(X, Y2 / 1000, color="g", s=2)
plt.scatter(X, Y3 / 1000, color="pink", s=2)
plt.title("Annual demand -- wecc")
# Label configuration
plt.xlabel("Hours in a year", fontsize=9)
plt.ylabel("GW", fontsize=9)
plt.yticks(fontsize=9)
plt.xticks(fontsize=9)
# plt.legend(["2012", "2019", "2050"])
plt.legend(["2012", "2050"])

plt.show()
