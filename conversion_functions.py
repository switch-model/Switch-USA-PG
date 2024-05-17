"""
Functions to convert data from PowerGenome for use with Switch
"""

from statistics import mean, mode
from typing import List

import numpy as np
import pandas as pd
import math

from powergenome.time_reduction import kmeans_time_clustering

km_per_mile = 1.60934


# convenience functions to get first/final keys/values from dicts
# (e.g., first year in a dictionary organized by years)
# note: these use the order of creation, not lexicographic order
def first_key(d: dict):
    return next(iter(d.keys()))


def first_value(d: dict):
    return next(iter(d.values()))


def final_key(d: dict):
    return next(reversed(d.keys()))


def final_value(d: dict):
    return next(reversed(d.values()))


def switch_fuel_cost_table(
    aeo_fuel_region_map, fuel_prices, IPM_regions, scenario, year_list
):
    """
    Create the fuel_cost input file based on REAM Scenario 178.
    Inputs:
        * aeo_fuel_region_map: has aeo_fuel_regions and the ipm regions within each aeo_fuel_region
        * fuel_prices: output from PowerGenome gc.fuel_prices
        * IPM_regions: from settings('model_regions')
        * scenario: filtering the fuel_prices table. Suggest using 'reference' for now.
        * year_list: the periods - 2020, 2030, 2040, 2050.  To filter the fuel_prices year column
    Output:
        the fuel_cost_table
            * load_zone: IPM region
            * fuel: based on PowerGenome fuel_prices table
            * period: based on year_list
            * fuel_cost: based on fuel_prices.price
    """

    ref_df = fuel_prices.copy()
    ref_df = ref_df.loc[
        ref_df["scenario"].isin(scenario)
    ]  # use reference scenario for now
    ref_df = ref_df[ref_df["year"].isin(year_list)]
    ref_df = ref_df.drop(["full_fuel_name", "scenario"], axis=1)

    # loop through aeo_fuel_regions.
    # for each of the ipm regions in the aeo_fuel, duplicate the fuel_prices table while adding ipm column
    fuel_cost = pd.DataFrame(columns=["year", "price", "fuel", "region", "load_zone"])
    data = list()
    for region in aeo_fuel_region_map.keys():
        df = ref_df.copy()
        # lookup all fuels available in this region or with no region specified
        # (generally user-defined fuels added earlier)
        df = df[df["region"].isin({region, ""})]
        for ipm in aeo_fuel_region_map[region]:
            ipm_region = ipm
            df["load_zone"] = ipm_region
            fuel_cost = fuel_cost.append(df)
    #     fuel_cost = fuel_cost.append(data)
    fuel_cost.rename(columns={"year": "period", "price": "fuel_cost"}, inplace=True)
    fuel_cost = fuel_cost[["load_zone", "fuel", "period", "fuel_cost"]]
    fuel_cost["period"] = fuel_cost["period"].astype(int)
    fuel_cost = fuel_cost[fuel_cost["load_zone"].isin(IPM_regions)]
    fuel_cost["fuel"] = fuel_cost[
        "fuel"
    ].str.capitalize()  # align with energy_source in gen_pro_info? switch error.
    return fuel_cost


def switch_fuels(fuel_prices, REAM_co2_intensity):
    """
    Create fuels table using fuel_prices (from gc.fuel_prices) and basing other columns on REAM scenario 178
    Output columns
        * fuel: based on the fuels contained in the PowerGenome fuel_prices table
        * co2_intensity: based on REAM scenario 178
        * upstream_co2_intensity: based on REAM scenario 178
    """
    fuels = pd.DataFrame(fuel_prices["fuel"].unique(), columns=["fuel"])
    fuels["co2_intensity"] = fuels["fuel"].apply(lambda x: REAM_co2_intensity[x])
    fuels["upstream_co2_intensity"] = 0  # based on REAM scenario 178
    # switch error - capitalize to align with gen pro info energy_source?
    fuels["fuel"] = fuels["fuel"].str.capitalize()
    return fuels


def create_dict_plantgen(df, column):
    """
    Create dictionary from two columns, removing na's beforehand
    {plant_gen_id: year}
    """
    df = df[df[column].notna()]
    ids = df["plant_gen_id"].to_list()
    dates = df[column].to_list()
    dictionary = dict(zip(ids, dates))
    return dictionary


def plant_gen_id(df):
    """
    Create unique id for generator by combining plant_id_eia and generator_id
    """
    plant_id_eia = df["plant_id_eia"]
    df["plant_gen_id"] = plant_id_eia.astype(str) + "_" + df["generator_id"].astype(str)
    return df


def add_generic_gen_build_info(units, settings):
    """
    Update units dataframe with synthetic plant_gen_id, build_year, capacity_mw
    and capacity_mwh for generic existing generators. These are generators that
    PowerGenome reported as existing but didn't get unit-level construction info
    from eia_build_info(), e.g., distributed generation.
    """
    # Distributed generation and a few others are in the "existing" list with
    # non-zero Existing_Cap_MW, but no plant_id_eia and therefore no build_year
    # or capacity_mw.
    # TODO: fix this upstream in PowerGenome, e.g., create dummy plant
    # construction info there in addition to the overall project info

    # We could assess the amount of capacity available for each Resource in each
    # model_year, then create records with various build_year and capacity
    # values that will result in the specified amount of capacity being
    # available in each year. But this doesn't have a simple solution, because
    # capacity is aging out at the same time as it is being added, so we'd
    # probably need to write a short linear program like "min sum(build[g,
    # y]*(2050-y) for g in gens, y in {1950...2050}), such that known_online[g,
    # y] = sum(build[g, y'] for y' in {y-max_age..y}), y in {2030, 2040, 2050}".
    # This would work for multi-period (foresight) models, but then we'd also
    # need some way to coordinate these choices between the separate periods of
    # myopic models.

    # Instead of that, we assume the plants were built in an arbitrary recent
    # year. That way they will persist through the study (if long-lived) and
    # the build_year will be consistent across multiple stages of myopic models.

    generic = units["existing"] & units["build_year"].isna()

    units.loc[generic, "plant_gen_id"] = "generic"
    # somewhat arbitrary, but should in the right range and stable across model
    # years
    units.loc[generic, "build_year"] = settings.get(
        "eia_data_year",
        settings.get("data_year", settings.get("atb_data_year", 2020)),
    )
    units.loc[generic, "capacity_mw"] = units.loc[generic, "Existing_Cap_MW"]
    units.loc[generic, "capacity_mwh"] = units.loc[generic, "Existing_Cap_MWh"]

    return units


def gen_info_table(
    all_gens,
    spur_capex_mw_mile,
):
    """
    Create the gen_info table
    Inputs:
        * gens: from PowerGenome gc.create_all_generators() with some extra data
        * spur_capex_mw_mile: based on the settings file ('transmission_investment_cost')['spur']['capex_mw_mile']
        * cogen_tech, baseload_tech, energy_tech, sched_outage_tech, forced_outage_tech
            - these are user defined dictionaries.  Will map values based on the technology
    Output columns:
        * GENERATION_PROJECT: basing on index
        * gen_tech: based on technology
        * gen_energy_source: based on energy_tech input
        * gen_load_zone: IPM region
        * gen_max_age: based on retirement_age
        * gen_can_retire_early: based on Can_Retire and/or New_Build from PowerGenome
        * gen_is_variable: only solar and wind are true
        * gen_is_baseload: from PowerGenome
        * gen_full_load_heat_rate: based on Heat_Rate_MMBTU_per_MWh from all_gen
        * gen_variable_om: based on Var_OM_Cost_per_MWh_mean from all_gen
        * gen_connect_cost_per_mw: based on spur_capex_mw_mile * spur_miles plus substation cost
        * gen_dbid: same as generation_project
        * gen_scheduled_outage_rate: from PowerGenome
        * gen_forced_outage_rate: from PowerGenome
        * gen_capacity_limit_mw: omitted for new thermal plants; upper limits on new renewables (MW total across all).
        * gen_min_load_fraction: from PowerGenome
        * gen_ramp_limit_up: from PowerGenome
        * gen_ramp_limit_down: from PowerGenome
        * gen_min_uptime: from PowerGenome
        * gen_min_downtime: from PowerGenome
        * gen_startup_om: from PowerGenome
        * gen_is_cogen: from PowerGenome
        * gen_storage_efficiency: from PowerGenome
        * gen_store_to_release_ratio: batteries use 1
        * gen_can_provide_cap_reserves: all 1s
        * gen_self_discharge_rate, gen_storage_energy_to_power_ratio: blanks

    """

    gen_info = all_gens.copy().reset_index(drop=True)

    gen_info["GENERATION_PROJECT"] = gen_info["Resource"]
    # TODO Change the upstream powergenome code to set up co2_pipeline_capex_mw as 0 when no access to ccs tech
    # for now, modifyng the translation layer --RR
    if "co2_pipeline_capex_mw" not in gen_info.columns:
        gen_info["co2_pipeline_capex_mw"] = 0

    # Include co2 pipeline costs as part of connection -- could also be in build capex
    gen_info["gen_connect_cost_per_mw"] = gen_info[
        ["spur_capex", "interconnect_capex_mw", "co2_pipeline_capex_mw"]
    ].sum(axis=1)

    # create gen_connect_cost_per_mw from spur_miles and spur_capex_mw_mile
    gen_info["spur_capex_mw_mi"] = gen_info["region"].map(spur_capex_mw_mile)
    gen_info["spur_miles"] = gen_info["spur_miles"].fillna(0)
    gen_info.loc[
        gen_info["gen_connect_cost_per_mw"] == 0, "gen_connect_cost_per_mw"
    ] = (gen_info["spur_capex_mw_mi"] * gen_info["spur_miles"])
    gen_info = gen_info.drop(["spur_miles", "spur_capex_mw_mi"], axis=1)

    # clean up gen_is_variable; usually only solar and wind technologies are true
    gen_info["gen_is_variable"] = gen_info["gen_is_variable"].astype(bool)

    # gen_storage_efficiency and gen_store_to_release_ratio (input vs output MW rating)
    storage_gens = gen_info["STOR"] == 1
    gen_info.loc[storage_gens, "gen_storage_efficiency"] = (
        gen_info["Eff_Up"] * gen_info["Eff_Down"]
    )
    gen_info.loc[storage_gens, "gen_store_to_release_ratio"] = 1
    gen_info["gen_can_provide_cap_reserves"] = 1

    gen_info["gen_self_discharge_rate"] = None
    gen_info["gen_storage_energy_to_power_ratio"] = None

    gen_info["gen_dbid"] = gen_info["GENERATION_PROJECT"]

    # get capacity limit if any, but ignore -1 (existing plants that won't
    # show up as buildable in the future anyway)
    gen_info["gen_capacity_limit_mw"] = gen_info["Max_Cap_MW"].replace({-1: None})

    # fill in CCS capture rate, using NaN for non-CCS plants
    gen_info["gen_ccs_capture_efficiency"] = gen_info["CO2_Capture_Rate"]
    gen_info.loc[
        gen_info["gen_ccs_capture_efficiency"] == 0, "gen_ccs_capture_efficiency"
    ] = None

    # identify generators that can retire early
    try:
        # settings for newer GenX
        gen_info["gen_can_retire_early"] = gen_info["Can_Retire"].astype("Int64")
    except KeyError:
        # settings for older GenX
        gen_info["gen_can_retire_early"] = (gen_info["New_Build"] >= 0).astype("Int64")

    # rename columns
    gen_info.rename(
        columns={
            "technology": "gen_tech",
            "region": "gen_load_zone",
            "retirement_age": "gen_max_age",
            "Heat_Rate_MMBTU_per_MWh": "gen_full_load_heat_rate",
            "Var_OM_Cost_per_MWh_mean": "gen_variable_om",
            # gen_amortization_period is optional and often not needed (Switch
            # will use gen_max_age by default). But we always report it in case
            # the settings use a longer retirement_age (or none) to prevent
            # age-based retirement.
            "cap_recovery_years": "gen_amortization_period",
            "Min_Power": "gen_min_load_fraction",
            "Ramp_Up_Percentage": "gen_ramp_limit_up",
            "Ramp_Dn_Percentage": "gen_ramp_limit_down",
            "Up_Time": "gen_min_uptime",
            "Down_Time": "gen_min_downtime",
            "Start_Cost_per_MW": "gen_startup_om",
        },
        inplace=True,
    )

    # use zero instead of NaN for some columns (other NaNs will get converted
    # to "." for Switch when saving later)
    gen_info["gen_variable_om"] = gen_info["gen_variable_om"].fillna(0)
    gen_info["gen_connect_cost_per_mw"] = gen_info["gen_connect_cost_per_mw"].fillna(0)

    cols = [
        "GENERATION_PROJECT",
        "gen_tech",
        "gen_energy_source",
        "gen_load_zone",
        "gen_max_age",
        "gen_amortization_period",
        "gen_can_retire_early",
        "gen_is_variable",
        "gen_is_baseload",
        "gen_full_load_heat_rate",
        "gen_variable_om",
        "gen_connect_cost_per_mw",
        "gen_dbid",
        "gen_scheduled_outage_rate",
        "gen_forced_outage_rate",
        "gen_capacity_limit_mw",
        "gen_min_load_fraction",
        "gen_ramp_limit_up",
        "gen_ramp_limit_down",
        "gen_min_uptime",
        "gen_min_downtime",
        "gen_startup_om",
        "gen_is_cogen",
        "gen_storage_efficiency",
        "gen_store_to_release_ratio",
        "gen_can_provide_cap_reserves",
        "gen_self_discharge_rate",
        "gen_ccs_capture_efficiency",
        "gen_ccs_energy_load",
        "gen_storage_energy_to_power_ratio",
        "gen_type",
        "ESR_1",  # Below are the columns used for current policy
        "ESR_2",
        "ESR_3",
        "ESR_4",
        "ESR_5",
        "ESR_6",
        "ESR_7",
        "ESR_8",
        "ESR_9",
        "ESR_10",
        "ESR_11",
        "ESR_12",
        "ESR_13",
        "ESR_14",
        "ESR_15",
        "ESR_16",
        "MinCapTag_1",
        "MinCapTag_2",
        "MinCapTag_3",
        "MinCapTag_4",
        "MinCapTag_5",
    ]  # index

    gen_info = gen_info[cols]
    return gen_info


hydro_forced_outage_tech = {
    # "conventional_hydroelectric": 0.05,
    # "hydroelectric_pumped_storage": 0.05,
    # "small_hydroelectric": 0.05,
    "conventional_hydroelectric": 0,
    "hydroelectric_pumped_storage": 0,
    "small_hydroelectric": 0,
}


def match_hydro_forced_outage_tech(x):
    for key in hydro_forced_outage_tech:
        if key in x:
            return hydro_forced_outage_tech[key]


def fuel_market_tables(fuel_prices, aeo_fuel_region_map, scenario):
    """
    Create regional_fuel_markets and zone_to_regional_fuel_market
    SWITCH does not seem to like this overlapping with fuel_cost. So all of this might be incorrect.
    """

    # create initial regional fuel market.  Format: region - fuel
    reg_fuel_mar_1 = fuel_prices.copy()
    reg_fuel_mar_1 = reg_fuel_mar_1.loc[
        reg_fuel_mar_1["scenario"] == scenario
    ]  # use reference for now
    reg_fuel_mar_1 = reg_fuel_mar_1.drop(
        ["year", "price", "full_fuel_name", "scenario"], axis=1
    )
    reg_fuel_mar_1 = reg_fuel_mar_1.rename(columns={"region": "regional_fuel_market"})
    reg_fuel_mar_1 = reg_fuel_mar_1[["regional_fuel_market", "fuel"]]

    fuel_markets = reg_fuel_mar_1["regional_fuel_market"].unique()

    # from region to fuel
    group = reg_fuel_mar_1.groupby("regional_fuel_market")
    fuel_market_dict = {}
    for region in fuel_markets:
        df = group.get_group(region)
        fuel = df["fuel"].to_list()
        fuel = list(set(fuel))
        fuel_market_dict[region] = fuel

    # create zone_regional_fuel_market
    data = list()
    for region in aeo_fuel_region_map.keys():
        for i in range(len(aeo_fuel_region_map[region])):
            ipm = aeo_fuel_region_map[region][i]
            for fuel in fuel_market_dict[region]:
                data.append([ipm, ipm + "-" + fuel])

    zone_regional_fm = pd.DataFrame(data, columns=["load_zone", "regional_fuel_market"])

    # use that to finish regional_fuel_markets
    regional_fuel_markets = zone_regional_fm.copy()
    regional_fuel_markets["fuel_list"] = regional_fuel_markets[
        "regional_fuel_market"
    ].str.split("-")
    regional_fuel_markets["fuel"] = regional_fuel_markets["fuel_list"].apply(
        lambda x: x[-1]
    )
    regional_fuel_markets = regional_fuel_markets[["regional_fuel_market", "fuel"]]

    return regional_fuel_markets, zone_regional_fm


def ts_tp_pg_kmeans(
    representative_point: pd.DataFrame,
    point_weights: List[int],
    days_per_period: int,
    planning_year: int,
    planning_start_year: int,
):
    """Create timeseries and timepoints tables when using kmeans time reduction in PG

    Parameters
    ----------
    representative_point : pd.DataFrame
        The representative periods used. Single column dataframe with col name "slot"
    point_weights : List[int]
        The weight assigned to each period. Equal to the number of periods in the year
        that each period represents.
    days_per_period : int
        How long each period lasts in days
    planning_periods : List[int]
        A list of the planning years
    planning_period_start_years : List[int]
        A list of the start year for each planning period, used to calculate the number
        of years in each period

    Returns
    -------
    pd.DataFrame, pd.DataFrame
        A tuple of the timeseries and timepoints dataframes
    """
    ts_data = {
        "timeseries": [],
        "ts_period": [],
        "ts_duration_of_tp": [],
        "ts_num_tps": [],
        "ts_scale_to_period": [],
    }
    tp_data = {
        "timestamp": [],
        "timeseries": [],
    }
    planning_yrs = planning_year - planning_start_year + 1
    for p, weight in zip(representative_point, point_weights):
        num_hours = days_per_period * 24
        ts = f"{planning_year}_{p}"
        ts_data["timeseries"].append(ts)
        ts_data["ts_period"].append(planning_year)
        ts_data["ts_duration_of_tp"].append(1)
        ts_data["ts_num_tps"].append(num_hours)
        ts_data["ts_scale_to_period"].append(weight * planning_yrs)

        tp_data["timestamp"].extend([f"{ts}_{i}" for i in range(num_hours)])
        tp_data["timeseries"].extend([ts for i in range(num_hours)])

    timeseries = pd.DataFrame(ts_data)
    timepoints = pd.DataFrame(tp_data)
    timepoints["timepoint_id"] = timepoints.index + 1
    timepoints = timepoints[["timepoint_id", "timestamp", "timeseries"]]
    return timeseries, timepoints


def hydro_timepoints_pg_kmeans(timepoints: pd.DataFrame) -> pd.DataFrame:
    """Create the timepoints table when using kmeans time reduction in PG

    This assumes that the hydro timeseries are identical to the model timeseries.

    Parameters
    ----------
    timepoints : pd.DataFrame
        The timepoints table

    Returns
    -------
    pd.DataFrame
        Identical to the incoming timepoints table except "timepoint_id" is renamed to
        "tp_to_hts"
    """

    hydro_timepoints = timepoints.copy()
    hydro_timepoints = hydro_timepoints.rename(columns={"timeseries": "tp_to_hts"})

    return hydro_timepoints[["timepoint_id", "tp_to_hts"]]


def hydro_timeseries_pg_kmeans(
    gen: pd.DataFrame,
    hydro_variability: pd.DataFrame,
    hydro_timepoints: pd.DataFrame,
    outage_rate: float = 0,
) -> pd.DataFrame:
    """Create hydro timeseries table when using kmeans time reduction in PG

    The hydro timeseries table has columns hydro_project, timeseries, outage_rate,
    hydro_min_flow_mw, and hydro_avg_flow_mw. The "timeseries" column links to the
    column "tp_to_hts" in hydro_timepoints.csv. "hydro_min_flow_mw" uses the resource
    minimum capacity (calculated in PG from EIA860). "hydro_avg_flow_mw" is the average
    of flow during each timeseries.

    Parameters
    ----------
    existing_gen : pd.DataFrame
        All existing generators, one row per generator. Columns must include "Resource",
        "Existing_Cap_MW", "Min_Power", and "HYDRO".
    hydro_variability : pd.DataFrame
        Hourly flow/generation capacity factors. Should have column names that correspond
        to the "Resource" column in `existing_gen`. Additional column names will be
        filtered out.
    hydro_timepoints : pd.DataFrame
        All timepoints for hydro, with the column "tp_to_hts"
    outage_rate : float, optional
        The average outage rate for hydro generators, by default 0.05

    Returns
    -------
    pd.DataFrame
        The hydro_timeseries table for Switch
    """

    hydro_df = gen.copy()
    # ? why multiply Min_Power
    # hydro_df["min_cap_mw"] = hydro_df["Existing_Cap_MW"] * hydro_df["Min_Power"]
    hydro_df = hydro_df.loc[hydro_df["HYDRO"] == 1, :]

    hydro_variability = hydro_variability.loc[:, hydro_df["Resource"]]

    # for col in hydro_variability.columns:
    #     hydro_variability[col] *= hydro_df.loc[
    #         hydro_df["Resource"] == col, "Existing_Cap_MW"
    #     ].values[0]
    hydro_variability["timeseries"] = hydro_timepoints["tp_to_hts"].values
    hydro_ts = hydro_variability.melt(id_vars=["timeseries"])
    hydro_ts = hydro_ts.groupby(["timeseries", "Resource"], as_index=False).agg(
        hydro_avg_flow_mw=("value", "mean"), hydro_min_flow_mw=("value", "min")
    )

    # hydro_ts["hydro_min_flow_mw"] = hydro_ts["Resource"].map(
    #     hydro_df.set_index("Resource")["Min_Power"]
    # )
    hydro_ts["hydro_avg_flow_mw"] = hydro_ts["hydro_avg_flow_mw"] * hydro_ts[
        "Resource"
    ].map(hydro_df.set_index("Resource")["Existing_Cap_MW"])
    hydro_ts["hydro_min_flow_mw"] = hydro_ts["hydro_min_flow_mw"] * hydro_ts[
        "Resource"
    ].map(hydro_df.set_index("Resource")["Existing_Cap_MW"])

    hydro_ts["outage_rate"] = outage_rate
    hydro_ts = hydro_ts.rename(columns={"Resource": "hydro_project"})
    cols = [
        "hydro_project",
        "timeseries",
        # "outage_rate",
        "hydro_min_flow_mw",
        "hydro_avg_flow_mw",
    ]
    return hydro_ts[cols]


def variable_cf_pg_kmeans(
    all_gens: pd.DataFrame, all_gen_variability: pd.DataFrame, timepoints: pd.DataFrame
) -> pd.DataFrame:
    """Create the variable capacity factors table when using kmeans time reduction in PG

    Variable generators are identified as those with hourly average capacity factors
    less than 1.

    Parameters
    ----------
    all_gens : pd.DataFrame
        All resources. Must have the columns "Resource" and "gen_is_variable".
    all_gen_variability : pd.DataFrame
        Wide dataframe with hourly capacity factors of all generators.
    timepoints : pd.DataFrame
        Timepoints table with column "timepoint_id"

    Returns
    -------
    pd.DataFrame
        Tidy dataframe with columns "GENERATION_PROJECT", "timepoint", and
        "gen_max_capacity_factor"
    """

    vre_gens = all_gens.loc[all_gens["gen_is_variable"] == 1, "Resource"]
    vre_variability = all_gen_variability[vre_gens]
    vre_variability["timepoint_id"] = timepoints["timepoint_id"].values
    vre_ts = vre_variability.melt(
        id_vars=["timepoint_id"], value_name="gen_max_capacity_factor"
    )
    vre_ts = vre_ts.rename(
        columns={"Resource": "GENERATION_PROJECT", "timepoint_id": "timepoint"}
    )

    return vre_ts.reindex(
        columns=["GENERATION_PROJECT", "timepoint", "gen_max_capacity_factor"]
    )


def load_pg_kmeans(load_curves: pd.DataFrame, timepoints: pd.DataFrame) -> pd.DataFrame:
    """Create the loads table when using kmeans time reduction in PG

    Parameters
    ----------
    load_curves : pd.DataFrame
        Wide dataframe with one column of demand values for each zone
    timepoints : pd.DataFrame
        Timepoints table with column "timepoint_id"

    Returns
    -------
    pd.DataFrame
        Tidy dataframe with columns "LOAD_ZONE" and "TIMEPOINT"
    """
    load_curves = load_curves.astype(int)
    load_curves["TIMEPOINT"] = timepoints["timepoint_id"].values
    load_ts = load_curves.melt(id_vars=["TIMEPOINT"], value_name="zone_demand_mw")
    load_ts = load_ts.rename(columns={"region": "LOAD_ZONE"})
    load_ts["zone_demand_mw"] = load_ts["zone_demand_mw"].astype("object")

    # change the order of the columns
    return load_ts.reindex(columns=["LOAD_ZONE", "TIMEPOINT", "zone_demand_mw"])


def graph_timestamp_map_kmeans(timepoints_df):
    """
    Create the graph_timestamp_map table based on REAM Scenario 178
    Input:
        timeseries_df, timepoints_df: the SWITCH timeseries table
    Output columns:
        * timestamp: dates based on the timeseries table
        * time_row: the period decade year based on the timestamp
        * time_column: format: yyyymmdd. Using 2012 because that is the year data is based on.
    """

    timepoints_df_copy = timepoints_df.copy()
    graph_timeseries_map = pd.DataFrame(columns=["timestamp", "time_row", "timeseries"])
    graph_timeseries_map["timestamp"] = timepoints_df_copy["timestamp"]
    graph_timeseries_map["timeseries"] = timepoints_df_copy["timeseries"]
    graph_timeseries_map["time_row"] = [
        x[0] for x in graph_timeseries_map["timestamp"].str.split("_")
    ]

    # using 2012 for financial year
    graph_timeseries_map["time_column"] = graph_timeseries_map["timeseries"].apply(
        lambda x: str(2012) + x[5:]
    )

    return graph_timeseries_map


def timeseries(
    load_curves,
    planning_year,
    planning_start_year,
    settings,
):  # 20.2778, 283.8889
    """
    Create the timeseries table based on REAM Scenario 178
    Input:
        1) load_curves: created using PowerGenome make(final_load_curves(pg_engine, scenario_settings[][]))
        2) max_weight: the weight to apply to the days with highest values
        3) avg_weight: the weight to apply to the days with average value
        3) ts_duration_of_tp: how many hours should the timpoint last
        4) ts_num_tps: number of timpoints in the selected day
    Output columns:
        - TIMESERIES: format: yyyy_yyyy-mm-dd
        - ts_period: the period decade
        - ts_duration_of_tp: based on input value
        - ts_num_tps: based on input value. Should multiply to 24 with ts_duration_of_tp
        - ts_scale_to_period: use the max&avg_weights for the average and max days in a month
    """
    if settings.get("sample_dates_fn") and settings.get("input_folder"):
        sample_dates = pd.read_csv(
            settings.get("input_folder") / settings["sample_dates_fn"]
        )
    else:
        sample_year = planning_year
        sample_year_start = str(sample_year) + "0101"
        sample_year_end = str(sample_year) + "1231"
        sample_dates = [
            d.strftime("%Y%m%d")
            for d in pd.date_range(sample_year_start, sample_year_end)
        ]

    leap_yr = str(sample_year) + "0229"
    if leap_yr in sample_dates:
        sample_dates.remove(leap_yr)  ### why remove Feb 29th? --RR

    hr_load_sum = pd.DataFrame(load_curves.sum(axis=1), columns=["sum_across_regions"])
    load_hrs = len(load_curves.index)  # number of hours PG outputs data for in a year
    baseyear_hours = len(sample_dates) * 24
    hr_interval = round(load_hrs / baseyear_hours)
    # hr_int_list = list(range(1, int(24 / hr_interval) + 1))
    hr_interval_load_sum = hr_load_sum.groupby(hr_load_sum.index // hr_interval).sum()
    # create initial date list for 2020
    timestamp = list()
    for d in range(len(sample_dates)):
        for i in range(1, 25):
            date_hr = sample_dates[d]
            timestamp.append(date_hr)

    timeseries = [x[:4] + "_" + x[:4] + "-" + x[4:6] + "-" + x[6:8] for x in timestamp]
    ts_period = [x[:4] for x in timestamp]
    timepoint_id = list(range(len(timestamp)))

    column_list = ["timeseries", "ts_period"]
    data = np.array([timeseries, ts_period]).T
    initial_df = pd.DataFrame(
        data, columns=column_list, index=hr_interval_load_sum.index
    )
    initial_df = initial_df.join(hr_interval_load_sum)

    if settings.get("chunk_days"):
        chunk_days = settings.get("chunk_days")
        # split dataframe into chunks of representative_days
        chunk_hr = chunk_days * 24
        n_chunks = len(sample_dates) // chunk_days
        num_days = chunk_days * n_chunks
        chunk_df = []
        for i in range(n_chunks):
            ck_df = (
                (initial_df.iloc[i * chunk_hr : (i + 1) * chunk_hr, :])
                .groupby("timeseries")
                .sum()
            )
            chunk_df.append(ck_df)
    else:
        chunk_days = 8760 / (12 * 24)
        month_hrs = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
        year_cumul = [
            744,
            1416,
            2160,
            2880,
            3624,
            4344,
            5088,
            5832,
            6552,
            7296,
            8016,
            8760,
        ]  # cumulative hours by month
        num_days = sum(month_hrs) / 24
        # split dataframe into months (grouped by day)
        chunk_df = []
        chunk_df.append(
            (initial_df.iloc[0 : year_cumul[0], :]).groupby("timeseries").sum()
        )
        for i in range(len(year_cumul) - 1):
            M_df = (
                (initial_df.iloc[year_cumul[i] : year_cumul[i + 1], :])
                .groupby("timeseries")
                .sum()
            )
            chunk_df.append(M_df)

    # find mean and max for each month, add date to a dataframe
    timeseries_df = pd.DataFrame(
        columns=["sum_across_regions", "timeseries", "close_to_mean"]
    )
    for df in chunk_df:
        df["timeseries"] = df.index
        mean = df["sum_across_regions"].mean()
        df["close_to_mean"] = abs(df["sum_across_regions"] - mean)
        df_mean = df.loc[df["close_to_mean"] == df["close_to_mean"].min()]
        df_max = df.loc[df["sum_across_regions"] == df["sum_across_regions"].max()]
        timeseries_df = timeseries_df.append(df_max)
        timeseries_df = timeseries_df.append(df_mean)
        timeseries_df["timeseries"] = timeseries_df.index

    # add in other columns
    timeseries_df["ts_period"] = str(sample_year)
    ts_duration_of_tp = settings.get("ts_duration_of_tp", 1)
    ts_num_tps = settings.get("ts_num_tps", 24 / ts_duration_of_tp)
    timeseries_df["ts_duration_of_tp"] = ts_duration_of_tp  # assuming 4 for now
    timeseries_df["ts_num_tps"] = ts_num_tps  # assuming 6 for now
    timeseries_df = timeseries_df.reset_index(drop=True)
    timeseries_df = timeseries_df.drop(["sum_across_regions"], axis=1)

    timeseries_df["ts_scale_to_period"] = None

    planning_yrs = planning_year - planning_start_year + 1
    max_days = settings.get("max_days")
    sample_to_year_ratio = 8760 / (num_days * 24)
    max_weight = round(planning_yrs * max_days * sample_to_year_ratio, 4)
    avg_weight = round(planning_yrs * (chunk_days - max_days) * sample_to_year_ratio, 4)

    for i in range(len(timeseries_df)):
        if i % 2 == 0:
            timeseries_df.loc[i, "ts_scale_to_period"] = max_weight
    timeseries_df["ts_scale_to_period"].replace(
        to_replace=[None], value=avg_weight, inplace=True
    )

    timeseries_df = timeseries_df[
        [
            "timeseries",
            "ts_period",
            "ts_duration_of_tp",
            "ts_num_tps",
            "ts_scale_to_period",
        ]
    ]

    timeseries_dates = timeseries_df["timeseries"].to_list()
    timestamp_interval = list()
    for i in range(ts_num_tps):
        s_interval = ts_duration_of_tp * i
        stamp_interval = str(f"{s_interval:02d}")
        timestamp_interval.append(stamp_interval)

    timepoint_id = list(range(1, len(timeseries_dates) + 1))
    timestamp = [x[:4] + x[10:12] + x[13:] for x in timeseries_dates]

    column_list = ["timepoint_id", "timestamp", "timeseries"]
    timepoints_df = pd.DataFrame(columns=column_list)
    for i in timestamp_interval:
        timestamp_interval = [x + i for x in timestamp]
        df_data = np.array([timepoint_id, timestamp_interval, timeseries_dates]).T
        df = pd.DataFrame(df_data, columns=column_list)
        timepoints_df = timepoints_df.append(df)

    timepoints_df["timepoint_id"] = range(
        1, len(timepoints_df["timeseries"].to_list()) + 1
    )

    return timeseries_df, timepoints_df, timestamp_interval


def timeseries_full(
    load_curves,
    planning_year,
    planning_start_year,
    settings,
):  # 20.2778, 283.8889
    """Create timeseries and timepoints tables when using yearly data with 8760 hours
    Apply this function reduce_time_domain: False & full_time_domain: True in settings
    Parameters
    ----------
    planning_periods : List[int]
        A list of the planning years
    planning_period_start_years : List[int]
        A list of the start year for each planning period, used to calculate the number
        of years in each period

    Returns
    -------
    pd.DataFrame, pd.DataFrame
        A tuple of the timeseries and timepoints dataframes
    """

    if settings.get("sample_dates_fn") and settings.get("input_folder"):
        sample_dates = pd.read_csv(
            settings.get("input_folder") / settings["sample_dates_fn"]
        )
    else:
        sample_year = planning_year
        sample_year_start = str(sample_year) + "0101"
        sample_year_end = str(sample_year) + "1231"
        sample_dates = [
            d.strftime("%Y%m%d")
            for d in pd.date_range(sample_year_start, sample_year_end)
        ]

    leap_yr = str(sample_year) + "0229"
    if leap_yr in sample_dates:
        sample_dates.remove(leap_yr)  ### why remove Feb 29th? --RR
    num_days = len(sample_dates)
    num_hours = 24 * num_days
    sample_to_year_ratio = 8760 / (num_days * 24)
    planning_yrs = planning_year - planning_start_year + 1

    timeseries_df = pd.DataFrame()
    ts = f"{sample_year}_{sample_year}-full"
    timeseries_df["timeseries"] = [ts]
    timeseries_df["ts_period"] = [f"{sample_year}"]
    timeseries_df["ts_duration_of_tp"] = [1]  # each hour as one timepoint
    timeseries_df["ts_num_tps"] = [num_hours]
    timeseries_df["ts_scale_to_period"] = [planning_yrs * sample_to_year_ratio]

    timestamp_interval = list()
    for i in range(24):
        s_interval = i
        stamp_interval = str(f"{s_interval:02d}")
        timestamp_interval.append(stamp_interval)

    timepoints_df = pd.DataFrame()
    timepoints_df["timeseries"] = [ts for i in range(num_hours)]
    timepoints_df["timepoint_id"] = range(
        1, len(timepoints_df["timeseries"].to_list()) + 1
    )

    timepoints_df["timestamp"] = [
        f"{d}{i}" for d in sample_dates for i in timestamp_interval
    ]

    timepoints_df = timepoints_df[["timepoint_id", "timestamp", "timeseries"]]
    return timeseries_df, timepoints_df, timestamp_interval


def hydro_time_tables(existing_gen, hydro_variability, timepoints_df, planning_year):
    """
    Create the hydro_timepoints table based on REAM Scenario 178
    Inputs:
        1) timepoints_df: the SWITCH timepoints table
    Output Columns
        * timepoint_id: from the timepoints table
        * tp_to_hts: format: yyyy_M#. Based on the timestamp date from the timepoints table
    """

    hydro_timepoints = timepoints_df.copy()
    hydro_timepoints = hydro_timepoints.rename(columns={"timeseries": "tp_to_hts"})
    convert_to_hts = {
        "01": "_M1",
        "02": "_M2",
        "03": "_M3",
        "04": "_M4",
        "05": "_M5",
        "06": "_M6",
        "07": "_M7",
        "08": "_M8",
        "09": "_M9",
        "10": "_M10",
        "11": "_M11",
        "12": "_M12",
    }

    def convert(tstamp):
        month = tstamp[4:6]
        year = tstamp[0:4]
        return year + convert_to_hts[month]

    hydro_timepoints["tp_to_hts"] = hydro_timepoints["timestamp"].apply(convert)
    hydro_timepoints.drop("timestamp", axis=1, inplace=True)

    hydro_list = [
        "Conventional Hydroelectric",
        # "Hydroelectric Pumped Storage",
        "Small Hydroelectric",
    ]

    #### edit by RR
    # filter existing gen to just hydro technologies
    hydro_df = existing_gen.copy()
    # hydro_df["index"] = hydro_df.index
    hydro_df = hydro_df[hydro_df["technology"].isin(hydro_list)]
    hydro_indx = hydro_df["Resource"].to_list()
    hydro_region = hydro_df["region"].to_list()

    # slice the hours to 8760
    hydro_variability = hydro_variability.iloc[:8760]
    hydro_variability = hydro_variability.loc[:, hydro_indx]
    hydro_variability.columns = hydro_indx
    ####

    # get cap size for each hydro tech
    hydro_Cap_Size = hydro_df["Existing_Cap_MW"].to_list()  # cap size for each hydro
    # multiply cap size by hourly
    for i in range(len(hydro_Cap_Size)):
        hydro_variability.iloc[:, i] = hydro_variability.iloc[:, i].apply(
            lambda x: x * hydro_Cap_Size[i]
        )

    hydro_transpose = hydro_variability.transpose()

    month_hrs = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
    year_cumul = [
        744,
        1416,
        2160,
        2880,
        3624,
        4344,
        5088,
        5832,
        6552,
        7296,
        8016,
        8760,
    ]  # cumulative hours by month

    # split dataframe into months
    month_df = []
    month_df.append((hydro_transpose.iloc[:, 0 : year_cumul[0]]))
    for i in range(len(year_cumul) - 1):
        M_df = hydro_transpose.iloc[:, year_cumul[i] : year_cumul[i + 1]]
        month_df.append(M_df)

    month_names = [
        "M1",
        "M2",
        "M3",
        "M4",
        "M5",
        "M6",
        "M7",
        "M8",
        "M9",
        "M10",
        "M11",
        "M12",
    ]
    df_list = list()
    for i in range(len(month_df)):
        df = pd.DataFrame(hydro_transpose.index, columns=["hydro_project"])
        df["timeseries"] = month_names[i]
        df["outage_rate"] = list(
            map(match_hydro_forced_outage_tech, hydro_df["Resource"])
        )
        df["hydro_min_flow_mw"] = month_df[i].min(axis=1).to_list()
        df["hydro_avg_flow_mw"] = month_df[i].mean(axis=1).to_list()
        df_list.append(df)

    hydro_timeseries = pd.concat(df_list, axis=0)
    hydro_timeseries["timeseries"] = (
        str(planning_year) + "_" + hydro_timeseries["timeseries"]
    )

    return hydro_timepoints, hydro_timeseries


def hydro_system_module_tables(
    gen,
    hydro_variability: pd.DataFrame,
    hydro_timepoints: pd.DataFrame,
    flow_per_mw: float = 1.02,
) -> pd.DataFrame:
    """
    Create the tables specific for module hydro_system
    Inputs:
        1) flow_per_mw: 1/[(1000 kg/m3)(9.8 N/kg)(100 m)(1 MWs/1e6 Nm)] = 1.02 m3/s/MW
        2) reservoir_capacity_m3 <- flow_per_mw * Hydro_Energy_to_Power_Ratio * [generator capacity (MW)]
        3) inflow_m3_per_s <- flow_per_mw * [variable inflow (% of nameplate power)] * [generator capacity (MW)]
    Output tables:
        * water_modes.csv:
            # add these rows, representing the reservoir upstream of each dam
            WATER_NODES <- generator name + “_inlet” (or similar)
            wn_is_sink <- 0
            wnode_constant_consumption <- 0
            wnode_constant_inflow <- 0
        * water_nodes.csv:
            # add a second set of rows to this file, representing reservoir downstream of each dam
             WATER_NODES <- generator name + “_outlet” (or similar)
            wn_is_sink <- 1
            wnode_constant_consumption <- 0
            wnode_constant_inflow <- 0
        * water_connections.csv
            WATER_CONNECTIONS <- generator name
            water_node_from <- generator name + “_inlet”
            water_node_to <- generator name + “_outlet”
        *reservoirs.csv
            RESERVOIRS <- generator name + “_inlet”
            res_min_vol <- 0
            res_max_vol <- reservoir_capacity_m3 (see above)
            # arbitrarily assume reservoir must start and end at 50% full
            initial_res_vol <- 0.5 * reservoir_capacity_m3
            final_res_vol <- 0.5 * reservoir_capacity_m3
        *hydro_generation_projects.csv
            HYDRO_GENERATION_PROJECTS <- generator name (should match gen_info.csv)
            # hydro_efficiency is MW output per m3/s of input
            hydro_efficiency <- 1 / flow_per_mw
            hydraulic_location <- generator name (should match water_connections.csv)
        *water_node_tp_flows.csv
            WATER_NODES <- generator name + “_inlet”
            TIMEPOINTS <- timepoint
            wnode_tp_inflow <- inflow_m3_per_s
            wnode_tp_consumption <- 0
    """

    hydro_df = gen.copy()
    hydro_df = hydro_df.loc[hydro_df["HYDRO"] == 1, :]

    hydro_variability = hydro_variability.loc[:, hydro_df["Resource"]]

    # for water_nodes.csv
    water_nodes_in = pd.DataFrame()
    water_nodes_in["WATER_NODES"] = hydro_df["Resource"] + "_inlet"
    water_nodes_in["wn_is_sink"] = 0
    water_nodes_in["wnode_constant_consumption"] = 0
    water_nodes_in["wnode_constant_inflow"] = 0
    water_nodes_out = pd.DataFrame()
    water_nodes_out["WATER_NODES"] = hydro_df["Resource"] + "_outlet"
    water_nodes_out["wn_is_sink"] = 1
    water_nodes_out["wnode_constant_consumption"] = 0
    water_nodes_out["wnode_constant_inflow"] = 0
    water_nodes = pd.concat([water_nodes_in, water_nodes_out])
    # for water_connections.csv
    water_connections = pd.DataFrame()
    water_connections["WATER_CONNECTIONS"] = hydro_df["Resource"]
    water_connections["water_node_from"] = hydro_df["Resource"] + "_inlet"
    water_connections["water_node_to"] = hydro_df["Resource"] + "_outlet"
    water_connections["wc_capacity"] = hydro_df["Existing_Cap_MW"] * flow_per_mw  # m3/s

    # for reservoirs.csv
    # note: reservoir volume and level are given in million m3
    reservoirs = pd.DataFrame()
    reservoirs["RESERVOIRS"] = hydro_df["Resource"] + "_inlet"
    reservoirs["res_min_vol"] = 0
    # (MW)(hour)((m3/s)/MW)(3600 sec/hour)(1 million m3/1e6 m3) = million m3
    reservoirs["res_max_vol"] = (
        hydro_df["Existing_Cap_MW"]  # MW
        * hydro_df["Hydro_Energy_to_Power_Ratio"]  # h
        * flow_per_mw  # m3/s / MW
        * 3600  # m3/sec -> m3/hour
        * 1e-6  # m3 -> million m3
    )

    # The initial_res_vol and final_res_vol calculations commented out below
    # would work with the standard hydro_system module. But that doesn't work
    # well with multi- timeseries periods, so we use a new version that does,
    # and we let it optimize the starting level and loop from end to start of
    # timeseries. If these are wanted for the new module (not used by other
    # models in this study), they should be stored in reservoir_ts_data.csv,
    # with reservoir and timeseries as the indexes.

    # reservoirs["initial_res_vol"] = 0.5 * reservoirs["res_max_vol"]
    # reservoirs["final_res_vol"] = 0.5 * reservoirs["res_max_vol"]

    # for hydro_generation_projects.csv
    hydro_pj = pd.DataFrame()
    hydro_pj["HYDRO_GENERATION_PROJECTS"] = hydro_df["Resource"]
    hydro_pj["hydro_efficiency"] = 1 / flow_per_mw
    hydro_pj["hydraulic_location"] = hydro_df["Resource"]
    # for water_node_tp_flows.csv
    hydro_variability["TIMEPOINTS"] = hydro_timepoints["timepoint_id"].values
    water_node_tp = hydro_variability.melt(id_vars=["TIMEPOINTS"])
    water_node_tp["wnode_tp_inflow"] = (
        flow_per_mw
        * water_node_tp["value"]
        * water_node_tp["Resource"].map(
            hydro_df.set_index("Resource")["Existing_Cap_MW"]
        )
    )
    water_node_tp["WATER_NODES"] = water_node_tp["Resource"] + "_inlet"
    water_node_tp["wnode_tp_consumption"] = 0
    cols = [
        "WATER_NODES",
        "TIMEPOINTS",
        "wnode_tp_inflow",
        "wnode_tp_consumption",
    ]
    water_node_tp_flows = water_node_tp[cols]

    return water_nodes, water_connections, reservoirs, hydro_pj, water_node_tp_flows


def graph_timestamp_map_table(timeseries_df, timestamp_interval):
    """
    Create the graph_timestamp_map table based on REAM Scenario 178
    Input:
        1) timeseries_df: the SWITCH timeseries table
        2) timestamp_interval:based on ts_duration_of_tp and ts_num_tps from the timeseries table.
                Should be between 0 and 24.
    Output columns:
        * timestamp: dates based on the timeseries table
        * time_row: the period decade year based on the timestamp
        * time_column: format: yyyymmdd. Using 2012 because that is the year data is based on.
    """

    timeseries_df_copy = timeseries_df.copy()
    timeseries_df_copy = timeseries_df_copy[["timeseries", "ts_period"]]
    # reformat timeseries for timestamp
    timeseries_df_copy["timestamp"] = timeseries_df_copy["timeseries"].apply(
        lambda x: x[5:9] + x[10:12] + x[13:]
    )

    # add in intervals to the timestamp
    graph_timeseries_map = pd.DataFrame(
        columns=["timeseries", "ts_period", "timestamp"]
    )
    for x in timestamp_interval:
        df = timeseries_df_copy[["timeseries", "ts_period"]]
        df["timestamp"] = timeseries_df_copy["timestamp"] + x
        graph_timeseries_map = graph_timeseries_map.append(df)

    # using 2012 for financial year
    graph_timeseries_map["time_column"] = graph_timeseries_map["timeseries"].apply(
        lambda x: str(2012) + x[10:12] + x[13:15]
    )

    graph_timeseries_map = graph_timeseries_map.drop(["timeseries"], axis=1)
    graph_timeseries_map = graph_timeseries_map.rename(
        columns={"ts_period": "time_row"}
    )
    graph_timeseries_map = graph_timeseries_map[
        ["timestamp", "time_row", "time_column"]
    ]

    return graph_timeseries_map


def loads_table(load_curves, timepoints_timestamp, timepoints_dict, planning_year):
    """
    Inputs:
        load_curves: from powergenome
        timepoints_timestamp: the timestamps in timepoints
        timepoints_dict: to go from timestamp to timepoint
        period_list: the decade list
    Output columns
        * load_zone: the IPM regions
        * timepoint: from timepoints
        * zone_demand_mw: based on load_curves
    Output df
        loads: the 'final' table
        loads_with_hour_year: include hour year so it is easier to do variable_capacity_factors
    """
    loads_initial = pd.DataFrame(columns=["year_hour", "LOAD_ZONE", "zone_demand_mw"])
    hours = load_curves.index.to_list()
    cols = load_curves.columns.to_list()

    # add load zone for each hour of the year, adding in the load_curve values for each hour
    for c in cols:
        df = pd.DataFrame()
        df["year_hour"] = hours
        df["LOAD_ZONE"] = c
        df["zone_demand_mw"] = load_curves[c].to_list()
        loads_initial = loads_initial.append(df)

    # convert timepoints to date of the year
    start = pd.to_datetime(f"2021-01-01 0:00")  # use 2021 due to 2020 being a leap year
    loads_initial["date"] = loads_initial["year_hour"].apply(
        lambda x: start + pd.to_timedelta(x, unit="H")
    )
    # reformat to timestamp format
    loads_initial["reformat"] = loads_initial["date"].apply(
        lambda x: x.strftime("%Y%m%d%H")
    )
    loads_initial["reformat"] = loads_initial["reformat"].astype(str)
    # create timestamp
    date_list = loads_initial["reformat"].to_list()

    updated_dates = [f"{planning_year}" + x[4:] for x in date_list]
    loads_initial["timestamp"] = updated_dates

    # filter to correct timestamps for timepoints
    loads = loads_initial.loc[loads_initial["timestamp"].isin(timepoints_timestamp)]
    loads["TIMEPOINT"] = loads["timestamp"].apply(lambda x: timepoints_dict[x])
    loads_with_year_hour = loads[["timestamp", "TIMEPOINT", "year_hour"]]
    loads = loads[["LOAD_ZONE", "TIMEPOINT", "zone_demand_mw"]]

    return loads, loads_with_year_hour


def variable_capacity_factors_table(
    all_gen_variability, year_hour, timepoints_dict, all_gen, planning_year
):
    """
    Inputs
        all_gen_variability: from powergenome
        year_hour: the hour of the year that has a timepoint (based on loads)
        timepoints_dict: convert timestamp to timepoint
        all_gen: from powergenome
    Output:
        GENERATION_PROJECT: based on all_gen index
            the plants here should only be the ones with gen_is_variable =True
        timepoint: based on timepoints
        gen_max_capacity_factor: based on all_gen_variability
    """

    v_capacity_factors = all_gen_variability.copy().transpose()
    v_capacity_factors["GENERATION_PROJECT"] = all_gen["Resource"].values
    v_c_f = v_capacity_factors.melt(
        id_vars="GENERATION_PROJECT",
        var_name="year_hour",
        value_name="gen_max_capacity_factor",
    )
    # reduce variability to just the hours of the year that have a timepoint
    # needs to start with 1 to allign with year_hour
    year_hour = [x - 1 for x in year_hour]

    v_c_f = v_c_f.loc[v_c_f["year_hour"].isin(year_hour), :]
    mod_vcf = v_c_f.copy()
    # get the dates from hour of the year
    start = pd.to_datetime("2021-01-01 0:00")  # 2020 is a leap year
    mod_vcf["date"] = mod_vcf["year_hour"].apply(
        lambda x: start + pd.to_timedelta(x, unit="H")
    )
    mod_vcf["reformat"] = mod_vcf["date"].apply(lambda x: x.strftime("%Y%m%d%H"))
    mod_vcf["reformat"] = mod_vcf["reformat"].astype(str)
    date_list = mod_vcf["reformat"].to_list()
    # change 2021 to correct period year/decade
    # to get the timestamp
    updated_dates = [str(planning_year) + x[4:] for x in date_list]
    #     mod_vcf_copy = mod_vcf.copy()
    mod_vcf["timestamp"] = updated_dates
    mod_vcf["timepoint"] = mod_vcf["timestamp"].apply(lambda x: timepoints_dict[x])
    mod_vcf.drop(["year_hour", "date", "reformat", "timestamp"], axis=1, inplace=True)
    # only get all_gen plants that are wind or solar
    all_gen = all_gen.loc[all_gen["gen_is_variable"] == 1, :]

    var_cap_fac = mod_vcf.loc[
        mod_vcf["GENERATION_PROJECT"].isin(all_gen["Resource"]), :
    ]

    # filter to final columns
    var_cap_fac = var_cap_fac[
        ["GENERATION_PROJECT", "timepoint", "gen_max_capacity_factor"]
    ]
    searchfor = ["pv", "solar", "wind", "distribute"]
    var_cap_fac = var_cap_fac[
        var_cap_fac["GENERATION_PROJECT"].str.contains("|".join(searchfor), case=False)
    ]

    return var_cap_fac


def load_zones_table(IPM_regions, zone_ccs_distance_km):
    load_zones = pd.DataFrame(
        columns=["LOAD_ZONE", "zone_ccs_distance_km", "zone_dbid"]
    )
    load_zones["LOAD_ZONE"] = IPM_regions
    load_zones["zone_ccs_distance_km"] = 0  # set to default 0
    load_zones["zone_dbid"] = range(1, len(IPM_regions) + 1)
    return load_zones


def create_transm_line_col(lz1, lz2, zone_dict):
    t_line = zone_dict[lz1] + "-" + zone_dict[lz2]
    return t_line


def transmission_lines_table(
    line_loss, add_cap, tx_capex_mw_mile_dict, zone_dict, settings
):
    """
    Create transmission_lines table based on REAM Scenario 178
    Output Columns:
        TRANSMISSION_LINE: zone_dbid-zone_dbid for trans_lz1 and lz2
        trans_lz1: split PG transmission_path_name
        trans_lz2: split PG transmission_path_name
        trans_length_km: PG distance_mile * km_per_mile
        trans_efficiency: PG line_loss_percentage (1 - line_loss_percentage)
        existing_trans_cap: PG line_max_cap_flow. Take absolute value and take max of the two values
        trans_dbid: id number
        trans_derating_factor: assuming PG DerateCapRes_1 (0.95)
        trans_terrain_multiplier:
            trans_capital_cost_per_mw_km * trans_terrain_multiplier = the average of the two regions
            ('transmission_investment_cost')['tx']['capex_mw_mile'])
        trans_new_build_allowed: how to determine what is allowed. Assume all 1s to start
    """
    transmission_df = line_loss[
        [
            "Network_Lines",
            "transmission_path_name",
            "distance_mile",
            "Line_Loss_Percentage",
        ]
    ]

    # split to get trans_lz1 and trans_lz2
    split_path_name = transmission_df["transmission_path_name"].str.split(
        "_to_", expand=True
    )
    transmission_df = transmission_df.join(split_path_name)

    # convert miles to km for trans_length_km
    transmission_df["trans_length_km"] = transmission_df["distance_mile"] * km_per_mile

    # for trans_efficiency do 1 - line_loss_percentage
    transmission_df["trans_efficiency"] = transmission_df["Line_Loss_Percentage"].apply(
        lambda x: 1 - x
    )

    transmission_df = transmission_df.join(
        add_cap[["Line_Max_Flow_MW", "Line_Min_Flow_MW", "DerateCapRes_1"]]
    )

    # want the max value so take abosolute of line_min_flow_mw (has negatives) and then take max
    transmission_df["line_min_abs"] = transmission_df["Line_Min_Flow_MW"].abs()
    transmission_df["existing_trans_cap"] = transmission_df[
        ["Line_Max_Flow_MW", "line_min_abs"]
    ].max(axis=1)

    # get rid of columns
    transm_final = transmission_df.drop(
        [
            "transmission_path_name",
            "distance_mile",
            "Line_Loss_Percentage",
            "Line_Max_Flow_MW",
            "Line_Min_Flow_MW",
            "line_min_abs",
        ],
        axis=1,
    )

    transm_final = transm_final.rename(
        columns={
            "Network_Lines": "trans_dbid",
            0: "trans_lz1",
            1: "trans_lz2",
            "DerateCapRes_1": "trans_derating_factor",
        }
    )

    transm_final["tz1_dbid"] = transm_final["trans_lz1"].apply(lambda x: zone_dict[x])
    transm_final["tz2_dbid"] = transm_final["trans_lz2"].apply(lambda x: zone_dict[x])
    transm_final["TRANSMISSION_LINE"] = (
        transm_final["tz1_dbid"].astype(str)
        + "-"
        + transm_final["tz2_dbid"].astype(str)
    )

    # use average of transmission cost from the two zones and convert to cost per MW-km
    transm_final["cap_cost_per_mw_km"] = (
        0.5
        * (
            transm_final["trans_lz1"].map(tx_capex_mw_mile_dict)
            + transm_final["trans_lz2"].map(tx_capex_mw_mile_dict)
        )
        / km_per_mile  # ($/mi) / (km/mi) = $/km
    )
    # benchmark value for transmission capital cost; each line will end up with
    # trans_capital_cost_per_mw_km * trans_terrain_multiplier = cap_cost_per_mw_km
    trans_capital_cost_per_mw_km = transm_final["cap_cost_per_mw_km"].min()
    if trans_capital_cost_per_mw_km == 0:
        trans_capital_cost_per_mw_km = 1  # avoid division by zero below

    transm_final["trans_terrain_multiplier"] = (
        transm_final["cap_cost_per_mw_km"] / trans_capital_cost_per_mw_km
    )

    # set as 1 for now
    transm_final["trans_new_build_allowed"] = 1
    # sort columns
    transm_final = transm_final[
        [
            "TRANSMISSION_LINE",
            "trans_lz1",
            "trans_lz2",
            "trans_length_km",
            "trans_efficiency",
            "existing_trans_cap",
            "trans_dbid",
            "trans_derating_factor",
            "trans_terrain_multiplier",
            "trans_new_build_allowed",
        ]
    ]
    return transm_final, trans_capital_cost_per_mw_km


def tx_cost_transform(tx_cost_df):
    tx_cost_df["cost_per_mw-km"] = (
        tx_cost_df["total_interconnect_cost_mw"] / tx_cost_df["total_mw-km_per_mw"]
    )
    trans_capital_cost_per_mw_km = tx_cost_df["cost_per_mw-km"].min()
    if trans_capital_cost_per_mw_km == 0:
        trans_capital_cost_per_mw_km = 1  # avoid division by zero later
    tx_cost_df["trans_terrain_multiplier"] = (
        tx_cost_df["cost_per_mw-km"] / trans_capital_cost_per_mw_km
    )
    tx_cost_df["trans_efficiency"] = 1 - tx_cost_df["total_line_loss_frac"]
    tx_cost_df["trans_length_km"] = tx_cost_df["total_mw-km_per_mw"]
    tx_cost_df["trans_new_build_allowed"] = 1
    tx_cost_df["existing_trans_cap"] = tx_cost_df["Line_Max_Flow_MW"]
    return tx_cost_df, trans_capital_cost_per_mw_km


def balancing_areas(
    pudl_engine,
    IPM_regions,
    all_gen,
    quickstart_res_load_frac,
    quickstart_res_wind_frac,
    quickstart_res_solar_frac,
    spinning_res_load_frac,
    spinning_res_wind_frac,
    spinning_res_solar_frac,
):
    """
    Function to create balancing_areas and zone_balancing_area tables
    Input:
        1) pudl_engine from init_pudl_connection
        2) IPM regions from settings.get('model_regions')
        3) all_gen pandas dataframe from gc.create_all_generators()
        4) quickstart_res_load_frac, quickstart_res_wind_frac, quickstart_res_solar_frac,
            spinning_res_load_frac, spinning_res_wind_frac, and spinning_res_solar_frac:
            --> set these equal to values based on REAM
    Output:
        BALANCING_AREAS
            * BALANCING_AREAS: based on balancing authority from pudl and connecting that to all_gen using plant_id_eia
            * other columns based on REAM Scenario 178
        ZONE_BALANCING_AREAS
            * Load_zone: IPM region
            * balancing_area
    """
    import pudl

    if pudl.__version__ <= "0.6.0":
        # get table from PUDL that has  balancing_authority_code_eia
        plants_entity_eia = pd.read_sql_table("plants_entity_eia", pudl_engine)
    else:
        plants_eia = pd.read_sql_table(
            "plants_eia860",
            pudl_engine,
            parse_dates=["report_date"],
            columns=["report_date", "plant_id_eia", "balancing_authority_code_eia"],
        )
        plants_entity_eia = plants_eia.sort_values(
            "report_date", ascending=False
        ).drop_duplicates(
            # take the latest reported row for this plant; this gets the latest
            # balancing_authority_code_eia if a plant has moved and also avoids
            # problems with unreported balancing_authority_code_eia's in some
            # earlier years.
            subset=["plant_id_eia"],
            keep="first",
        )
    # dataframe with only balancing_authority_code_eia and plant_id_eia
    plants_entity_eia = plants_entity_eia[
        ["balancing_authority_code_eia", "plant_id_eia"]
    ]
    # create a dictionary that has plant_id_eia as key and the balancing authority as value
    plants_entity_eia_dict = plants_entity_eia.set_index("plant_id_eia").T.to_dict(
        "list"
    )

    plant_region_df = all_gen.copy()
    plant_region_df = plant_region_df[["plant_id_eia", "region"]]

    # get rid of NAs
    plant_region_df = plant_region_df[plant_region_df["plant_id_eia"].notna()]

    """
    BALANCING_AREAS:
    take the plant_id_eia column from all_gen input, and return the balancing authority using
        the PUDL plants_entity_eia dictionary

    """

    # define function to get balancing_authority_code_eia from plant_id_eia
    def id_eia_to_bal_auth(plant_id_eia, plants_entity_eia_dict):
        if plant_id_eia in plants_entity_eia_dict.keys():
            return plants_entity_eia_dict[plant_id_eia][
                0
            ]  # get balancing_area from [balancing_area]
        else:
            return "-"

    # return balancing_authority_code_eia from PUDL table based on plant_id_eia
    plant_region_df["balancing_authority_code_eia"] = plant_region_df[
        "plant_id_eia"
    ].apply(lambda x: id_eia_to_bal_auth(x, plants_entity_eia_dict))

    # create output table
    balancing_areas = plant_region_df["balancing_authority_code_eia"].unique()
    BALANCING_AREAS = pd.DataFrame(balancing_areas, columns=["BALANCING_AREAS"])
    BALANCING_AREAS["quickstart_res_load_frac"] = quickstart_res_load_frac
    BALANCING_AREAS["quickstart_res_wind_frac"] = quickstart_res_wind_frac
    BALANCING_AREAS["quickstart_res_solar_frac"] = quickstart_res_solar_frac
    BALANCING_AREAS["spinning_res_load_frac"] = spinning_res_load_frac
    BALANCING_AREAS["spinning_res_wind_frac"] = spinning_res_wind_frac
    BALANCING_AREAS["spinning_res_solar_frac"] = spinning_res_solar_frac

    """
    ZONE_BALANCING_AREAS table:
        for each of the IPM regions, find the most common balancing_authority to create table
    """

    zone_b_a_list = list()
    for ipm in IPM_regions:
        region_df = plant_region_df.loc[plant_region_df["region"] == ipm]
        # take the most common balancing authority (assumption)
        bal_aut = mode(region_df["balancing_authority_code_eia"].to_list())
        zone_b_a_list.append([ipm, bal_aut])
    zone_b_a_list.append(["_ALL_ZONES", "."])  # Last line in the REAM inputs
    ZONE_BALANCING_AREAS = pd.DataFrame(
        zone_b_a_list, columns=["LOAD_ZONE", "balancing_area"]
    )

    return BALANCING_AREAS, ZONE_BALANCING_AREAS
