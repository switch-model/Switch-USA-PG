import os
import sys
import pandas as pd
import numpy as np
import scipy
import math
from datetime import datetime as dt
import ast
import itertools
from statistics import mode
from typing import List, Optional
import collections

from powergenome.resource_clusters import ResourceGroup
from pathlib import Path
import sqlalchemy as sa
import typer
from typing_extensions import Annotated

import pandas as pd
from powergenome.fuels import fuel_cost_table
from powergenome.generators import GeneratorClusters, create_plant_gen_id
from powergenome.util import (
    build_scenario_settings,
    init_pudl_connection,
    load_settings,
    check_settings,
    snake_case_col,
)

from powergenome.time_reduction import kmeans_time_clustering
from powergenome.eia_opendata import fetch_fuel_prices
from powergenome.eia_opendata import add_user_fuel_prices
import geopandas as gpd
from powergenome.generators import *
from powergenome.external_data import (
    make_demand_response_profiles,
    make_generator_variability,
    load_demand_segments,
)
from powergenome.GenX import (
    add_misc_gen_values,
    hydro_energy_to_power,
    add_co2_costs_to_o_m,
    create_policy_req,
    set_must_run_generation,
    min_cap_req,
)
from powergenome.co2_pipeline_cost import merge_co2_pipeline_costs


from conversion_functions import (
    switch_fuel_cost_table,
    switch_fuels,
    gen_info_table,
    hydro_time_tables,
    load_zones_table,
    fuel_market_tables,
    timeseries,
    timeseries_full,
    graph_timestamp_map_table,
    graph_timestamp_map_kmeans,
    loads_table,
    tx_cost_transform,
    variable_capacity_factors_table,
    transmission_lines_table,
    balancing_areas,
    ts_tp_pg_kmeans,
    hydro_timepoints_pg_kmeans,
    hydro_timeseries_pg_kmeans,
    hydro_system_module_tables,
    variable_cf_pg_kmeans,
    load_pg_kmeans,
    first_key,
    first_value,
    final_key,
    final_value,
    km_per_mile,
)

from powergenome.load_profiles import (
    make_load_curves,
    add_load_growth,
    make_final_load_curves,
    make_distributed_gen_profiles,
)

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


def fuel_files(
    fuel_prices: pd.DataFrame,
    planning_years: List[int],
    regions: List[str],
    fuel_region_map: dict[str, List[str]],
    fuel_emission_factors: dict[str, float],
    out_folder: Path,
):
    fuel_cost = switch_fuel_cost_table(
        fuel_region_map,
        fuel_prices,
        regions,
        scenario=["reference", "user"],
        year_list=planning_years,
    )

    fuels_table = switch_fuels(fuel_prices, fuel_emission_factors)
    fuels_table.loc[len(fuels_table.index)] = [
        "Fuel",
        0,
        0,
    ]  # adding in a dummy fuel for regional_fuel_market

    ### edit by RR
    IPM_regions = regions
    load_zones = load_zones_table(IPM_regions, zone_ccs_distance_km=0)
    # add in the dummy loadzone
    load_zones.loc[len(load_zones.index)] = [
        "loadzone",
        0,
        load_zones["zone_dbid"].max() + 1,
    ]
    load_zones.to_csv(out_folder / "load_zones.csv", index=False)

    regional_fuel_markets = pd.DataFrame(
        {"regional_fuel_market": "loadzone-Fuel", "fuel": "Fuel"}, index=[0]
    )
    regional_fuel_markets

    ### edited by RR. CHANGE COLUMN NAME from fuel to rfm.
    zone_regional_fm = pd.DataFrame(
        {"load_zone": "loadzone", "rfm": "loadzone-Fuel"}, index=[0]
    )
    zone_regional_fm
    # creating dummy values based on one load zone in REAM's input file
    # note:regional_fuel_market should align with the regional_fuel_market table.
    # TODO --RR
    fuel_supply_curves20 = pd.DataFrame(
        {
            "period": [2020, 2020, 2020, 2020, 2020, 2020],
            "tier": [1, 2, 3, 4, 5, 6],
            "unit_cost": [1.9, 4.0, 487.5, 563.7, 637.8, 816.7],
            "max_avail_at_cost": [651929, 3845638, 3871799, 3882177, 3889953, 3920836],
        }
    )
    fuel_supply_curves20.insert(0, "regional_fuel_market", "loadzone-Fuel")
    fuel_supply_curves30 = fuel_supply_curves20.copy()
    fuel_supply_curves30["period"] = 2030
    fuel_supply_curves40 = fuel_supply_curves20.copy()
    fuel_supply_curves40["period"] = 2040
    fuel_supply_curves50 = fuel_supply_curves20.copy()
    fuel_supply_curves50["period"] = 2050
    fuel_supply_curves = pd.concat(
        [
            fuel_supply_curves20,
            fuel_supply_curves30,
            fuel_supply_curves40,
            fuel_supply_curves50,
        ]
    )
    fuel_supply_curves

    regional_fuel_markets.to_csv(out_folder / "regional_fuel_markets.csv", index=False)
    zone_regional_fm.to_csv(
        out_folder / "zone_to_regional_fuel_market.csv", index=False
    )
    fuel_supply_curves.to_csv(out_folder / "fuel_supply_curves.csv", index=False)

    ###

    fuel_cost.to_csv(out_folder / "fuel_cost.csv", index=False)
    fuels_table.to_csv(out_folder / "fuels.csv", index=False)


def generator_and_load_files(
    gc: GeneratorClusters,
    all_fuel_prices,
    pudl_engine: sa.engine,
    scen_settings_dict: dict[dict],
    out_folder: Path,
    pg_engine: sa.engine,
    hydro_variability_new: pd.DataFrame,
):
    """
    Steps:
    use PowerGenome functions to define all_gen (unchanged across years), with
        parameters for all generators
    rename columns in all_gen to match Switch conventions
    split all_gen into existing_gen_units (exploded by vintage) and new_gens

    """

    # TODO: maybe move all the arguments into an `options` dict that can be
    # passed to all the functions, so we don't have to worry about which
    # functions need which arguments

    out_folder.mkdir(parents=True, exist_ok=True)
    first_year_settings = first_value(scen_settings_dict)

    # get tables of generators, organized by model_year or build_year
    # (model_year shows generators active in a particular model year, used for
    # gathering operational data like variable capacity factors; build_year
    # shows gens built in a particular year, used to gather construction data
    # like capital cost and capacity built)
    gens_by_model_year, gens_by_build_year = gen_tables(
        gc, pudl_engine, scen_settings_dict
    )

    #########
    # create Switch input files from these tables

    gen_build_costs_file(gens_by_build_year, out_folder)

    # This uses gens_by_model_year, to increase the chance that a generator
    # cluster that exists in an early year will also be modeled in a later year,
    # even if it retires, so we can reuse that info in chained models where we
    # turn off age-based retirement.
    # We have to send the fuel prices so it can check which gens use a real fuel
    # and which don't, because PowerGenome gives a heat rate for all of them.
    gen_info_file(first_year_settings, gens_by_model_year, all_fuel_prices, out_folder)

    # balancing_tables(first_year_settings, pudl_engine, all_gen_units, out_folder)

    gen_build_predetermined_file(gens_by_build_year, out_folder)

    operational_files(
        scen_settings_dict,
        pg_engine,
        hydro_variability_new,
        gens_by_model_year,
        out_folder,
    )


def operational_files(
    scen_settings_dict,
    pg_engine,
    hydro_variability_new,
    gens_by_model_year,
    out_folder,
):
    """
    Create all files describing time-varying operation of the system, i.e.,
    for loads, hydro, variable capacity factors for renewables, etc.
    """

    timepoint_start = 1
    # will hold all years of each type of data
    output = collections.defaultdict(list)

    timepoint_start = 1
    for model_year, year_settings in scen_settings_dict.items():

        period_all_gen = gens_by_model_year.query("model_year == @model_year")
        print("Gathering generator variability data.")
        period_all_gen_variability = make_generator_variability(period_all_gen)
        period_all_gen_variability.columns = period_all_gen["Resource"]
        if "gen_is_baseload" in period_all_gen.columns:
            period_all_gen_variability = set_must_run_generation(
                period_all_gen_variability,
                period_all_gen.loc[
                    period_all_gen["gen_is_baseload"] == True, "Resource"
                ].to_list(),
            )

        # TODO: is this needed? can it be eliminated by improvements upstream?
        # ####### add by Rangrang, need to discuss further about CF of hydros in MIS_D_MD
        # change the variability of hyfro generators in MIS_D_MS
        # the profiles for them were missing and were filled with 1, which does not make sense since
        # all variable resources should have a variable capacity factoe between 0-1.
        hydro_variability_new = hydro_variability_new.iloc[:8760]
        MIS_D_MS_hydro = [
            col
            for col in period_all_gen_variability.columns
            if "MIS_D_MS" in col
            if "hydro" in col
        ]
        for col in MIS_D_MS_hydro:
            period_all_gen_variability[col] = hydro_variability_new["MIS_D_MS"]

        period_lc = make_final_load_curves(pg_engine, year_settings)

        cluster_time = year_settings.get("reduce_time_domain") is True

        # do time clustering/sampling
        if cluster_time:
            assert "time_domain_periods" in year_settings
            assert "time_domain_days_per_period" in year_settings

            # results is a dict with keys "resource_profiles" (gen_variability), "load_profiles",
            # "time_series_mapping" (maps clusters sequentially to potential periods in year),
            # "ClusterWeights", etc. See PG for full details.
            print("Beginning clustering of timeseries.")
            results, representative_point, weights = kmeans_time_clustering(
                resource_profiles=period_all_gen_variability,
                load_profiles=period_lc,
                days_in_group=year_settings["time_domain_days_per_period"],
                num_clusters=year_settings["time_domain_periods"],
                include_peak_day=year_settings.get("include_peak_day", True),
                load_weight=year_settings.get("demand_weight_factor", 1),
                variable_resources_only=year_settings.get(
                    "variable_resources_only", True
                ),
            )
            print("Finished clustering timeseries.")
            period_lc_sampled = results["load_profiles"]
            period_variability_sampled = results["resource_profiles"]

        #######
        # Omit existing generators that have no active capacity this period.
        # In some cases, PowerGenome may include generators that are post-
        # retirement (or maybe pre-construction?), to make sure the same sample
        # weeks are selected for every period. Here we filter those out because
        # Switch will not accept time-varying data for generators that cannot be
        # used.
        period_all_gen = period_all_gen.query("Existing_Cap_MW.notna() or new_build")
        period_all_gen_variability = period_all_gen_variability.loc[
            :, period_all_gen["Resource"]
        ]
        if cluster_time:
            period_variability_sampled = period_variability_sampled.loc[
                :, period_all_gen["Resource"]
            ]

        # timeseries_df and timepoints_df
        if cluster_time:
            timeseries_df, timepoints_df = ts_tp_pg_kmeans(
                representative_point["slot"],
                weights,
                year_settings["time_domain_days_per_period"],
                year_settings["model_year"],
                year_settings["model_first_planning_year"],
            )
            timepoints_df["timepoint_id"] = range(
                timepoint_start, timepoint_start + len(timepoints_df)
            )
            timepoint_start = timepoints_df["timepoint_id"].max() + 1
        else:
            if year_settings.get("full_time_domain") is True:
                timeseries_df, timepoints_df, timestamp_interval = timeseries_full(
                    period_lc_sampled,
                    year_settings["model_year"],
                    year_settings["model_first_planning_year"],
                    settings=year_settings,
                )
            else:
                timeseries_df, timepoints_df, timestamp_interval = timeseries(
                    period_lc_sampled,
                    year_settings["model_year"],
                    year_settings["model_first_planning_year"],
                    settings=year_settings,
                )

            timepoints_df["timepoint_id"] = range(
                timepoint_start, timepoint_start + len(timepoints_df)
            )
            timepoint_start = timepoints_df["timepoint_id"].max() + 1

            # create lists and dictionary for later use
            timepoints_timestamp = timepoints_df[
                "timestamp"
            ].to_list()  # timestamp list
            timepoints_tp_id = timepoints_df[
                "timepoint_id"
            ].to_list()  # timepoint_id list
            timepoints_dict = dict(
                zip(timepoints_timestamp, timepoints_tp_id)
            )  # {timestamp: timepoint_id}

        output["timeseries.csv"].append(timeseries_df)
        output["timepoints.csv"].append(timepoints_df)

        # hydro timepoint data
        if cluster_time:
            hydro_timepoints_df = hydro_timepoints_pg_kmeans(timepoints_df)
            hydro_timeseries_table = hydro_timeseries_pg_kmeans(
                period_all_gen,
                period_variability_sampled.loc[
                    :, period_all_gen.loc[period_all_gen["HYDRO"] == 1, "Resource"]
                ],
                hydro_timepoints_df,
            )
        else:
            hydro_timepoints_df, hydro_timeseries_table = hydro_time_tables(
                period_all_gen,
                period_all_gen_variability,
                timepoints_df,
                year_settings["model_year"],
            )
        output["hydro_timepoints.csv"].append(hydro_timepoints_df)
        output["hydro_timeseries.csv"].append(hydro_timeseries_table)

        # hydro network data
        if cluster_time:
            (
                water_nodes,
                water_connections,
                reservoirs,
                hydro_pj,
                water_node_tp_flows,
            ) = hydro_system_module_tables(
                period_all_gen,
                period_variability_sampled.loc[
                    :, period_all_gen.loc[period_all_gen["HYDRO"] == 1, "Resource"]
                ],
                hydro_timepoints_df,
                flow_per_mw=1.02,
            )
        else:
            (
                water_nodes,
                water_connections,
                reservoirs,
                hydro_pj,
                water_node_tp_flows,
            ) = hydro_system_module_tables(
                period_all_gen,
                period_all_gen_variability.loc[
                    :, period_all_gen.loc[period_all_gen["HYDRO"] == 1, "Resource"]
                ],
                timepoints_df,
                flow_per_mw=1.02,
            )
        output["water_nodes.csv"].append(water_nodes)
        output["water_connections.csv"].append(water_connections)
        output["reservoirs.csv"].append(reservoirs)
        output["hydro_generation_projects.csv"].append(hydro_pj)
        output["water_node_tp_flows.csv"].append(water_node_tp_flows)

        # loads
        if cluster_time:
            loads = load_pg_kmeans(period_lc_sampled, timepoints_df)
            timepoints_tp_id = timepoints_df[
                "timepoint_id"
            ].to_list()  # timepoint_id list
            dummy_df = pd.DataFrame({"TIMEPOINT": timepoints_tp_id})
            dummy_df.insert(0, "LOAD_ZONE", "loadzone")
            dummy_df.insert(2, "zone_demand_mw", 0)
            loads = loads.append(dummy_df)
        else:
            loads, loads_with_year_hour = loads_table(
                period_lc_sampled,
                timepoints_timestamp,
                timepoints_dict,
                year_settings["model_year"],
            )
            # for fuel_cost and regional_fuel_market issue
            dummy_df = pd.DataFrame({"TIMEPOINT": timepoints_tp_id})
            dummy_df.insert(0, "LOAD_ZONE", "loadzone")
            dummy_df.insert(2, "zone_demand_mw", 0)
            loads = loads.append(dummy_df)
            # year_hour is used by vcf below
            year_hour = loads_with_year_hour["year_hour"].to_list()
        output["loads.csv"].append(loads)

        # capacity factors for variable generators
        if cluster_time:
            vcf = variable_cf_pg_kmeans(
                period_all_gen, period_variability_sampled, timepoints_df
            )
        else:
            vcf = variable_capacity_factors_table(
                period_all_gen_variability,
                year_hour,
                timepoints_dict,
                period_all_gen,
                year_settings["model_year"],
            )
        output["variable_capacity_factors.csv"].append(vcf)

        # timestamp map for graphs
        if cluster_time:
            graph_timestamp_map = graph_timestamp_map_kmeans(timepoints_df)
        else:
            graph_timestamp_map = graph_timestamp_map_table(
                timeseries_df, timestamp_interval
            )
        output["graph_timestamp_map.csv"].append(graph_timestamp_map)

    # drop_duplicates isn't enough for some files, because they may have
    # different capacities calculated in different years (!)
    aggregation_rules = {
        "reservoirs.csv": {"res_min_vol": "min", "res_max_vol": "max"},
        "water_connections.csv": {"wc_capacity": "max"},
    }
    for file, agg_rule in aggregation_rules.items():
        df = pd.concat(output[file])
        group_cols = df.columns.difference(agg_rule.keys())
        df = df.groupby(group_cols.to_list()).agg(agg_rule).reset_index()
        output[file] = [df]

    # Write to CSV files (remove any remaining duplicate rows, e.g., based on the
    # same generator reported in different model years)
    for file, dfs in output.items():
        pd.concat(dfs).drop_duplicates().to_csv(out_folder / file, index=False)


def gen_build_costs_file(gens_by_build_year, out_folder):
    """
    Input:
        * gens_by_build_year: from gen_tables, based on gc.create_all_gens
        * out_folder: directory to store the output
    Output columns
        * GENERATION_PROJECT: Resourc
        * build_year: based off of the build years from gens_by_build_year
        * gen_overnight_cost: uses PG capex_mw and regional_cost_multiplier
        * gen_fixed_om: uses PG Fixed_OM_Cost_per_MWyr_mean for all generators
        * gen_storage_energy_overnight_cost: uses PG capex_mw and regional_cost_multiplier
        * gen_storage_energy_fixed_om: PG Fixed_OM_Cost_per_MWhyr for all generators
    """
    defs = {
        "GENERATION_PROJECT": "Resource",
        "BUILD_YEAR": "build_year",
        "gen_overnight_cost": "capex_mw * regional_cost_multiplier",
        "gen_fixed_om": "Fixed_OM_Cost_per_MWyr_mean",
        "gen_storage_energy_overnight_cost": "capex_mwh * regional_cost_multiplier",
        "gen_storage_energy_fixed_om": "Fixed_OM_Cost_per_MWhyr",
    }

    gen_build_costs = pd.DataFrame(
        {col: gens_by_build_year.eval(expr) for col, expr in defs.items()}
    )

    gen_build_costs.to_csv(out_folder / "gen_build_costs.csv", index=False, na_rep=".")


def gen_build_predetermined_file(gens_by_build_year, out_folder):
    """
    Output columns
        * GENERATION_PROJECT: Resource from gens_by_build_year
        * build_year: from gens_by_build_year
        * build_gen_predetermined: based on capacity_mw from gens_by_build_year
        * build_gen_energy_predetermined: based on capacity_mwh from gens_by_build_year
    """

    # write the relevant columns out for Switch
    gbp_cols = {
        "Resource": "GENERATION_PROJECT",
        "build_year": "build_year",
        "capacity_mw": "build_gen_predetermined",
        "capacity_mwh": "build_gen_energy_predetermined",
    }

    gbp = gens_by_build_year.loc[gens_by_build_year["existing"], gbp_cols.keys()]
    gbp = gbp.rename(columns=gbp_cols)

    gbp.to_csv(out_folder / "gen_build_predetermined.csv", index=False, na_rep=".")


def gen_info_file(
    settings,
    gens_by_model_year: pd.DataFrame,
    fuel_prices: pd.DataFrame,
    out_folder: Path,
):
    # consolidate to one row per generator cluster (we assume data is the same
    # for all rows)
    gens = gens_by_model_year.drop_duplicates(subset="Resource")

    set_retirement_age(gens, settings)

    gen_info = gen_info_table(
        gens,
        settings.get("transmission_investment_cost")["spur"]["capex_mw_mile"],
    )

    graph_tech_colors_data = {
        "gen_type": [
            "Biomass",
            "Coal",
            "Naturalgas",
            "Geothermal",
            "Hydro",
            "Nuclear",
            "Oil",
            "Solar",
            "Storage",
            "Waste",
            "Wave",
            "Wind",
            "Other",
        ],
        "color": [
            "green",
            "saddlebrown",
            "gray",
            "red",
            "royalblue",
            "blueviolet",
            "orange",
            "gold",
            "aquamarine",
            "black",
            "blue",
            "deepskyblue",
            "white",
        ],
    }
    graph_tech_colors_table = pd.DataFrame(graph_tech_colors_data)
    graph_tech_colors_table.insert(0, "map_name", "default")
    graph_tech_colors_table

    graph_tech_types_table = gen_info.drop_duplicates(subset="gen_tech")
    graph_tech_types_table["map_name"] = "default"
    graph_tech_types_table["energy_source"] = graph_tech_types_table[
        "gen_energy_source"
    ]

    cols = ["map_name", "gen_type", "gen_tech", "energy_source"]
    graph_tech_types_table = graph_tech_types_table[cols]

    # Drop the heat rate that PowerGenome provides for many non-fuel-using generators
    fuels = fuel_prices["fuel"].unique()
    fuels = [fuel.capitalize() for fuel in fuels]
    non_fuel_table = graph_tech_types_table[
        ~graph_tech_types_table["energy_source"].isin(fuels)
    ]
    non_fuel_energy_table = (
        non_fuel_table[["energy_source"]].drop_duplicates().sort_values("energy_source")
    )
    gen_info.loc[
        gen_info["gen_energy_source"].isin(non_fuel_energy_table["energy_source"]),
        "gen_full_load_heat_rate",
    ] = None

    graph_tech_colors_table.to_csv(out_folder / "graph_tech_colors.csv", index=False)
    graph_tech_types_table.to_csv(out_folder / "graph_tech_types.csv", index=False)
    non_fuel_energy_table.to_csv(
        out_folder / "non_fuel_energy_sources.csv", index=False
    )

    # identify generators participating in ESR or minimum capacity programs,
    # then drop those columns
    ESR_col = [col for col in gen_info.columns if col.startswith("ESR")]
    ESR_generators = gen_info[["GENERATION_PROJECT"] + ESR_col]
    min_cap_col = [col for col in gen_info.columns if col.startswith("MinCapTag")]
    min_cap_gens = gen_info[["GENERATION_PROJECT"] + min_cap_col]
    gen_info = gen_info.drop(columns=ESR_col + min_cap_col)

    gen_info.to_csv(out_folder / "gen_info.csv", index=False, na_rep=".")

    # save deviations from mean O&M cost in gen_om_by_period.csv to allow variation by study period.
    om_cols = [
        "Fixed_OM_Cost_per_MWyr",
        "Var_OM_Cost_per_MWh",
        "Fixed_OM_Cost_per_MWhyr",
    ]
    # drop existing generators that are retired by this time
    gen_om_by_period = gens_by_model_year.query("Existing_Cap_MW.notna() or new_build")
    # calculate difference from the mean
    gen_om_by_period[om_cols] -= gen_om_by_period[[c + "_mean" for c in om_cols]].values
    # ignore tiny differences from the mean
    gen_om_by_period[om_cols] = gen_om_by_period[om_cols].mask(
        gen_om_by_period[om_cols].abs() <= 1e-9, 0
    )
    # drop zeros (not essential, but helpful for seeing only the ones with adjustments)
    gen_om_by_period[om_cols] = gen_om_by_period[om_cols].replace({0: float("nan")})
    gen_om_by_period = gen_om_by_period.dropna(subset=om_cols, how="all")

    # filter columns
    gen_om_by_period = gen_om_by_period[["Resource", "model_year"] + om_cols]
    gen_om_by_period.columns = [
        "GENERATION_PROJECT",
        "PERIOD",
        "gen_fixed_om_by_period",
        "gen_variable_om_by_period",
        "gen_storage_energy_fixed_om_by_period",
    ]
    gen_om_by_period.to_csv(
        out_folder / "gen_om_by_period.csv", index=False, na_rep="."
    )

    ################
    # ESR and min_cap programs

    # create esr_generators.csv: list of generators participating in ESR (RPS/CES) programs
    ESR_generators_long = pd.melt(
        ESR_generators, id_vars=["GENERATION_PROJECT"], value_vars=ESR_col
    )
    ESR_generators_long = ESR_generators_long[ESR_generators_long["value"] == 1].rename(
        columns={"variable": "ESR_PROGRAM", "GENERATION_PROJECT": "ESR_GEN"}
    )
    ESR_generators_long = ESR_generators_long[["ESR_PROGRAM", "ESR_GEN"]]
    ESR_generators_long.to_csv(out_folder / "esr_generators.csv", index=False)

    # make min_cap_generators.csv, showing generators that can help satisfy
    # minimum capacity rules
    min_cap_generators_long = pd.melt(
        min_cap_gens, id_vars=["GENERATION_PROJECT"], value_vars=min_cap_col
    )
    min_cap_generators_long = min_cap_generators_long[
        min_cap_generators_long["value"] == 1
    ].rename(
        columns={"variable": "MIN_CAP_PROGRAM", "GENERATION_PROJECT": "MIN_CAP_GEN"}
    )
    min_cap_generators_long = min_cap_generators_long[
        ["MIN_CAP_PROGRAM", "MIN_CAP_GEN"]
    ]
    min_cap_generators_long.to_csv(out_folder / "min_cap_generators.csv", index=False)
    ###############################################################


def gen_tables(gc, pudl_engine, scen_settings_dict):
    """
    Return dataframes showing all generator clusters that can be operated in
    each model_year and that can be built in each build_year. gens_by_model_year
    has one row for every model_year when the generator or unit can be operated.
    gens_by_build_year has one row for every build_year when the generators can
    be built or were built.

    These dataframes each show both new and existing generators. They contain
    all the data from gc.create_all_generators() (and gc.units_model after
    running this) plus some extra data.

    The "existing" column identifies generators that have a scheduled
    construction plan in the past or near future; for these, capacity_mw and
    possibly capacity_mwh will also have values. The "new_build" column
    identifies generators that can be built during the study period.

    This is the main place where generator data is read from PowerGenome, so it
    is also the best place to filter or adapt the data as needed before use
    elsewhere.

    Note: this changes all the generator-related attributes of gc.
    """
    # we save and restore gc.settings, but calling gc.create_all_generators()
    # has unknown side effects on gc, including updating all the
    # generator-related attributes.
    orig_gc_settings = gc.settings
    gen_dfs = []
    unit_dfs = []
    """
    # for testing:
    year_settings = first_value(scen_settings_dict)
    """
    for year_settings in scen_settings_dict.values():

        """
        # for testing:
        gen_df = gc.all_resources.copy()
        """
        gc.settings = year_settings
        gen_df = gc.create_all_generators().copy()

        # identify existing and new-build for reference later
        # (these could overlap in principle, but don't as of Feb. 2024)
        if gc.current_gens:
            gen_df["existing"] = gen_df["Resource"].isin(
                gc.existing_resources["Resource"]
            )
        else:  # must all be new
            gen_df["existing"] = False
        gen_df["new_build"] = gen_df["Resource"].isin(gc.new_resources["Resource"])

        # clean up some resource and technology labels
        gen_df["Resource"] = gen_df["Resource"].str.rstrip("_")
        gen_df["technology"] = gen_df["technology"].str.rstrip("_")

        # gather some extra data from PowerGenome
        gen_df = add_misc_gen_values(gen_df, year_settings)
        gen_df = hydro_energy_to_power(
            gen_df,
            year_settings.get("hydro_factor"),
            year_settings.get("regional_hydro_factor", {}),
        )
        gen_df = add_co2_costs_to_o_m(gen_df)

        # apply capacity derating if needed (e.g., to get the right average
        # output for small hydro); Switch restricts output via
        # gen_forced_outage_rate; we supersede the previous forced outage rate,
        # since we assume the capacity_factor is based on historical output,
        # including the effect of forced outages
        if year_settings.get("derate_capacity"):
            derate = gen_df["technology"].isin(year_settings.get("derate_techs", []))
            gen_df.loc[derate, "gen_forced_outage_rate"] = 1 - gen_df.loc[
                derate, "capacity_factor"
            ].fillna(1).clip(0, 1)

        # If running an operation model, only consider existing projects. This
        # is rarely used; normally we setup operation models based on solved
        # capacity-planning models, but if specified, we drop the option for
        # new gens.
        if year_settings.get("operation_model"):
            gen_df = gen_df.loc[gen_df["existing"], :]
            # make sure new_build is turned off for any that overlap
            gen_df = gen_df["new_build"] = False

        # in greenfield scenarios, Existing_Cap_MW might be omitted
        if "Existing_Cap_MW" not in gen_df.columns:
            gen_df["Existing_Cap_MW"] = float("nan")

        # identify storage gens for the next few steps
        storage_gens = gen_df["STOR"].astype(bool)

        # Use $0 as capex and fixed O&M for existing plants (our settings don't
        # have all of these for existing plants as of Mar 2024)
        for c in ["capex_mw", "Fixed_OM_Cost_per_MWyr"]:
            gen_df[c] = gen_df[c].fillna(0)
        for c in ["capex_mwh", "Fixed_OM_Cost_per_MWhyr"]:
            gen_df.loc[storage_gens, c] = gen_df.loc[storage_gens, c].fillna(0)

        # Use 1 as regional_cost_multiplier if not specified (i.e., for existing gens)
        gen_df["regional_cost_multiplier"] = gen_df["regional_cost_multiplier"].fillna(
            1
        )

        # Remove storage-related params for non-storage gens (we get a lot of
        # these as of Mar 2024)
        gen_df.loc[
            ~storage_gens,
            ["Existing_Cap_MWh", "capex_mwh", "Fixed_OM_Cost_per_MWhyr"],
        ] = None

        # record which model year these generators could be used in
        gen_df["model_year"] = year_settings["model_year"]

        gen_dfs.append(gen_df)

        # find build_year, capacity_mw and capacity_mwh for existing generating
        # units online in this model_year for each gen cluster
        eia_unit_info = eia_build_info(gc)
        unit_df = gen_df.merge(eia_unit_info, on="Resource", how="left")
        unit_dfs.append(unit_df)

    gc.settings = orig_gc_settings

    gens_by_model_year = pd.concat(gen_dfs, ignore_index=True)
    units_by_model_year = pd.concat(unit_dfs, ignore_index=True)

    # Set same info as eia_build_info() (build_year, capacity_mw and
    # capacity_mwh) for generic generators (Resources in the "existing" list
    # that didn't get matching record(s) from the eia_unit_info, currently only
    # distributed generation). We do this after the loop so we can infer a
    # sequence of capacity additions that results in the available capacity
    # reported for each model year.
    generic = units_by_model_year["existing"] & units_by_model_year["build_year"].isna()
    generic_units = units_by_model_year[generic].drop(
        columns=["plant_gen_id", "build_year", "capacity_mw", "capacity_mwh"]
    )
    generic_units = generic_units.merge(
        generic_gen_build_info(generic_units, first_value(scen_settings_dict)),
        on="Resource",
        how="left",
    )
    units_by_model_year = (
        pd.concat([units_by_model_year[~generic], generic_units])
        .sort_values(["Resource", "model_year", "build_year"])
        .reset_index()
    )

    assert (
        units_by_model_year.query("existing")["build_year"].notna().all()
    ), "Some existing generating units have no build_year assigned."

    # In PowerGenome, Fixed_OM_Cost_per_MWyr, Var_OM_Cost_per_MWh and Fixed_OM_Cost_per_MWhyr vary by
    # model year, not build year. So we calculate averages across model years to
    # use for all build years, but also leave the per-model-year values to use
    # as an ancillary input. (The averages are Fixed_OM_Cost_per_MWyr_mean,
    # Var_OM_Cost_per_MWh_mean and Fixed_OM_Cost_per_MWhyr_mean.)
    for col in [
        "Fixed_OM_Cost_per_MWyr",
        "Var_OM_Cost_per_MWh",
        "Fixed_OM_Cost_per_MWhyr",
    ]:
        mean = gens_by_model_year.groupby("Resource")[col].mean()
        gens_by_model_year[col + "_mean"] = gens_by_model_year["Resource"].map(mean)
        # gens_by_model_year[col + "_mean"] = (
        #     gens_by_model_year.groupby("Resource")[col].transform(lambda x: x.mean()
        # )

    # create by_build_year tables from these

    # Merge the repeated records for existing gens from different model years.
    # We take the first row found for non-numeric columns and the mean for
    # numeric columns, since values may vary across model years (e.g.,
    # capacity_mw for a single unit can change between model years due to
    # derating by the cluster average capacity factor, since the cluster makeup
    # changes over time; fixed O&M rises over time for some plants)
    numeric_cols = unit_df.select_dtypes(include="number").columns.drop("build_year")
    dup_rules = {
        c: "mean" if c in numeric_cols else "first" for c in units_by_model_year.columns
    }
    unit_info = units_by_model_year.groupby(
        ["Resource", "plant_gen_id", "build_year"], as_index=False
    ).agg(dup_rules)
    # average model_year is not meaningful
    unit_info = unit_info.drop(columns=["model_year"])

    # aggregate by build_year
    # this could in theory be done with groupby()[columns].sum(), but then
    # pandas 1.4.4 sometimes drops capacity_mw for this dataframe (seems to happen
    # with columns where one's name is a shorter version of the other, and can't
    # be reproduced if you save the table as .csv and read it back in.)
    build_year_info = unit_info.groupby(["Resource", "build_year"], as_index=False).agg(
        {"capacity_mw": "sum", "capacity_mwh": "sum"}
    )

    # Existing gen clusters are duplicated across model years, so we first
    # consolidate to one row per resource, then replicate data for each
    # resource/build_year combo
    existing_gens = gens_by_model_year.query("existing").drop_duplicates(
        subset="Resource", keep="first"
    )
    # turn off "new_build" flag if set; those will be duplicated in
    # new_gens_by_build_year
    existing_gens["new_build"] = False
    existing_gens_by_build_year = existing_gens.merge(
        build_year_info, on="Resource", how="left"
    )

    # Create dataframe showing when the new generators can be built and
    # consolidating by build year instead of model year. This is simple, since
    # for new gens the build_year is the same as the model_year (and that's the
    # year for which costs are shown)
    new_gens_by_build_year = gens_by_model_year.query("new_build")
    new_gens_by_build_year["build_year"] = new_gens_by_build_year["model_year"]
    # turn off "existing" flag if set; those are duplicated in
    # existing_gens_by_build_year
    new_gens_by_build_year["existing"] = False

    gens_by_build_year = pd.concat(
        [existing_gens_by_build_year, new_gens_by_build_year], ignore_index=True
    )
    # Remove storage-related params (always 0?) for non-storage gens
    # (could be done in the loop above, and in theory the sums would come out
    # as NaNs, but in practice they sometimes come out as 0 instead)
    gens_by_build_year.loc[
        gens_by_build_year["STOR"] == 0,
        [
            "Existing_Cap_MWh",
            "capex_mwh",
            "Fixed_OM_Cost_per_MWhyr",
            "capacity_mwh",
        ],
    ] = None

    assert (
        gens_by_build_year["new_build"] & gens_by_build_year["Existing_Cap_MW"].notna()
    ).sum() == 0, "Some new-build generators have Existing_Cap_MW assigned."

    return gens_by_model_year, gens_by_build_year


def set_retirement_age(df, settings):
    # set retirement age (500 years if not specified) This uses the same logic
    # as powergenome.generators.label_retirement_year(), which doesn't seem to
    # get called for a lot of the generators we are using.
    # Note: In the economic retirement cases, retirement_ages is set as ~ in the
    # .yml files, which comes back as None instead of a missing entry or empty
    # dict, so we work around that
    retirement_ages = settings.get("retirement_ages") or {}
    df["retirement_age"] = df["technology"].map(retirement_ages).fillna(500)
    return df


def eia_build_info(gc: GeneratorClusters):
    """
    Return a dataframe showing Resource, plant_gen_id, build_year, capacity_mw
    and capacity_mwh for all EIA generating units that were aggregated for the
    previous call to gc.create_all_generators().

    Note: capacity_mw will be de-rated according to the unit's average capacity
    factor if specified in gc.settings (typical for small hydro, geothermal,
    possibly biomass)

    Inputs: - gc: GeneratorClusters object previously used to call
    gc.create_all_generators
    """

    units = gc.all_units.copy()

    if "Resource" not in units.columns:
        # PowerGenome before March 2024
        # Construct a resource ID the same way PowerGenome does when "extra_outputs" is set
        units["Resource"] = (
            units["model_region"]
            + "_"
            + snake_case_col(units["technology"])
            + "_"
            + units["cluster"].astype(str)
        )

    # assign unique ID for each unit, for de-duplication later
    units = create_plant_gen_id(units)

    # Set retirement age (units has a retirement_age, but it's not completely
    # filled in.) We do this temporarily here so we can back-calculate the
    # right in-service year to assign
    set_retirement_age(units, gc.settings)

    # drop any with no retirement year (generally only ~1 planned builds that
    # don't have an online date so PG couldn't assign a retirement date)
    units = units.query("retirement_year.notna()")

    # infer the build date from retirement_year and retirement_age
    # (may not be the right year, but will cause it to retire at the right
    # time, which is generally most important)
    units["build_year"] = (units["retirement_year"] - units["retirement_age"]).astype(
        "Int64"
    )

    # make sure "capacity_mw" comes from the right column
    capacity_col = gc.settings.get("capacity_col", "capacity_mw")
    units["capacity_mw"] = units[capacity_col]

    return units[
        [
            "Resource",
            "plant_gen_id",
            "build_year",
            "capacity_mw",
            "capacity_mwh",
        ]
    ]


def as_col(series):
    # convert pandas series to numpy column
    return series.to_numpy()[:, np.newaxis]


def infer_build_years(df):
    """
    Find capacity built in specific years that would make the specified amount
    available in each model year, taking account of retirements. `df` must
    contain `retirement_age`, `model_year` and `Existing_Cap_MW`. Return
    dataframe showing `Resource`, `build_year` and `capacity_mw`.

    If exact solution is not possible, we use a least-squares fit instead.

    This sets up a linear algebra problem that sums the construction in each
    build_year that would still be active in each model_year. Then it uses a
    scipy solver to find the amount to add in each build_year to get the right
    amount for each model_year. This solves A x = b for x, where x is the amount
    added in each build_year (column vector), b is the amount online in each
    model_year (column vector), and A is a matrix with 1 for every build_year
    (column) that is available to use in each model_year (row). This uses a
    least-squares solver with non-negative x values. If the system is
    undertermined (generally true), it will find an exact solution. If an exact
    solution is not possible (e.g., there are multiple dips in capacity within
    the lifespan of the asset), it will find a least-squares fit and issue a
    warning.
    """
    # df = pd.DataFrame({'Resource': ['a', 'a', 'a'], 'model_year': [2025, 2030, 2035], 'retirement_age': [30, 30, 30], 'Existing_Cap_MW': [10, 20, 10], 'Existing_Cap_MWh': [5, 10, 10]})
    first_build_year = (df["model_year"] - df["retirement_age"] + 1).min()
    last_build_year = df["model_year"].max()
    # reverse order of years so the algorithm will prefer later ones
    build_year = np.arange(last_build_year, first_build_year - 1, -1)
    # identify build_years (columns) that would still be in service for each
    # model_year (row)
    in_service_flag = (
        (build_year <= as_col(df["model_year"]))
        & (build_year > as_col(df["model_year"] - df["retirement_age"]))
    ).astype(int)
    # now we want a vector showing capacity built in each year such that
    # in_service_flag ~matrix multiply~ built = Existing_Cap_MW, i.e., the flag
    # shows which build years are active for each model year, and we want the
    # sum of the in_service_flags for this model_year times capacity built each
    # build_year (built_mw) to match Existing_Cap_MW for this model_year. This
    # can be seen as a non-negative least-squares problem:
    built_mw, rnorm_mw = scipy.optimize.nnls(in_service_flag, df["Existing_Cap_MW"])
    built_mwh, rnorm_mwh = scipy.optimize.nnls(in_service_flag, df["Existing_Cap_MWh"])
    if rnorm_mw > 0:
        print(
            f"WARNING: MW construction schedule for {df['Resource'].iloc[0]} cannot match reported capacity"
        )
    if rnorm_mwh > 0:
        print(
            f"WARNING: MWh construction schedule for {df['Resource'].iloc[0]} cannot match reported capacity"
        )

    result = pd.DataFrame(
        {
            "build_year": build_year,
            "capacity_mw": built_mw,
            "capacity_mwh": built_mwh,
        }
    ).round(6)
    # drop 0's and then drop any empty rows
    result[["capacity_mw", "capacity_mwh"]] = result[
        ["capacity_mw", "capacity_mwh"]
    ].replace(0.0, np.nan)
    result = result.dropna(subset=["capacity_mw", "capacity_mwh"], how="all")
    return result


def generic_gen_build_info(gens, settings):
    """
    Return dataframe with Resource, dummy generator id and inferred build_year,
    capacity_mw and capacity_mwh columns for generic existing generators.

    These are generators that PowerGenome reported as existing but didn't get
    unit-level construction info from eia_build_info(), e.g., distributed
    generation.

    The gens dataframe must have Resource, retirement_age, model_year,
    Existing_Cap_MW and Existing_Cap_MWh (capacity online as of that year). The
    construction plan is achieves the specified capacity as of each model year
    if possible.

    This sets up a least-squares problem to find build_years and quantities that
    are compatible with the reported total capacity online for each resource:
    minimize (Ax - b)^2, subject to x >= 0, where A is the build_year:model_year
    correspondence matrix (1 for any build years that are active in a particular
    model year), x is the capacity built each year, and b is the capacity online
    in each model year.

    For monotonically increasing capacity or capacity with one dip, this should
    always have an exact solution. For more complex patterns, especially with
    long retirement ages, an exact solution may not be possible, in which case a
    warning will be shown.
    """
    # gens = pd.DataFrame({'Resource': ['a', 'a', 'a'], 'model_year': [2025, 2030, 2035], 'retirement_age': [30, 30, 30], 'Existing_Cap_MW': [10, 20, 10], 'Existing_Cap_MWh': [5, 20, 10]})

    # Set retirement age for use in installation date calculations (also
    # calculated in other places but not kept; maybe these should be
    # consolidated?)
    gens = set_retirement_age(gens.copy(), settings)
    gens["Existing_Cap_MW"] = gens["Existing_Cap_MW"].fillna(0.0)
    gens["Existing_Cap_MWh"] = gens["Existing_Cap_MWh"].fillna(0.0)

    result = (
        gens.groupby("Resource")[
            "retirement_age", "model_year", "Existing_Cap_MW", "Existing_Cap_MWh"
        ]
        .apply(infer_build_years)
        .reset_index()
        .drop(columns=["level_1"])
    )
    result["plant_gen_id"] = "generic"
    return result


def other_tables(
    scen_settings_dict,
    out_folder,
):
    # get first year settings, which include any generic, cross-year settings
    first_year_settings = first_value(scen_settings_dict)

    if first_year_settings.get("emission_policies_fn"):

        # create carbon_policies_regional.csv
        dfs = [
            # start with a dummy data frame, so we always get some output
            pd.DataFrame(
                columns=[
                    "CO2_PROGRAM",
                    "PERIOD",
                    "LOAD_ZONE",
                    "carbon_cap_tco2_per_yr",
                    "carbon_cost_dollar_per_tco2",
                ]
            )
        ]
        # gather data across years
        for model_year, scen_settings in scen_settings_dict.items():
            co2_cap = create_policy_req(scen_settings, col_str_match="CO_2_")
            if co2_cap is not None:
                co2_cap["PERIOD"] = model_year
                # mask out caps for regions that aren't participating in programs
                co2_target_cols = [
                    c for c in co2_cap.columns if c.startswith("CO_2_Max_Mtons_")
                ]
                for tc in co2_target_cols:
                    participate = co2_cap[
                        tc.replace("CO_2_Max_Mtons_", "CO_2_Cap_Zone_")
                    ].astype(bool)
                    co2_cap.loc[~participate, tc] = float("nan")
                # rescale caps from millions of tonnes to tonnes
                co2_cap[co2_target_cols] *= 1000000
                # keep only necessary columns
                co2_cap = co2_cap[["Region_description", "PERIOD"] + co2_target_cols]
                # convert existing cols into load zones and program names
                co2_cap = co2_cap.rename(columns={"Region_description": "LOAD_ZONE"})
                co2_cap.columns = co2_cap.columns.str.replace("CO_2_Max_Mtons_", "ETS ")
                # switch from wide to long format and drop the non-participants
                co2_cap_long = co2_cap.melt(
                    id_vars=["LOAD_ZONE", "PERIOD"],
                    var_name="CO2_PROGRAM",
                    value_name="carbon_cap_tco2_per_yr",
                ).dropna(subset=["carbon_cap_tco2_per_yr"])
                # Add the carbon cost if available
                co2_cap_long["carbon_cost_dollar_per_tco2"] = scen_settings.get(
                    "carbon_cost_dollar_per_tco2", "."
                )
                # reorder the columns for Switch
                co2_cap_long = co2_cap_long[dfs[0].columns]
                dfs.append(co2_cap_long)

        # aggregate annual data and write to file
        co2_cap_long = pd.concat(dfs, axis=0)
        co2_cap_long.to_csv(out_folder / "carbon_policies_regional.csv", index=False)

        # create alternative versions of the carbon cap
        if not co2_cap_long.empty:
            # TODO: use input data for this
            for carbon_price in [50, 200, 1000]:
                ccl = co2_cap_long.copy()
                ccl["carbon_cost_dollar_per_tco2"] = carbon_price
                ccl.to_csv(
                    out_folder / f"carbon_policies_regional.{carbon_price}.csv",
                    index=False,
                )

        # create carbon_policies.csv (possibly empty), an all-region cap for use
        # with switch_model.policies.carbon_policies or
        # mip_modules.carbon_policies
        co2_cap = co2_cap_long.query('CO2_PROGRAM == "ETS 1"')
        co2_cap_all_regions = (
            co2_cap.groupby("PERIOD")
            .agg(
                {
                    "carbon_cap_tco2_per_yr": "sum",
                    # we use simple mean for cost per tco2, because it is not
                    # clear how it should be applied overall if it differs
                    # between regions, since it is only applied to excess beyond
                    # the target (and in practice, in the code above it will
                    # always be consistent between regions)
                    "carbon_cost_dollar_per_tco2": "mean",
                }
            )
            .reset_index()
        )
        co2_cap_all_regions.to_csv(out_folder / "carbon_policies.csv", index=False)

        # create esr_requirements.csv with clean energy standards / RPS requirements
        dfs = [
            # start with a dummy data frame with standard columns
            pd.DataFrame(columns=["ESR_PROGRAM", "PERIOD", "load_zone", "rps_share"])
        ]
        # gather data across years
        for model_year, scen_settings in scen_settings_dict.items():
            energy_share_req = create_policy_req(scen_settings, col_str_match="ESR")
            if energy_share_req is not None:
                ESR_col = [
                    col for col in energy_share_req.columns if col.startswith("ESR")
                ]
                energy_share_long = pd.melt(
                    energy_share_req, id_vars=["Region_description"], value_vars=ESR_col
                )
                energy_share_long["PERIOD"] = model_year
                energy_share_long = energy_share_long.rename(
                    columns={
                        "variable": "ESR_PROGRAM",
                        "Region_description": "load_zone",
                        "value": "rps_share",
                    }
                )
                # drop the zero-value and nan rows (no policy in effect)
                energy_share_long.loc[
                    energy_share_long["rps_share"] == 0, "rps_share"
                ] = float("nan")
                energy_share_long = energy_share_long.dropna(subset=["rps_share"])
                # use standard columns in standard order
                energy_share_long = energy_share_long[dfs[0].columns]
                dfs.append(energy_share_long)

        # aggregate across years
        energy_share_long = pd.concat(dfs, axis=0)
        energy_share_long.to_csv(out_folder / "esr_requirements.csv", index=False)

        # remove any generator assignments for inactive ESR programs
        try:
            esr_gens = pd.read_csv(out_folder / "esr_generators.csv")
        except FileNotFoundError:
            pass
        else:
            esr_gens = esr_gens.loc[
                esr_gens["ESR_PROGRAM"].isin(energy_share_long["ESR_PROGRAM"]), :
            ]
            esr_gens.to_csv(out_folder / "esr_generators.csv", index=False)

        # create min_cap_requirements.csv with minimum capacity requirements for
        # some technologies (empty, if no policies defined)
        dfs = [pd.DataFrame(columns=["MIN_CAP_PROGRAM", "PERIOD", "min_cap_mw"])]
        # gather data across years
        for model_year, scen_settings in scen_settings_dict.items():
            mcr = min_cap_req(scen_settings)
            if mcr is not None:
                mcr["MIN_CAP_PROGRAM"] = "MinCapTag_" + mcr[
                    "MinCapReqConstraint"
                ].astype(str)
                mcr["PERIOD"] = model_year
                mcr = mcr.rename(
                    columns={
                        "Min_MW": "min_cap_mw",
                    }
                )
                # use standard columns in standard order
                mcr = mcr[dfs[0].columns]
                dfs.append(mcr)
        # aggregate across years and write to file
        mcr = pd.concat(dfs, axis=0)
        mcr.to_csv(out_folder / "min_cap_requirements.csv", index=False)

        # remove any generator assignments for inactive min_cap programs
        try:
            min_cap_gens = pd.read_csv(out_folder / "min_cap_generators.csv")
        except FileNotFoundError:
            pass
        else:
            min_cap_gens = min_cap_gens.loc[
                min_cap_gens["MIN_CAP_PROGRAM"].isin(mcr["MIN_CAP_PROGRAM"]), :
            ]
            min_cap_gens.to_csv(out_folder / "min_cap_generators.csv", index=False)

    # interest and discount rates in financials.csv
    financials_table = pd.DataFrame(
        {"base_financial_year": [first_year_settings["atb_data_year"]]}
    )
    for name, default in [("interest_rate", 0.05), ("discount_rate", 0.03)]:
        if name in first_year_settings:
            financials_table[name] = first_year_settings[name]
        else:
            financials_table[name] = default
            print(
                f"\nNo {name} setting found (usually in switch_params.yml); "
                f"using default value of {default}."
            )
    financials_table.to_csv(out_folder / "financials.csv", index=False)

    periods_data = {
        "INVESTMENT_PERIOD": scen_settings_dict.keys(),
        "period_start": [
            s["model_first_planning_year"] for s in scen_settings_dict.values()
        ],
        "period_end": scen_settings_dict.keys(),  # convention for MIP studies
    }
    periods_table = pd.DataFrame(periods_data)
    periods_table.to_csv(out_folder / "periods.csv", index=False)

    # this could potentially have different Voll for different segments, and it's
    # not clear what the difference is between the Voll and $/MWh columns. For
    # now, we just use the average of Voll across all segments.
    demand_segments = load_demand_segments(first_value(scen_settings_dict))
    lost_load_cost_table = pd.DataFrame(
        {"unserved_load_penalty": [demand_segments.loc[:, "Voll"].mean()]}
    )
    lost_load_cost_table.to_csv(out_folder / "lost_load_cost.csv", index=False)

    # write switch version txt file
    with open(out_folder / "switch_inputs_version.txt", "w") as file:
        file.write("2.0.9")


from powergenome.generators import load_ipm_shapefile
from powergenome.GenX import (
    network_line_loss,
    network_max_reinforcement,
    network_reinforcement_cost,
    add_cap_res_network,
)
from powergenome.transmission import (
    agg_transmission_constraints,
    transmission_line_distance,
)
from powergenome.util import init_pudl_connection, load_settings, check_settings
from statistics import mean


def transmission_tables(scen_settings_dict, out_folder, pg_engine):
    """
    pulling in information from PowerGenome transmission notebook
    Schivley Greg, PowerGenome, (2022), GitHub repository,
        https://github.com/PowerGenome/PowerGenome/blob/master/notebooks/Transmission.ipynb
    """
    # we just use settings for the first year, since these don't change across
    # years in the model
    settings = first_value(scen_settings_dict)
    model_regions = settings["model_regions"]

    transmission = agg_transmission_constraints(pg_engine=pg_engine, settings=settings)

    ## transmission lines
    # pulled from SWITCH load_zones file
    # need zone_dbid information to populate transmission_line column
    def load_zones_table(regions, zone_ccs_distance_km):
        load_zones = pd.DataFrame(
            columns=["LOAD_ZONE", "zone_ccs_distance_km", "zone_dbid"]
        )
        load_zones["LOAD_ZONE"] = regions
        load_zones["zone_ccs_distance_km"] = 0  # set to default 0
        load_zones["zone_dbid"] = range(1, len(regions) + 1)
        return load_zones

    model_regions = settings.get("model_regions")
    load_zones = load_zones_table(model_regions, zone_ccs_distance_km=0)
    zone_dict = dict(
        zip(load_zones["LOAD_ZONE"].to_list(), load_zones["zone_dbid"].to_list())
    )
    if not settings.get("user_transmission_costs"):
        model_regions_gdf = load_ipm_shapefile(settings)

        transmission_line_distance(
            trans_constraints_df=transmission,
            ipm_shapefile=model_regions_gdf,
            settings=settings,
        )

        line_loss = network_line_loss(transmission=transmission, settings=settings)
        # unused
        # reinforcement_cost = network_reinforcement_cost(
        #     transmission=transmission, settings=settings
        # )
        # max_reinforcement = network_max_reinforcement(
        #     transmission=transmission, settings=settings
        # )
        transmission = agg_transmission_constraints(
            pg_engine=pg_engine, settings=settings
        )
        add_cap = add_cap_res_network(transmission, settings)

        tx_capex_mw_mile_dict = settings.get("transmission_investment_cost")["tx"][
            "capex_mw_mile"
        ]

        transmission_lines, trans_capital_cost_per_mw_km = transmission_lines_table(
            line_loss,
            add_cap,
            tx_capex_mw_mile_dict,
            zone_dict,
            settings,
        )
        transmission_lines

    else:
        # use .csv file from settings["user_transmission_costs"]
        transmission_lines = pd.read_csv(
            settings["input_folder"] / settings["user_transmission_costs"]
        )
        # Adjust dollar year of transmission costs
        if settings.get("target_usd_year"):
            adjusted_annuities = []
            adjusted_costs = []
            for row in transmission_lines.itertuples():
                adj_annuity = inflation_price_adjustment(
                    row.total_interconnect_annuity_mw,
                    row.dollar_year,
                    settings.get("target_usd_year"),
                ).round(0)
                adjusted_annuities.append(adj_annuity)

                adj_cost = inflation_price_adjustment(
                    row.total_interconnect_cost_mw,
                    row.dollar_year,
                    settings.get("target_usd_year"),
                ).round(0)
                adjusted_costs.append(adj_cost)

            transmission_lines["total_interconnect_annuity_mw"] = adjusted_annuities
            transmission_lines["total_interconnect_cost_mw"] = adjusted_costs
            transmission_lines["adjusted_dollar_year"] = settings.get("target_usd_year")

        transmission_lines["tz1_dbid"] = transmission_lines["start_region"].map(
            zone_dict
        )

        transmission["start_region"] = (
            transmission["transmission_path_name"].str.split("_to_").str[0]
        )
        transmission["dest_region"] = (
            transmission["transmission_path_name"].str.split("_to_").str[1]
        )
        # fix the values of existing_trans_cap for the mismatched rows
        transmission_lines["line"] = (
            transmission_lines["start_region"].astype(str)
            + " "
            + transmission_lines["dest_region"].astype(str)
        )
        transmission_lines["sorted_line"] = [
            " ".join(sorted(x.split())) for x in transmission_lines["line"].tolist()
        ]
        transmission["line"] = (
            transmission["start_region"].astype(str)
            + " "
            + transmission["dest_region"].astype(str)
        )
        transmission["sorted_line"] = [
            " ".join(sorted(x.split())) for x in transmission["line"].tolist()
        ]
        transmission_lines = pd.merge(
            transmission_lines,
            transmission[["sorted_line", "Line_Max_Flow_MW"]],
            how="left",
        )

        transmission_lines, trans_capital_cost_per_mw_km = tx_cost_transform(
            transmission_lines
        )
        transmission_lines["tz2_dbid"] = transmission_lines["dest_region"].map(
            zone_dict
        )
        transmission_lines = transmission_lines.rename(
            columns={"start_region": "trans_lz1", "dest_region": "trans_lz2"}
        )
        transmission_lines["trans_dbid"] = range(1, len(transmission_lines) + 1)
        transmission_lines["trans_derating_factor"] = 0.95
        transmission_lines["TRANSMISSION_LINE"] = (
            transmission_lines["tz1_dbid"].astype(str)
            + "-"
            + transmission_lines["tz2_dbid"].astype(str)
        )
        transmission_lines = transmission_lines[
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
        # transmission_lines["existing_trans_cap"] = transmission_lines[
        #     "existing_trans_cap"
        # ].replace("", 0)
        transmission_lines.fillna(0, inplace=True)

    trans_params_table = pd.DataFrame(
        {
            "trans_capital_cost_per_mw_km": trans_capital_cost_per_mw_km,
            "trans_lifetime_yrs": 60,  # it was 20, now change to 60 for national_emm comparison by RR
            "trans_fixed_om_fraction": settings.get("trans_fixed_om_fraction", 0.0),
        },
        index=[0],
    )
    trans_params_table

    # calculate expansion limits for all lines and periods
    dfs = [
        pd.DataFrame(
            columns=["TRANSMISSION_LINE", "PERIOD", "Line_Max_Reinforcement_MW"]
        )
    ]
    trans = transmission_lines[["TRANSMISSION_LINE", "existing_trans_cap"]].rename(
        columns={"existing_trans_cap": "Line_Max_Flow_MW"}
    )
    for model_year, scen_settings in scen_settings_dict.items():
        trans["PERIOD"] = model_year
        # add Line_Max_Reinforcement_MW (expansion limit) to transmission dataframe
        # (added automatically by network_max_reinforcement(), based on
        # Line_Max_Flow_MW and scenario settings)
        network_max_reinforcement(transmission=trans, settings=scen_settings)
        dfs.append(trans[dfs[0].columns].copy())

    # combine all years into one dataframe
    trans_path_expansion_limit = pd.concat(dfs).rename(
        columns={"Line_Max_Reinforcement_MW": "trans_path_expansion_limit_mw"}
    )

    transmission_lines.to_csv(out_folder / "transmission_lines.csv", index=False)
    trans_params_table.to_csv(out_folder / "trans_params.csv", index=False)
    trans_path_expansion_limit.to_csv(
        out_folder / "trans_path_expansion_limit.csv", index=False
    )

    # create alternative transmission limits
    # TODO: use input data for this
    trans_limits = [(0, 0), (15, 400), (50, 400), (100, 400), (200, 400)]
    for frac, min_mw in trans_limits:
        dfs = []
        for year in scen_settings_dict:
            dfs.append(
                pd.DataFrame(
                    {
                        "TRANSMISSION_LINE": transmission_lines["TRANSMISSION_LINE"],
                        "PERIOD": year,
                        # next line reimplements powergenome.GenX.network_max_reinforcement
                        "trans_path_expansion_limit_mw": (
                            transmission_lines["existing_trans_cap"] * frac * 0.01
                        )
                        .clip(lower=min_mw)
                        .round(0),
                    }
                )
            )
        pd.concat(dfs).to_csv(
            out_folder / f"trans_path_expansion_limit.{frac}.csv", index=False
        )


import ast
import itertools
from statistics import mode


def balancing_tables(settings, pudl_engine, all_gen, out_folder):
    IPM_regions = settings.get("model_regions")
    bal_areas, zone_bal_areas = balancing_areas(
        pudl_engine,
        IPM_regions,
        all_gen,
        quickstart_res_load_frac=0.03,
        quickstart_res_wind_frac=0.05,
        quickstart_res_solar_frac=0.05,
        spinning_res_load_frac=".",
        spinning_res_wind_frac=".",
        spinning_res_solar_frac=".",
    )

    bal_areas

    # adding in the dummy loadzone for the fuel_cost / regional_fuel_market issue
    zone_bal_areas.loc[len(zone_bal_areas.index)] = ["loadzone", "BANC"]
    zone_bal_areas

    bal_areas.to_csv(out_folder / "balancing_areas.csv", index=False)
    zone_bal_areas.to_csv(out_folder / "zone_balancing_areas.csv", index=False)


def year_name(years):
    yrs = list(years)  # may be any iterable
    if len(yrs) > 1:
        return "foresight"
    else:
        return str(yrs[0])

    # fancy name, not used since we probably won't run different foresight
    # models with the same model name
    # if len(yrs) > 2:
    #     gap = yrs[1] - yrs[0]
    #     if all(y2 - y1 == gap for y1, y2 in zip(yrs[:-1], yrs[1:])):
    #         # multiple periods, evenly spaced, label as YYYA_YYYZ_NN
    #         # where NN is the gap between years
    #         return f"{yrs[0]}_{yrs[-1]}_{gap}"

    # # all other cases, label as YYYA_YYYB_...
    # return "_".join(str(y) for y in yrs)


def scenario_files(in_folder, out_folder, settings):
    """
    Create switch/scenarios*.txt, defining all the cases to run.
    """
    # get dataframe of all possible scenario input data
    scen_def_fn = in_folder / settings["scenario_definitions_fn"]
    scenario_definitions = pd.read_csv(scen_def_fn)

    # need some way to list all the crosses that we do automatically;
    # some of these are identified as PG scenarios, but we just do the cross
    # on one element of the data (trans limits); some of them are just done outside PG to
    # avoid creating too much data (carbon cost).


"""
# settings for testing
# settings_file = "MIP_results_comparison/case_settings/26-zone/settings"
settings_file = "MIP_results_comparison/case_settings/26-zone/settings-atb2023"
results_folder = "/tmp/pg_test"
# case_id = ["base_short"]
case_id = ["base_20_week"]
# year = [2030] # [2030, 2040, 2050]
year = []
myopic = False
"""


def main(
    settings_file: str,
    results_folder: str,
    # case_id: Annotated[Optional[List[str]], typer.Option()] = None,
    # year: Annotated[Optional[List[float]], typer.Option()] = None,
    case_id: List[str] = None,
    year: List[int] = None,
    myopic: bool = False,
):
    """Create inputs for the Switch model using PowerGenome data

    Example usage:

    $ python pg_to_switch.py settings results --case-id case1 --case-id case2 --year 2030

    Parameters
    ----------
    settings_file : str
        The path to a YAML file or folder of YAML files with settings parameters
    results_folder : str
        The folder where results will be saved
    case_id : Annotated[List[str], typer.Option, optional
        A list of case IDs, by default None. If using from CLI, provide a single case ID
        after each flag
    year : Annotated[List[int], typer.Option, optional
        A list of years, by default None. If using from CLI, provide a single year
        after each flag
    myopic : bool, optional
        A flag indicating whether to create model inputs in myopic mode (separate
        models for each study year) or as a single multi-year model (default).
        If only one year is chosen with the --year flag, this will have no effect.
    """
    cwd = Path.cwd()
    results_folder = cwd / results_folder
    results_folder.mkdir(parents=True, exist_ok=True)
    # Load settings, create db connections, and build dictionary of settings across
    # cases/years

    settings = load_settings(path=settings_file)
    pudl_engine, pudl_out, pg_engine = init_pudl_connection(
        freq="AS",
        start_year=min(settings.get("eia_data_years")),
        end_year=max(settings.get("eia_data_years")),
    )
    check_settings(settings, pg_engine)
    input_folder = cwd / settings["input_folder"]
    settings["input_folder"] = input_folder

    # note: beginning Feb. 2024, we no longer filter the settings dictionary
    # down to the specified years; instead we allow it to hold all available
    # data (possibly more than needed), and draw from it as needed for each
    # scenario.

    # get dataframe of scenario descriptions
    scen_def_fn = input_folder / settings["scenario_definitions_fn"]
    scenario_definitions = pd.read_csv(scen_def_fn)

    # filter scenarios to match requested case_id(s) and year(s) if any
    # note: typer gives an empty list (not None) if no options are specified
    if case_id:
        filter_cases = case_id
    else:
        filter_cases = scenario_definitions.case_id.unique()

    if year:
        filter_years = year
    else:
        filter_years = scenario_definitions.year.unique()

    scenario_definitions = scenario_definitions.loc[
        (
            scenario_definitions["case_id"].isin(filter_cases)
            & scenario_definitions["year"].isin(filter_years)
        ),
        :,
    ]

    if scenario_definitions.empty:
        if case_id or year:
            extra = " matching the requested case_id(s) or year(s)"
        else:
            extra = ""
        print(f"WARNING: No scenarios{extra} were found in {scen_def_fn}.\n")
    else:
        missing = set(case_id).difference(set(scenario_definitions["case_id"]))
        if missing:
            print(
                f"WARNING: requested case(s) {missing} were not found in {scen_def_fn}.\n"
            )
        missing = set(year).difference(set(scenario_definitions["year"]))
        if missing:
            print(
                f"WARNING: requested year(s) {missing} were not found in {scen_def_fn}.\n"
            )
        # if case_id and year and len(found) < len(filter_cases) * len(filter_years):
        #     print(f"Note that not every combination of case_id and year specified on the command line was defined in {scen_def_fn}.")

    # build scenario_settings dict, with scenario-specific values for each
    # scenario. scenario_settings will have keys for all available years, then
    # all available cases within that year (possibly mixed case-year
    # combinations)
    scenario_settings = build_scenario_settings(settings, scenario_definitions)

    # convert scenario_settings dict from year/case to case/year so we can easily
    # retrieve all years for each case later
    case_settings = {}
    for y, cases in scenario_settings.items():
        for c, case_year_settings in cases.items():
            case_settings.setdefault(c, {})[y] = case_year_settings

    if myopic:
        # run each case/year separately; split the settings for each year into
        # separate dicts and process them individually
        to_run = [
            (c, {y: year_settings})
            for c, scen_settings_dict in case_settings.items()
            for y, year_settings in scen_settings_dict.items()
        ]
    else:
        # run all years together within each case
        to_run = list(case_settings.items())

    print("\nPreparing models for the following cases:")
    for c, scen_settings_dict in to_run:
        all_years = scen_settings_dict.keys()
        print(f"{c}: {', '.join(str(y) for y in all_years)}")
    print()

    # load hydro_variability_new, and need to add variability for region 'MIS_D_MS'
    # by copying values from ' MIS_AR'
    hydro_var = pd.read_csv(input_folder / settings["hydro_variability_fn"])
    hydro_var["MIS_D_MS"] = hydro_var["MIS_AR"].values
    hydro_variability_new = hydro_var.copy()

    """
    # values for testing
    c, scen_settings_dict = to_run[0]
    """
    # Run through the different cases and save files in a new folder for each.
    for c, scen_settings_dict in to_run:
        # c is case_id for this case
        # scen_settings_dict has all settings for this case, organized by year
        all_years = scen_settings_dict.keys()
        print(f"\nstarting case {c} ({', '.join(str(y) for y in all_years)})")
        out_folder = results_folder / year_name(all_years) / c
        out_folder.mkdir(parents=True, exist_ok=True)

        first_year_settings = first_value(scen_settings_dict)
        final_year_settings = final_value(scen_settings_dict)

        # Retrieve gc for this case, using settings for first year so we get all
        # plants that survive up to that point (using last year would exclude
        # plants that retire during the study)
        # We set the multi_period flag based on the myopic flag; if multi_period
        # is set (foresight models), PowerGenome uses the same generators for all
        # periods (i.e., all generators in the database), which ensures that the
        # time clustering ends up the same for all periods. But then PowerGenome
        # also generates time-based data for generators even for periods when
        # they are not operational.
        gc = GeneratorClusters(
            pudl_engine,
            pudl_out,
            pg_engine,
            first_year_settings,
            multi_period=(not myopic),
        )

        # gc.fuel_prices already spans all years. We assume any added fuels show
        # up in the last year of the study. Then add_user_fuel_prices() adds them
        # to all years of the study (it only accepts one price for all years).
        all_fuel_prices = add_user_fuel_prices(final_year_settings, gc.fuel_prices)

        # generate Switch input tables from the PowerGenome settings/data
        generator_and_load_files(
            gc,
            all_fuel_prices,
            pudl_engine,
            scen_settings_dict,
            out_folder,
            pg_engine,
            hydro_variability_new,
        )
        fuel_files(
            fuel_prices=all_fuel_prices,
            planning_years=list(scen_settings_dict.keys()),
            regions=final_year_settings["model_regions"],
            fuel_region_map=final_year_settings["aeo_fuel_region_map"],
            fuel_emission_factors=final_year_settings["fuel_emission_factors"],
            out_folder=out_folder,
        )
        other_tables(
            scen_settings_dict=scen_settings_dict,
            out_folder=out_folder,
        )
        transmission_tables(
            scen_settings_dict,
            out_folder,
            pg_engine,
        )

    scenario_files(input_folder, results_folder, settings)


if __name__ == "__main__" and "ipykernel" not in sys.argv[0]:
    # running as a script, not from a jupyter environment
    typer.run(main)
