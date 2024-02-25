import os
import sys
import pandas as pd
import numpy as np
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
from powergenome.generators import GeneratorClusters
from powergenome.util import (
    build_scenario_settings,
    init_pudl_connection,
    load_settings,
    check_settings,
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
    derate_by_capacity_factor,
    switch_fuel_cost_table,
    switch_fuels,
    create_dict_plantgen,
    create_dict_plantpudl,
    plant_dict,
    plant_gen_id,
    plant_pudl_id,
    existing_gens_by_vintage,
    gen_build_costs_table,
    generation_projects_info,
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

    out_folder.mkdir(parents=True, exist_ok=True)
    first_year_settings = first_value(scen_settings_dict)

    all_gen = pg_generator_params(gc, first_year_settings)

    existing_gen, all_gen_units, existing_gen_units = existing_gen_info(
        gc, pudl_engine, first_year_settings, all_gen
    )

    new_gen_list, newgens = new_gen_info(gc, scen_settings_dict)

    ###########################################################
    # check how to deal with the plants below in other models #
    ###########################################################
    # remove plants registered/filed after the first year of model year
    # otherwise it causes problems in switch post solve

    # TODO: make sure filtering matches Switch's own filtering and don't
    # duplicate between here and conversion_functions.gen_build_costs_table()

    first_period_first_year = first_year_settings["model_first_planning_year"]
    existing_gen_units = existing_gen_units.loc[
        existing_gen_units["build_year"] <= first_period_first_year
    ]
    to_remove_cap = (
        existing_gen_units.loc[
            existing_gen_units["build_year"] > first_period_first_year
        ]
        .groupby("GENERATION_PROJECT", as_index=False)
        .agg(
            {
                "GENERATION_PROJECT": "first",
                "build_gen_predetermined": "sum",
            }
        )
    )
    existing_gen["reduce_capacity"] = (
        existing_gen["Resource"]
        .map(to_remove_cap.set_index("GENERATION_PROJECT")["build_gen_predetermined"])
        .fillna(0)
    )
    existing_gen["Existing_Cap_MW"] = (
        pd.to_numeric(existing_gen["Existing_Cap_MW"], errors="coerce")
        - existing_gen["reduce_capacity"]
    )
    existing_gen = existing_gen[existing_gen["Existing_Cap_MW"] > 0].drop(
        ["reduce_capacity"], axis=1
    )

    #########
    # create Switch input files from these tables

    # TODO: probably need to move this below the operation_model filtering
    gen_build_costs_file(first_year_settings, existing_gen_units, newgens, out_folder)

    # Create a complete list of existing and new-build options
    ## if running an operation model, remove all candidate projects.
    if first_year_settings.get("operation_model", False):
        newgens = pd.DataFrame()
    complete_gens = pd.concat([existing_gen, newgens]).drop_duplicates(
        subset=["Resource"]
    )
    complete_gens = add_misc_gen_values(complete_gens, first_year_settings)

    gen_info_file(
        all_fuel_prices,
        complete_gens,
        first_year_settings,
        out_folder,
        existing_gen_units,
    )

    balancing_tables(first_year_settings, pudl_engine, all_gen_units, out_folder)

    gen_build_predetermined_file(existing_gen_units, out_folder)

    time_varying_files(
        scen_settings_dict,
        pg_engine,
        hydro_variability_new,
        existing_gen,
        new_gen_list,
        out_folder,
    )


def time_varying_files(
    scen_settings_dict,
    pg_engine,
    hydro_variability_new,
    existing_gen,
    new_gen_list,
    out_folder,
):
    timepoint_start = 1
    output = collections.defaultdict(list)  # will hold all years of each type of

    timepoint_start = 1
    for year_settings, period_ng in zip(
        scen_settings_dict.values(),
        new_gen_list,
    ):
        ## if running an operation model, remove all candidate projects.
        if year_settings.get("operation_model") is True:
            period_ng = pd.DataFrame()

        period_all_gen = pd.concat([existing_gen, period_ng])
        period_all_gen_variability = make_generator_variability(period_all_gen)
        period_all_gen_variability.columns = period_all_gen["Resource"]
        if "gen_is_baseload" in period_all_gen.columns:
            period_all_gen_variability = set_must_run_generation(
                period_all_gen_variability,
                period_all_gen.loc[
                    period_all_gen["gen_is_baseload"] == True, "Resource"
                ].to_list(),
            )

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

        if cluster_time:
            assert "time_domain_periods" in year_settings
            assert "time_domain_days_per_period" in year_settings

            # results is a dict with keys "resource_profiles" (gen_variability), "load_profiles",
            # "time_series_mapping" (maps clusters sequentially to potential periods in year),
            # "ClusterWeights", etc. See PG for full details.
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
            period_lc = results["load_profiles"]
            period_variability = results["resource_profiles"]

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
                    period_lc,
                    year_settings["model_year"],
                    year_settings["model_first_planning_year"],
                    settings=year_settings,
                )
            else:
                timeseries_df, timepoints_df, timestamp_interval = timeseries(
                    period_lc,
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
                period_variability.loc[
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
                period_variability.loc[
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
            loads = load_pg_kmeans(period_lc, timepoints_df)
            timepoints_tp_id = timepoints_df[
                "timepoint_id"
            ].to_list()  # timepoint_id list
            dummy_df = pd.DataFrame({"TIMEPOINT": timepoints_tp_id})
            dummy_df.insert(0, "LOAD_ZONE", "loadzone")
            dummy_df.insert(2, "zone_demand_mw", 0)
            loads = loads.append(dummy_df)
        else:
            loads, loads_with_year_hour = loads_table(
                period_lc,
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
                period_all_gen, period_variability, timepoints_df
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

    # Write to CSV files
    for file, dfs in output.items():
        pd.concat(dfs).to_csv(out_folder / file, index=False)


def gen_build_costs_file(first_year_settings, existing_gen_units, newgens, out_folder):
    # TODO: maybe move gen_build_costs_table code into this function
    gen_build_costs = gen_build_costs_table(
        first_year_settings, existing_gen_units, newgens
    )
    gen_build_costs.to_csv(out_folder / "gen_build_costs.csv", index=False)


def gen_build_predetermined_file(existing_gen_units, out_folder):
    # write the relevant columns out for Switch
    gen_build_predetermined_cols = [
        "GENERATION_PROJECT",
        "build_year",
        "build_gen_predetermined",
        "build_gen_energy_predetermined",
    ]

    existing_gen_units[gen_build_predetermined_cols].to_csv(
        out_folder / "gen_build_predetermined.csv", index=False
    )


def gen_info_file(
    fuel_prices: pd.DataFrame,
    complete_gens: pd.DataFrame,
    settings: dict,
    out_folder: Path,
    gen_buildpre: pd.DataFrame,
):
    if settings.get("cogen_tech"):
        cogen_tech = settings["cogen_tech"]
    else:
        cogen_tech = {
            "Onshore Wind Turbine": False,
            "Biomass": False,
            "Conventional Hydroelectric": False,
            "Conventional Steam Coal": False,
            "Natural Gas Fired Combined Cycle": False,
            "Natural Gas Fired Combustion Turbine": False,
            "Natural Gas Steam Turbine": False,
            "Nuclear": False,
            "Solar Photovoltaic": False,
            "Hydroelectric Pumped Storage": False,
            "Offshore Wind Turbine": False,
            "OffShoreWind_Class1_Moderate_fixed_1": False,
            "Landbased Wind Turbine": False,
            "Small Hydroelectric": False,
            "NaturalGas_CCCCSAvgCF_Conservative": False,
            "NaturalGas_CCAvgCF_Moderate": False,
            "NaturalGas_CTAvgCF_Moderate": False,
            "Battery_*_Moderate": False,
            "NaturalGas_CCS100_Moderate": False,
            "heat_load_shifting": False,
            "UtilityPV_Class1_Moderate": False,
            "UtilityPV_Class1_Moderate_100": False,
        }
    if settings.get("baseload_tech"):
        baseload_tech = settings.get("baseload_tech")
    else:
        baseload_tech = {
            "Onshore Wind Turbine": False,
            "Biomass": False,
            "Conventional Hydroelectric": False,
            "Conventional Steam Coal": True,
            "Natural Gas Fired Combined Cycle": False,
            "Natural Gas Fired Combustion Turbine": False,
            "Natural Gas Steam Turbine": False,
            "Nuclear": True,
            "Solar Photovoltaic": False,
            "Hydroelectric Pumped Storage": False,
            "Offshore Wind Turbine": False,
            "OffShoreWind_Class1_Moderate_fixed_1": False,
            "Landbased Wind Turbine": False,
            "Small Hydroelectric": False,
            "NaturalGas_CCCCSAvgCF_Conservative": False,
            "NaturalGas_CCAvgCF_Moderate": False,
            "NaturalGas_CTAvgCF_Moderate": False,
            "Battery_*_Moderate": False,
            "NaturalGas_CCS100_Moderate": False,
            "heat_load_shifting": False,
            "UtilityPV_Class1_Moderate": False,
            "UtilityPV_Class1_Moderate_100": False,
        }
    if settings.get("energy_tech"):
        energy_tech = settings["energy_tech"]
    else:
        energy_tech = {
            "Onshore Wind Turbine": "Wind",
            "Biomass": "Bio Solid",
            "Conventional Hydroelectric": "Water",
            "Conventional Steam Coal": "Coal",
            "Natural Gas Fired Combined Cycle": "Naturalgas",
            "Natural Gas Fired Combustion Turbine": "Naturalgas",
            "Natural Gas Steam Turbine": "Naturalgas",
            "Nuclear": "Uranium",
            "Solar Photovoltaic": "Solar",
            "Hydroelectric Pumped Storage": "Water",
            "Offshore Wind Turbine": "Wind",
            "OffShoreWind_Class1_Moderate_fixed_1": "Wind",
            "Landbased Wind Turbine": "Wind",  ## add by RR because run into an erro of KeyError: 'LandbasedWind_Class1_Moderate_'
            "LandbasedWind_Class1_Moderate": "Wind",  ## add by RR because run into an erro of KeyError: 'LandbasedWind_Class1_Moderate_'
            "landbasedwind_class3_moderate": "Wind",  ## add by RR
            "Small Hydroelectric": "Water",
            "NaturalGas_CCCCSAvgCF_Conservative": "Naturalgas",
            "NaturalGas_CCAvgCF_Moderate": "Naturalgas",
            "NaturalGas_CTAvgCF_Moderate": "Naturalgas",
            "Battery_*_Moderate": "Electricity",
            "NaturalGas_CCS100_Moderate": "Naturalgas",
            "heat_load_shifting": False,
            "UtilityPV_Class1_Moderate": "Solar",
            "UtilityPV_Class1_Moderate_100": "Solar",
        }
    if settings.get("forced_outage_tech"):
        forced_outage_tech = settings["forced_outage_tech"]
    else:
        forced_outage_tech = {
            "Onshore Wind Turbine": 0.0,
            "Biomass": 0.04,
            "Conventional Hydroelectric": 0.05,
            "Conventional Steam Coal": 0.04,
            # "Natural Gas Fired Combined Cycle": 0.4,
            # "Natural Gas Fired Combustion Turbine": 0.4,
            # "Natural Gas Steam Turbine": 0.4,
            "Natural Gas Fired Combined Cycle": 0.04,
            "Natural Gas Fired Combustion Turbine": 0.04,
            "Natural Gas Steam Turbine": 0.04,
            "Nuclear": 0.04,
            "Solar Photovoltaic": 0.0,
            "Hydroelectric Pumped Storage": 0.05,
            "Offshore Wind Turbine": 0.05,
            "OffShoreWind_Class1_Moderate_fixed_1": 0.05,
            "Landbased Wind Turbine": 0.05,
            "Small Hydroelectric": 0.05,
            # "NaturalGas_CCCCSAvgCF_Conservative": 0.4,
            # "NaturalGas_CCAvgCF_Moderate": 0.4,
            # "NaturalGas_CTAvgCF_Moderate": 0.4,
            # "NaturalGas_CCS100_Moderate": 0.4,
            "NaturalGas_CCCCSAvgCF_Conservative": 0.04,
            "NaturalGas_CCAvgCF_Moderate": 0.04,
            "NaturalGas_CTAvgCF_Moderate": 0.04,
            "NaturalGas_CCS100_Moderate": 0.04,
            "Battery_*_Moderate": 0.02,
            "heat_load_shifting": False,
            "UtilityPV_Class1_Moderate": 0.0,
            "UtilityPV_Class1_Moderate_100": 0.0,
        }
    if settings.get("sched_outage_tech"):
        sched_outage_tech = settings["sched_outage_tech"]
    else:
        sched_outage_tech = {
            "Onshore Wind Turbine": 0.0,
            "Biomass": 0.06,
            "Conventional Hydroelectric": 0.05,
            "Conventional Steam Coal": 0.06,
            # "Natural Gas Fired Combined Cycle": 0.6,
            # "Natural Gas Fired Combustion Turbine": 0.6,
            # "Natural Gas Steam Turbine": 0.6,
            "Natural Gas Fired Combined Cycle": 0.06,
            "Natural Gas Fired Combustion Turbine": 0.06,
            "Natural Gas Steam Turbine": 0.06,
            "Nuclear": 0.06,
            "Solar Photovoltaic": 0.0,
            "Hydroelectric Pumped Storage": 0.05,
            "Offshore Wind Turbine": 0.01,
            "OffShoreWind_Class1_Moderate_fixed_1": 0.01,
            "Landbased Wind Turbine": 0.01,
            "Small Hydroelectric": 0.05,
            # "NaturalGas_CCCCSAvgCF_Conservative": 0.6,
            # "NaturalGas_CCAvgCF_Moderate": 0.6,
            # "NaturalGas_CTAvgCF_Moderate": 0.6,
            # "NaturalGas_CCS100_Moderate": 0.6,
            "NaturalGas_CCCCSAvgCF_Conservative": 0.06,
            "NaturalGas_CCAvgCF_Moderate": 0.06,
            "NaturalGas_CTAvgCF_Moderate": 0.06,
            "NaturalGas_CCS100_Moderate": 0.06,
            "Battery_*_Moderate": 0.01,
            "heat_load_shifting": False,
            "UtilityPV_Class1_Moderate": 0.0,
            "UtilityPV_Class1_Moderate_100": 0.0,
        }

    gen_info = generation_projects_info(
        complete_gens,
        settings.get("transmission_investment_cost")["spur"]["capex_mw_mile"],
        settings.get("retirement_ages"),
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

    gen_type_tech = {
        "Onshore Wind Turbine": "Wind",
        "Biomass": "Biomass",
        "Conventional Hydroelectric": "Hydro",
        "Conventional Steam Coal": "Coal",
        "Natural Gas Fired Combined Cycle": "Naturalgas",
        "Natural Gas Fired Combustion Turbine": "Naturalgas",
        "Natural Gas Steam Turbine": "Naturalgas",
        "Nuclear": "Nuclear",
        "Solar Photovoltaic": "Solar",
        "Hydroelectric Pumped Storage": "Hydro",
        "Offshore Wind Turbine": "Wind",
        "OffShoreWind_Class1_Moderate_fixed_1": "Wind",
        "Landbased Wind Turbine": "Wind",  ## add by RR because run into an erro of KeyError: 'LandbasedWind_Class1_Moderate_'
        "LandbasedWind_Class1_Moderate": "Wind",
        "Small Hydroelectric": "Hydro",
        "NaturalGas_CCCCSAvgCF_Conservative": "Naturalgas",
        "NaturalGas_CCAvgCF_Moderate": "Naturalgas",
        "NaturalGas_CTAvgCF_Moderate": "Naturalgas",
        "Battery_*_Moderate": "Storage",
        "NaturalGas_CCS100_Moderate": "Naturalgas",
        "UtilityPV_Class1_Moderate": "Solar",
        "UtilityPV_Class1_Moderate_100": "Solar",
    }

    graph_tech_types_table = gen_info.drop_duplicates(subset="gen_tech")
    graph_tech_types_table["map_name"] = "default"
    graph_tech_types_table["energy_source"] = graph_tech_types_table[
        "gen_energy_source"
    ]

    cols = ["map_name", "gen_type", "gen_tech", "energy_source"]
    graph_tech_types_table = graph_tech_types_table[cols]

    fuels = fuel_prices["fuel"].unique()
    fuels = [fuel.capitalize() for fuel in fuels]
    non_fuel_table = graph_tech_types_table[
        ~graph_tech_types_table["energy_source"].isin(fuels)
    ]
    non_fuel_energy = list(set(non_fuel_table["energy_source"].to_list()))
    non_fuel_energy_table = pd.DataFrame(non_fuel_energy, columns=["energy_source"])

    # non_fuel_energy_table = pd.DataFrame(non_fuel_energy, columns=['energy_source'])

    gen_info.loc[
        gen_info["gen_energy_source"].isin(non_fuel_energy),
        "gen_full_load_heat_rate",
    ] = "."

    graph_tech_colors_table.to_csv(out_folder / "graph_tech_colors.csv", index=False)
    graph_tech_types_table.to_csv(out_folder / "graph_tech_types.csv", index=False)
    non_fuel_energy_table.to_csv(
        out_folder / "non_fuel_energy_sources.csv", index=False
    )
    # change the gen_capacity_limit_mw for those from gen_build_predetermined
    gen_info_new = pd.merge(gen_info, gen_buildpre, how="left", on="GENERATION_PROJECT")
    gen_info_new.loc[
        gen_info_new["build_gen_predetermined"].notna(), "gen_capacity_limit_mw"
    ] = gen_info_new[["gen_capacity_limit_mw", "build_gen_predetermined"]].max(axis=1)
    gen_info = gen_info_new.drop(
        ["build_year", "build_gen_predetermined", "build_gen_energy_predetermined"],
        axis=1,
    )
    # remove the duplicated GENERATION_PROJECT from generation_projects_info .csv, and aggregate the "gen_capacity_limit_mw"
    gen_info["total_capacity"] = gen_info.groupby(["GENERATION_PROJECT"])[
        "gen_capacity_limit_mw"
    ].transform("sum")
    gen_info = gen_info.drop(
        ["gen_capacity_limit_mw"],
        axis=1,
    )
    gen_info.rename(columns={"total_capacity": "gen_capacity_limit_mw"}, inplace=True)
    gen_info = gen_info.drop_duplicates(subset="GENERATION_PROJECT")

    # identify generators participating in ESR or minimum capacity programs,
    # then drop those columns
    ESR_col = [col for col in gen_info.columns if col.startswith("ESR")]
    ESR_generators = gen_info[["GENERATION_PROJECT"] + ESR_col]
    min_cap_col = [col for col in gen_info.columns if col.startswith("MinCapTag")]
    min_cap_gens = gen_info[["GENERATION_PROJECT"] + min_cap_col]
    gen_info = gen_info.drop(columns=ESR_col + min_cap_col)

    # SWITCH 2.0.7 changes file name from  "generation_projects_info.csv" to "gen_info.csv"
    gen_info.to_csv(out_folder / "gen_info.csv", index=False)

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


def pg_generator_params(gc, first_year_settings):
    all_gen = gc.create_all_generators()
    all_gen = add_misc_gen_values(all_gen, first_year_settings)
    all_gen = hydro_energy_to_power(
        all_gen,
        first_year_settings.get("hydro_factor"),
        first_year_settings.get("regional_hydro_factor", {}),
    )
    all_gen["Resource"] = all_gen["Resource"].str.rstrip("_")
    all_gen["technology"] = all_gen["technology"].str.rstrip("_")
    # all_gen["plant_id_eia"] = all_gen["plant_id_eia"].astype("Int64")
    # existing_gen = all_gen.loc[
    #     all_gen["plant_id_eia"].notna(), :
    # ]  # gc.create_region_technology_clusters()

    ##### add for greenfield scenario, edit at 08/07/2023
    if "Existing_Cap_MW" in all_gen.columns:
        print("Existing_Cap_MW column exists")
    else:
        all_gen["Existing_Cap_MW"] = 0
    return all_gen


def existing_gen_info(gc, pudl_engine, first_year_settings, all_gen):
    existing_gen = all_gen.loc[all_gen["Existing_Cap_MW"] > 0, :]
    data_years = gc.settings.get("eia_data_years", [])
    if not isinstance(data_years, list):
        data_years = [data_years]
    data_years = [str(y) for y in data_years]
    s = f"""
        SELECT
            "plant_id_eia",
            "generator_id",
            "operational_status",
            "retirement_date",
            "planned_retirement_date",
            "current_planned_operating_date"
        FROM generators_eia860
        WHERE strftime('%Y',report_date) in ({','.join('?'*len(data_years))})
    """
    # generators_eia860 = pd.read_sql_table("generators_eia860", pudl_engine)
    generators_eia860 = pd.read_sql_query(
        s,
        pudl_engine,
        params=data_years,
        parse_dates=[
            "planned_retirement_date",
            "retirement_date",
            "current_planned_operating_date",
        ],
    )

    generators_entity_eia = pd.read_sql_table("generators_entity_eia", pudl_engine)
    # create copies of PUDL tables and filter to relevant columns
    pudl_gen = generators_eia860.copy()
    pudl_gen = pudl_gen[
        [
            "plant_id_eia",
            "generator_id",
            "operational_status",
            "retirement_date",
            "planned_retirement_date",
            "current_planned_operating_date",
        ]
    ]  #'utility_id_eia',

    pudl_gen_entity = generators_entity_eia.copy()
    pudl_gen_entity = pudl_gen_entity[
        ["plant_id_eia", "generator_id", "operating_date"]
    ]

    eia_Gen = gc.operating_860m
    eia_Gen = eia_Gen[
        [
            "utility_id_eia",
            "utility_name",
            "plant_id_eia",
            "plant_name",
            "generator_id",
            "operating_year",
            "planned_retirement_year",
        ]
    ]
    eia_Gen = eia_Gen.loc[eia_Gen["plant_id_eia"].notna(), :]

    # create identifier to connect to powergenome data
    eia_Gen["plant_gen_id"] = (
        eia_Gen["plant_id_eia"].astype(str) + "_" + eia_Gen["generator_id"]
    )

    eia_Gen_prop = gc.proposed_gens.reset_index()
    eia_Gen_prop = eia_Gen_prop[
        [
            # "utility_id_eia",
            # "utility_name",
            "plant_id_eia",
            # "plant_name",
            "generator_id",
            "planned_operating_year",
        ]
    ]
    eia_Gen_prop = eia_Gen_prop.loc[eia_Gen_prop["plant_id_eia"].notna(), :]
    eia_Gen_prop["plant_gen_id"] = (
        eia_Gen_prop["plant_id_eia"].astype(str) + "_" + eia_Gen_prop["generator_id"]
    )

    # create copies of potential_build_yr (powergenome)
    pg_build = gc.units_model.copy()
    if first_year_settings.get("derate_capacity"):
        pg_build = derate_by_capacity_factor(
            derate_techs=first_year_settings.get("derate_techs", []),
            unit_df=pg_build,
            existing_gen_df=existing_gen,
            cap_col=first_year_settings.get("capacity_col", "capacity_mw"),
        )
    pg_build = pg_build[
        [
            "plant_id_eia",
            "generator_id",
            "unit_id_pg",
            "planned_operating_year",
            "planned_retirement_date",
            "operating_date",
            "operating_year",
            "retirement_year",
            first_year_settings.get("capacity_col", "capacity_mw"),
            "capacity_mwh",
            "technology",
        ]
    ]

    retirement_ages = first_year_settings.get("retirement_ages")

    row_list = []
    for row in all_gen.itertuples():
        if isinstance(row.plant_id_eia, list):
            for plant_id, unit_id in zip(row.plant_id_eia, row.unit_id_pg):
                new_row = row._replace(plant_id_eia=plant_id, unit_id_pg=unit_id)
                row_list.append(new_row)
        else:
            row_list.append(row)
    all_gen_units = pd.DataFrame(row_list)
    all_gen_units["plant_id_eia"] = all_gen_units["plant_id_eia"].astype("Int64")

    # add in the plant+generator ids to pg_build and pudl tables (plant_id_eia + generator_id)
    pudl_gen = plant_gen_id(pudl_gen)
    pudl_gen_entity = plant_gen_id(pudl_gen_entity)
    pg_build = plant_gen_id(pg_build)

    # add in the plant+pudl id to the all_gen and pg_build tables (plant_id_eia + unit_pudl_id)
    pg_build = plant_pudl_id(pg_build)
    all_gen_units = plant_pudl_id(all_gen_units)

    # formerly gen_buildpre = gen_build_predetermined(),
    # but starting Feb. 2024, we carry cost info in this file too,
    # so it is not just for gen_build_predetermined.
    existing_gen_units = existing_gens_by_vintage(
        all_gen_units,
        pudl_gen,
        pudl_gen_entity,
        pg_build,
        {},  # manual_build_yr,
        eia_Gen,
        eia_Gen_prop,
        {},  # plant_gen_manual,
        {},  # plant_gen_manual_proposed,
        {},  # plant_gen_manual_retired,
        retirement_ages,
        first_year_settings.get("capacity_col", "capacity_mw"),
    )

    return existing_gen, all_gen_units, existing_gen_units


def new_gen_info(gc, scen_settings_dict):
    new_gen_list = []
    for year_settings in scen_settings_dict.values():
        # define new generators (period_ng)
        orig_gc_settings = gc.settings
        gc.settings = year_settings
        period_ng = gc.create_new_generators()
        gc.settings = orig_gc_settings

        period_ng["Resource"] = period_ng["Resource"].str.rstrip("_")
        period_ng["technology"] = period_ng["technology"].str.rstrip("_")
        period_ng["build_year"] = year_settings["model_year"]
        period_ng["GENERATION_PROJECT"] = period_ng["Resource"]
        if year_settings.get("co2_pipeline_filters") and year_settings.get(
            "co2_pipeline_cost_fn"
        ):
            period_ng = merge_co2_pipeline_costs(
                df=period_ng,
                co2_data_path=year_settings["input_folder"]
                / year_settings.get("co2_pipeline_cost_fn"),
                co2_pipeline_filters=year_settings["co2_pipeline_filters"],
                region_aggregations=year_settings.get("region_aggregations"),
                fuel_emission_factors=year_settings["fuel_emission_factors"],
                target_usd_year=year_settings.get("target_usd_year"),
            )

        new_gen_list.append(period_ng)

    newgens = pd.concat(new_gen_list, ignore_index=True)
    newgens = add_co2_costs_to_o_m(newgens)
    return new_gen_list, newgens


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
        file.write("2.0.7")


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
        reinforcement_cost = network_reinforcement_cost(
            transmission=transmission, settings=settings
        )
        max_reinforcement = network_max_reinforcement(
            transmission=transmission, settings=settings
        )
        transmission = agg_transmission_constraints(
            pg_engine=pg_engine, settings=settings
        )
        add_cap = add_cap_res_network(transmission, settings)

        tx_capex_mw_mile_dict = settings.get("transmission_investment_cost")["tx"][
            "capex_mw_mile"
        ]

        def region_avg(tx_capex_mw_mile_dict, region1, region2):
            r1_value = tx_capex_mw_mile_dict[region1]
            r2_value = tx_capex_mw_mile_dict[region2]
            r_avg = mean([r1_value, r2_value])
            return r_avg

        def create_transm_line_col(lz1, lz2, zone_dict):
            t_line = zone_dict[lz1] + "-" + zone_dict[lz2]
            return t_line

        transmission_lines = transmission_lines_table(
            line_loss, add_cap, tx_capex_mw_mile_dict, zone_dict, settings
        )
        transmission_lines

        trans_capital_cost_per_mw_km = (
            min(
                settings.get("transmission_investment_cost")["tx"][
                    "capex_mw_mile"
                ].values()
            )
            / 1.60934
        )
    else:
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

        transmission_lines = tx_cost_transform(transmission_lines)
        transmission_lines["tz2_dbid"] = transmission_lines["dest_region"].map(
            zone_dict
        )
        transmission_lines = transmission_lines.rename(
            columns={"start_region": "trans_lz1", "dest_region": "trans_lz2"}
        )
        transmission_lines["trans_dbid"] = range(1, len(transmission_lines) + 1)
        transmission_lines["trans_derating_factor"] = 0.95
        trans_capital_cost_per_mw_km = transmission_lines["cost_per_mw-km"].min()
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
    out_folder = cwd / results_folder
    out_folder.mkdir(parents=True, exist_ok=True)
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

    # Run through the different cases and save files in a new folder for each.
    for c, scen_settings_dict in to_run:
        # c is case_id for this case
        # scen_settings_dict has all settings for this case, organized by year
        all_years = scen_settings_dict.keys()
        print(f"\nstarting case {c} ({', '.join(str(y) for y in all_years)})")
        case_folder = out_folder / year_name(all_years) / c
        case_folder.mkdir(parents=True, exist_ok=True)

        first_year_settings = first_value(scen_settings_dict)
        final_year_settings = final_value(scen_settings_dict)

        # retrieve gc for this case, using settings for first year so we get all
        # plants that survive up to that point (using last year would exclude
        # plants that retire during the study)
        gc = GeneratorClusters(pudl_engine, pudl_out, pg_engine, first_year_settings)

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
            case_folder,
            pg_engine,
            hydro_variability_new,
        )
        fuel_files(
            fuel_prices=all_fuel_prices,
            planning_years=list(scen_settings_dict.keys()),
            regions=final_year_settings["model_regions"],
            fuel_region_map=final_year_settings["aeo_fuel_region_map"],
            fuel_emission_factors=final_year_settings["fuel_emission_factors"],
            out_folder=case_folder,
        )
        other_tables(
            scen_settings_dict=scen_settings_dict,
            out_folder=case_folder,
        )
        transmission_tables(
            scen_settings_dict,
            case_folder,
            pg_engine,
        )


if __name__ == "__main__":
    typer.run(main)
