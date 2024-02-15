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
)
from powergenome.GenX import (
    add_misc_gen_values,
    hydro_energy_to_power,
    add_co2_costs_to_o_m,
    create_policy_req,
    set_must_run_generation,
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
    gen_build_predetermined,
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


def gen_projects_info_file(
    fuel_prices: pd.DataFrame,
    # gc: GeneratorClusters,
    # pudl_engine: sa.engine,
    # settings_list: List[dict],
    # settings_file: str,
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

    gen_project_info = generation_projects_info(
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

    graph_tech_types_table = gen_project_info.drop_duplicates(subset="gen_tech")
    graph_tech_types_table["map_name"] = "default"
    graph_tech_types_table["energy_source"] = graph_tech_types_table[
        "gen_energy_source"
    ]

    cols = ["map_name", "gen_type", "gen_tech", "energy_source"]
    graph_tech_types_table = graph_tech_types_table[cols]

    # settings = load_settings(path=settings_file)
    # pudl_engine, pudl_out, pg_engine = init_pudl_connection(
    #     freq="AS",
    #     start_year=min(settings.get("data_years")),
    #     end_year=max(settings.get("data_years")),
    # )
    # gc = GeneratorClusters(pudl_engine, pudl_out, pg_engine, settings_list[0])
    # fuel_prices = gc.fuel_prices
    fuels = fuel_prices["fuel"].unique()
    fuels = [fuel.capitalize() for fuel in fuels]
    non_fuel_table = graph_tech_types_table[
        ~graph_tech_types_table["energy_source"].isin(fuels)
    ]
    non_fuel_energy = list(set(non_fuel_table["energy_source"].to_list()))
    non_fuel_energy_table = pd.DataFrame(non_fuel_energy, columns=["energy_source"])

    # non_fuel_energy_table = pd.DataFrame(non_fuel_energy, columns=['energy_source'])

    gen_project_info.loc[
        gen_project_info["gen_energy_source"].isin(non_fuel_energy),
        "gen_full_load_heat_rate",
    ] = "."

    graph_tech_colors_table.to_csv(out_folder / "graph_tech_colors.csv", index=False)
    graph_tech_types_table.to_csv(out_folder / "graph_tech_types.csv", index=False)
    non_fuel_energy_table.to_csv(
        out_folder / "non_fuel_energy_sources.csv", index=False
    )
    # change the gen_capacity_limit_mw for those from gen_build_predetermined
    gen_project_info_new = pd.merge(
        gen_project_info, gen_buildpre, how="left", on="GENERATION_PROJECT"
    )
    gen_project_info_new.loc[
        gen_project_info_new["gen_predetermined_cap"].notna(), "gen_capacity_limit_mw"
    ] = gen_project_info_new[["gen_capacity_limit_mw", "gen_predetermined_cap"]].max(
        axis=1
    )
    gen_project_info = gen_project_info_new.drop(
        ["build_year", "gen_predetermined_cap", "gen_predetermined_storage_energy_mwh"],
        axis=1,
    )
    # remove the duplicated GENERATION_PROJECT from generation_projects_info .csv, and aggregate the "gen_capacity_limit_mw"
    gen_project_info["total_capacity"] = gen_project_info.groupby(
        ["GENERATION_PROJECT"]
    )["gen_capacity_limit_mw"].transform("sum")
    gen_project_info = gen_project_info.drop(
        ["gen_capacity_limit_mw"],
        axis=1,
    )
    gen_project_info.rename(
        columns={"total_capacity": "gen_capacity_limit_mw"}, inplace=True
    )
    gen_project_info = gen_project_info.drop_duplicates(subset="GENERATION_PROJECT")
    # SWITCH 2.0.7 changes file name from  "generation_projects_info.csv" to "gen_info.csv"
    gen_project_info.to_csv(out_folder / "gen_info.csv", index=False)


def gen_prebuild_newbuild_info_files(
    gc: GeneratorClusters,
    all_fuel_prices,
    pudl_engine: sa.engine,
    settings_list: List[dict],
    case_years: List,
    out_folder: Path,
    pg_engine: sa.engine,
    hydro_variability_new: pd.DataFrame,
):
    out_folder.mkdir(parents=True, exist_ok=True)
    settings = settings_list[0]
    all_gen = gc.create_all_generators()
    all_gen = add_misc_gen_values(all_gen, settings)
    all_gen = hydro_energy_to_power(
        all_gen,
        settings.get("hydro_factor"),
        settings.get("regional_hydro_factor", {}),
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
    if settings.get("derate_capacity"):
        pg_build = derate_by_capacity_factor(
            derate_techs=settings.get("derate_techs", []),
            unit_df=pg_build,
            existing_gen_df=existing_gen,
            cap_col=settings.get("capacity_col", "capacity_mw"),
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
            settings.get("capacity_col", "capacity_mw"),
            "capacity_mwh",
            "technology",
        ]
    ]

    retirement_ages = settings.get("retirement_ages")

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

    gen_buildpre, gen_build_with_id = gen_build_predetermined(
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
        settings.get("capacity_col", "capacity_mw"),
    )

    retired = gen_build_with_id.loc[
        gen_build_with_id["retirement_year"] < settings["model_year"], :
    ]
    retired_ids = retired["GENERATION_PROJECT"].to_list()

    # newbuild options
    periods_dict = {
        "new_gen": [],
        "load_curves": [],
    }
    planning_periods = []
    planning_period_start_yrs = []
    for settings in settings_list:
        gc.settings = settings
        period_ng = gc.create_new_generators()
        period_ng["Resource"] = period_ng["Resource"].str.rstrip("_")
        period_ng["technology"] = period_ng["technology"].str.rstrip("_")
        period_ng["build_year"] = settings["model_year"]
        period_ng["GENERATION_PROJECT"] = period_ng[
            "Resource"
        ]  # + f"_{settings['model_year']}"
        if settings.get("co2_pipeline_filters") and settings.get(
            "co2_pipeline_cost_fn"
        ):
            period_ng = merge_co2_pipeline_costs(
                df=period_ng,
                co2_data_path=settings["input_folder"]
                / settings.get("co2_pipeline_cost_fn"),
                co2_pipeline_filters=settings["co2_pipeline_filters"],
                region_aggregations=settings.get("region_aggregations"),
                fuel_emission_factors=settings["fuel_emission_factors"],
                target_usd_year=settings.get("target_usd_year"),
            )

        periods_dict["new_gen"].append(period_ng)

        period_lc = make_final_load_curves(pg_engine, settings)
        periods_dict["load_curves"].append(period_lc)
        planning_periods.append(settings["model_year"])
        planning_period_start_yrs.append(settings["model_first_planning_year"])

    newgens = pd.concat(periods_dict["new_gen"], ignore_index=True)
    newgens = add_co2_costs_to_o_m(newgens)

    build_yr_list = gen_build_with_id["build_year"].to_list()
    # using gen_build_with_id because it has plants that were removed for the final gen_build_pred. (ie. build year=2020)
    gen_project = gen_build_with_id["GENERATION_PROJECT"].to_list()
    build_yr_plantid_dict = dict(zip(gen_project, build_yr_list))

    ###########################################################
    # check how to deal with the plants below in other models #
    ###########################################################
    # remove plants registered/filed after the first year of model year
    # otherwise it causes problems in switch post solve
    gen_buildpre = gen_buildpre.loc[
        gen_buildpre["build_year"] <= planning_period_start_yrs[0]
    ]
    to_remove_cap = (
        gen_buildpre.loc[gen_buildpre["build_year"] > planning_period_start_yrs[0]]
        .groupby("GENERATION_PROJECT", as_index=False)
        .agg(
            {
                "GENERATION_PROJECT": "first",
                "gen_predetermined_cap": "sum",
            }
        )
    )
    existing_gen["reduce_capacity"] = (
        existing_gen["Resource"]
        .map(to_remove_cap.set_index("GENERATION_PROJECT")["gen_predetermined_cap"])
        .fillna(0)
    )
    existing_gen["Existing_Cap_MW"] = (
        pd.to_numeric(existing_gen["Existing_Cap_MW"], errors="coerce")
        - existing_gen["reduce_capacity"]
    )
    existing_gen = existing_gen[existing_gen["Existing_Cap_MW"] > 0].drop(
        ["reduce_capacity"], axis=1
    )

    gen_build_costs = gen_build_costs_table(settings, gen_buildpre, newgens)
    # Create a complete list of existing and new-build options
    ## if running an operation model, remove all candidate projects.
    if settings.get("operation_model") is True:
        newgens = pd.DataFrame()
    complete_gens = pd.concat([existing_gen, newgens]).drop_duplicates(
        subset=["Resource"]
    )
    complete_gens = add_misc_gen_values(complete_gens, settings)

    gen_projects_info_file(
        all_fuel_prices, complete_gens, gc.settings, out_folder, gen_buildpre
    )

    ts_list = []
    tp_list = []
    hts_list = []
    htp_list = []
    load_list = []
    vcf_list = []
    gts_map_list = []
    water_nodes_list = []
    water_connections_list = []
    reservoirs_list = []
    hydro_pj_list = []
    water_node_tp_flows_list = []
    timepoint_start = 1
    for settings, period_lc, period_ng in zip(
        settings_list, periods_dict["load_curves"], periods_dict["new_gen"]
    ):
        ## if running an operation model, remove all candidate projects.
        if settings.get("operation_model") is True:
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

        if settings.get("reduce_time_domain") is True:
            for p in ["time_domain_periods", "time_domain_days_per_period"]:
                assert p in settings.keys()

            # results is a dict with keys "resource_profiles" (gen_variability), "load_profiles",
            # "time_series_mapping" (maps clusters sequentially to potential periods in year),
            # "ClusterWeights", etc. See PG for full details.
            results, representative_point, weights = kmeans_time_clustering(
                resource_profiles=period_all_gen_variability,
                load_profiles=period_lc,
                days_in_group=settings["time_domain_days_per_period"],
                num_clusters=settings["time_domain_periods"],
                include_peak_day=settings.get("include_peak_day", True),
                load_weight=settings.get("demand_weight_factor", 1),
                variable_resources_only=settings.get("variable_resources_only", True),
            )

            period_lc = results["load_profiles"]
            period_variability = results["resource_profiles"]

            timeseries_df, timepoints_df = ts_tp_pg_kmeans(
                representative_point["slot"],
                weights,
                settings["time_domain_days_per_period"],
                settings["model_year"],
                settings["model_first_planning_year"],
            )
            timepoints_df["timepoint_id"] = range(
                timepoint_start, timepoint_start + len(timepoints_df)
            )
            timepoint_start = timepoints_df["timepoint_id"].max() + 1
            hydro_timepoints_df = hydro_timepoints_pg_kmeans(timepoints_df)
            hydro_timeseries_table = hydro_timeseries_pg_kmeans(
                period_all_gen,
                period_variability.loc[
                    :, period_all_gen.loc[period_all_gen["HYDRO"] == 1, "Resource"]
                ],
                hydro_timepoints_df,
            )
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
            vcf = variable_cf_pg_kmeans(
                period_all_gen, period_variability, timepoints_df
            )

            loads = load_pg_kmeans(period_lc, timepoints_df)
            timepoints_tp_id = timepoints_df[
                "timepoint_id"
            ].to_list()  # timepoint_id list
            dummy_df = pd.DataFrame({"TIMEPOINT": timepoints_tp_id})
            dummy_df.insert(0, "LOAD_ZONE", "loadzone")
            dummy_df.insert(2, "zone_demand_mw", 0)
            loads = loads.append(dummy_df)
            graph_timestamp_map = graph_timestamp_map_kmeans(timepoints_df)
        else:
            if settings.get("full_time_domain") is True:
                timeseries_df, timepoints_df, timestamp_interval = timeseries_full(
                    period_lc,
                    settings["model_year"],
                    settings["model_first_planning_year"],
                    settings=settings,
                )
            else:
                timeseries_df, timepoints_df, timestamp_interval = timeseries(
                    period_lc,
                    settings["model_year"],
                    settings["model_first_planning_year"],
                    settings=settings,
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
            hydro_timepoints_df, hydro_timeseries_table = hydro_time_tables(
                period_all_gen,
                period_all_gen_variability,
                timepoints_df,
                settings["model_year"],
            )
            hydro_timepoints_df

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
            loads, loads_with_year_hour = loads_table(
                period_lc, timepoints_timestamp, timepoints_dict, settings["model_year"]
            )
            # for fuel_cost and regional_fuel_market issue
            dummy_df = pd.DataFrame({"TIMEPOINT": timepoints_tp_id})
            dummy_df.insert(0, "LOAD_ZONE", "loadzone")
            dummy_df.insert(2, "zone_demand_mw", 0)
            loads = loads.append(dummy_df)

            year_hour = loads_with_year_hour["year_hour"].to_list()

            vcf = variable_capacity_factors_table(
                period_all_gen_variability,
                year_hour,
                timepoints_dict,
                period_all_gen,
                settings["model_year"],
            )

            graph_timestamp_map = graph_timestamp_map_table(
                timeseries_df, timestamp_interval
            )

        gts_map_list.append(graph_timestamp_map)
        ts_list.append(timeseries_df)
        tp_list.append(timepoints_df)
        htp_list.append(hydro_timepoints_df)
        hts_list.append(hydro_timeseries_table)
        load_list.append(loads)
        vcf_list.append(vcf)
        water_nodes_list.append(water_nodes)
        water_connections_list.append(water_connections)
        reservoirs_list.append(reservoirs)
        hydro_pj_list.append(hydro_pj)
        water_node_tp_flows_list.append(water_node_tp_flows)

    if gts_map_list:
        graph_timestamp_map = pd.concat(gts_map_list, ignore_index=True)
        graph_timestamp_map.to_csv(out_folder / "graph_timestamp_map.csv", index=False)

    timeseries_df = pd.concat(ts_list, ignore_index=True)
    timepoints_df = pd.concat(tp_list, ignore_index=True)
    hydro_timepoints_df = pd.concat(htp_list, ignore_index=True)
    hydro_timeseries_table = pd.concat(hts_list, ignore_index=True)
    loads = pd.concat(load_list, ignore_index=True)
    vcf = pd.concat(vcf_list, ignore_index=True)

    water_nodes_df = pd.concat(water_nodes_list, ignore_index=True)
    water_connections_df = pd.concat(water_connections_list, ignore_index=True)
    reservoirs_df = pd.concat(reservoirs_list, ignore_index=True)
    hydro_pj_df = pd.concat(hydro_pj_list, ignore_index=True)
    water_node_tp_flows_df = pd.concat(water_node_tp_flows_list, ignore_index=True)

    water_nodes_df.to_csv(out_folder / "water_nodes.csv", index=False)
    water_connections_df.to_csv(out_folder / "water_connections.csv", index=False)
    reservoirs_df.to_csv(out_folder / "reservoirs.csv", index=False)
    hydro_pj_df.to_csv(out_folder / "hydro_generation_projects.csv", index=False)
    water_node_tp_flows_df.to_csv(out_folder / "water_node_tp_flows.csv", index=False)

    timeseries_df.to_csv(out_folder / "timeseries.csv", index=False)
    timepoints_df.to_csv(out_folder / "timepoints.csv", index=False)
    hydro_timepoints_df.to_csv(out_folder / "hydro_timepoints.csv", index=False)

    balancing_tables(settings, pudl_engine, all_gen_units, out_folder)
    hydro_timeseries_table.to_csv(out_folder / "hydro_timeseries.csv", index=False)
    loads.to_csv(out_folder / "loads.csv", index=False)
    vcf.to_csv(out_folder / "variable_capacity_factors.csv", index=False)

    # SWITCH 2.0.7 changes column names "gen_predetermined_cap", "gen_predetermined_storage_energy_mwh" to
    #  "build_gen_predetermined", "build_gen_energy_predetermined" in gen_build_predetermined.csv.
    gen_buildpre = gen_buildpre.rename(
        columns={
            "gen_predetermined_cap": "build_gen_predetermined",
            "gen_predetermined_storage_energy_mwh": "build_gen_energy_predetermined",
        }
    )
    gen_buildpre.to_csv(out_folder / "gen_build_predetermined.csv", index=False)
    gen_build_costs.to_csv(out_folder / "gen_build_costs.csv", index=False)


### edit by RR


def other_tables(
    settings, period_start_list, period_end_list, atb_data_year, out_folder
):
    if settings.get("emission_policies_fn"):
        model_year = settings["model_year"]
        for i in model_year:
            # energy_share_req = create_policy_req(_settings, col_str_match="ESR")
            co2_cap = create_policy_req(settings, col_str_match="CO_2")
            df = {
                "period": [i],
                "carbon_cap_tco2_per_yr": [
                    co2_cap["CO_2_Max_Mtons_1"].sum() * 1000000
                ],  # Mton to ton, the unit in PG is Mton; column name need to be updated.
                "carbon_cap_tco2_per_yr_CA": [
                    "."
                ],  # change this value if the CA policy module is included.
                "carbon_cost_dollar_per_tco2": [
                    "."
                ],  # change this value if you would like to look at the social cost instead og carbon cap.
            }
    else:  # Based on REAM
        df = {
            # "period": [2030, 2040, 2050],
            # "carbon_cap_tco2_per_yr": [149423302.5, 76328672.3, 0],
            # "carbon_cap_tco2_per_yr_CA": [36292500, 11400000, 0],
            # "carbon_cost_dollar_per_tco2": [".", ".", "."],
            "period": [2050],
            "carbon_cap_tco2_per_yr": [0],
            "carbon_cap_tco2_per_yr_CA": [0],
            "carbon_cost_dollar_per_tco2": ["."],
        }

    carbon_policies_table = pd.DataFrame(df)

    carbon_policies_table

    # interest and discount based on REAM
    financials_data = {
        "base_financial_year": atb_data_year,
        "interest_rate": 0.05,
        "discount_rate": 0.05,
    }
    financials_table = pd.DataFrame(financials_data, index=[0])
    financials_table

    # based on REAM
    periods_data = {
        # "INVESTMENT_PERIOD": [2020, 2030, 2040, 2050],
        # "period_start": [2016, 2026, 2036, 2046],
        # "period_end": [2025, 2035, 2045, 2055],
        "INVESTMENT_PERIOD": period_end_list,
        "period_start": period_start_list,
        "period_end": period_end_list,
    }
    periods_table = pd.DataFrame(periods_data)
    periods_table
    # write switch version txt file
    file = open(out_folder / "switch_inputs_version.txt", "w")
    file.write("2.0.7")
    file.close()

    carbon_policies_table.to_csv(out_folder / "carbon_policies.csv", index=False)
    financials_table.to_csv(out_folder / "financials.csv", index=False)
    periods_table.to_csv(out_folder / "periods.csv", index=False)


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


def transmission_tables(settings, out_folder, pg_engine):
    """
    pulling in information from PowerGenome transmission notebook
    Schivley Greg, PowerGenome, (2022), GitHub repository,
        https://github.com/PowerGenome/PowerGenome/blob/master/notebooks/Transmission.ipynb
    """
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

    transmission_lines.to_csv(out_folder / "transmission_lines.csv", index=False)
    trans_params_table.to_csv(out_folder / "trans_params.csv", index=False)


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


def main(
    settings_file: str,
    results_folder: str,
    # case_id: Annotated[Optional[List[str]], typer.Option()] = None,
    # year: Annotated[Optional[List[float]], typer.Option()] = None,
    case_id: List[str] = None,
    year: List[float] = None,
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
    year : Annotated[List[float], typer.Option, optional
        A list of years, by default None. If using from CLI, provide a single year
        after each flag
    """
    cwd = Path.cwd()
    out_folder = cwd / results_folder
    out_folder.mkdir(parents=True, exist_ok=True)
    # Load settings, create db connections, and build dictionary of settings across
    # cases/years

    settings = load_settings(path=settings_file)
    if year is None:
        year = settings["model_year"]
    else:
        remove_model_years = []
        remove_start_years = []
        for model_year, start_year in zip(
            settings["model_year"], settings["model_first_planning_year"]
        ):
            if model_year not in year:
                # Create lists of years to remove.
                remove_model_years.append(model_year)
                remove_start_years.append(start_year)

        settings["model_year"] = [
            y for y in settings["model_year"] if y not in remove_model_years
        ]
        settings["model_first_planning_year"] = [
            y
            for y in settings["model_first_planning_year"]
            if y not in remove_start_years
        ]
    pudl_engine, pudl_out, pg_engine = init_pudl_connection(
        freq="AS",
        start_year=min(settings.get("eia_data_years")),
        end_year=max(settings.get("eia_data_years")),
    )
    check_settings(settings, pg_engine)
    input_folder = cwd / settings["input_folder"]
    settings["input_folder"] = input_folder
    scenario_definitions = pd.read_csv(
        input_folder / settings["scenario_definitions_fn"]
    )
    if case_id is None:
        case_id = scenario_definitions.case_id.unique()
    else:
        scenario_definitions = scenario_definitions.loc[
            scenario_definitions["case_id"].isin(case_id), :
        ]
    scenario_settings = build_scenario_settings(settings, scenario_definitions)

    # load hydro_variability_new, and need to add varibality for region 'MIS_D_MS'
    # by copying values from ' MIS_AR'
    hydro_var = pd.read_csv(input_folder / settings["hydro_variability_fn"])
    hydro_var["MIS_D_MS"] = hydro_var["MIS_AR"].values
    hydro_variability_new = hydro_var.copy()

    # Should switch the case_id/year layers in scenario settings dictionary.
    # Run through the different cases and save files in a new folder for each.
    for case_id in scenario_definitions["case_id"].unique():
        print(f"starting case {case_id}")
        case_folder = out_folder / case_id
        case_folder.mkdir(parents=True, exist_ok=True)

        settings[
            "case_id"
        ] = case_id  # add by Rangrang to make 'GenX.create_policy_req' work for this script.

        settings_list = []
        case_years = []
        case_start_years = []
        for year in settings["model_year"]:
            case_years.append(scenario_settings[year][case_id]["model_year"])
            settings_list.append(scenario_settings[year][case_id])
            case_start_years.append(
                scenario_settings[year][case_id]["model_first_planning_year"]
            )

        gc = GeneratorClusters(pudl_engine, pudl_out, pg_engine, settings_list[0])
        all_fuel_prices = add_user_fuel_prices(
            scenario_settings[year][case_id], gc.fuel_prices
        )
        gen_prebuild_newbuild_info_files(
            gc,
            all_fuel_prices,
            pudl_engine,
            settings_list,
            case_years,
            case_folder,
            pg_engine,
            hydro_variability_new,
        )
        fuel_files(
            fuel_prices=all_fuel_prices,
            planning_years=case_years,
            regions=settings["model_regions"],
            fuel_region_map=settings["aeo_fuel_region_map"],
            fuel_emission_factors=settings["fuel_emission_factors"],
            out_folder=case_folder,
        )
        # other_tables(atb_data_year=settings["atb_data_year"], out_folder=case_folder)
        other_tables(
            settings=settings,
            period_start_list=case_start_years,
            period_end_list=case_years,
            atb_data_year=settings["atb_data_year"],
            out_folder=case_folder,
        )
        transmission_tables(
            settings,
            # settings_list[0],
            case_folder,
            pg_engine,
        )


if __name__ == "__main__":
    typer.run(main)
