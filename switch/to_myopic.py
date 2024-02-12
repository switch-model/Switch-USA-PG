import pandas as pd
import os
import numpy as np
import subprocess


################################### FOR 26 ZONE ###################################
################################### make  resource_capacity.csv
in_folder = "/Users/rangrang/Library/CloudStorage/GoogleDrive-zhengr@hawaii.edu/My Drive/SWITCH_INPUTS_rr_east"
# year_list = [2030, 2040, 2050]
# case_list = [
#     "base_short_50",
#     "base_short_200",
#     "base_short_1000",
#     "base_short_no_ccs",
#     "base_short_retire",
#     "base_short_commit",
# ]

year_list = ["foresight"]
case_list = ["base_short"]
for i in case_list:
    tocompare_folder = os.path.join("tocompare/26z", i)
    resource_capacity_agg = pd.DataFrame()
    for y in year_list:
        # add the retirement back to the resource list
        ###### ADD capacity
        prebuild2030 = pd.read_csv(
            os.path.join(
                in_folder,
                "26z_results/inputs",
                str(y),
                i,
                "gen_build_predetermined.csv",
            )
        )
        # prebuild2050 = pd.read_csv(os.path.join(in_folder, "inputs_myopic2050/gen_build_predetermined.csv"))
        build2030 = pd.read_csv(
            os.path.join(
                in_folder, "26z_results/outputs", i, "out" + str(y), "BuildGen.csv"
            )
        )

        merge2030 = prebuild2030.merge(
            build2030,
            left_on=["GENERATION_PROJECT", "build_year"],
            right_on=["GEN_BLD_YRS_1", "GEN_BLD_YRS_2"],
            how="outer",
            indicator=True,
        )
        retired_2030 = pd.DataFrame(merge2030[merge2030["_merge"] == "left_only"])

        df = retired_2030
        df["GEN_BLD_YRS_1"] = df["GENERATION_PROJECT"]
        df["GEN_BLD_YRS_2"] = df["build_year"]
        df["BuildGen"] = df["build_gen_predetermined"]
        df = df[["GEN_BLD_YRS_1", "GEN_BLD_YRS_2", "BuildGen"]]
        build_gen = pd.concat([build2030, df])
        build_gen["end_value"] = build_gen["BuildGen"]
        ###### ADD energy capacity
        energy_build2030 = pd.read_csv(
            os.path.join(
                in_folder,
                "26z_results/outputs",
                i,
                "out" + str(y),
                "BuildStorageEnergy.csv",
            )
        )
        df_energy = energy_build2030
        df_energy = df_energy.rename(
            {
                "STORAGE_GEN_BLD_YRS_1": "GEN_BLD_YRS_1",
                "STORAGE_GEN_BLD_YRS_2": "GEN_BLD_YRS_2",
                "BuildStorageEnergy": "MWh",
            },
            axis=1,
        )
        resource = build_gen.merge(
            df_energy,
            on=["GEN_BLD_YRS_1", "GEN_BLD_YRS_2"],
            how="left",
            # indicator=True,
        )
        ###############################
        df = resource

        if y == "foresight":
            df2030 = df.loc[df["GEN_BLD_YRS_2"] <= 2030]
            df2030["start_value"] = np.where(
                df2030["GEN_BLD_YRS_2"] < 2030, df2030["BuildGen"], 0
            )
            df2030["start_MWh"] = np.where(
                df2030["GEN_BLD_YRS_2"] < 2030, df2030["MWh"], 0
            )
            df2030["planning_year"] = 2030

            df2040 = df.loc[df["GEN_BLD_YRS_2"] <= 2040]
            df2040["start_value"] = np.where(
                df2040["GEN_BLD_YRS_2"] < 2040, df2040["BuildGen"], 0
            )
            df2040["start_MWh"] = np.where(
                df2040["GEN_BLD_YRS_2"] < 2040, df2040["MWh"], 0
            )
            df2040["planning_year"] = 2040

            df2050 = df.loc[df["GEN_BLD_YRS_2"] <= 2050]
            df2050["start_value"] = np.where(
                df2050["GEN_BLD_YRS_2"] < 2050, df2050["BuildGen"], 0
            )
            df2050["start_MWh"] = np.where(
                df2050["GEN_BLD_YRS_2"] < 2050, df2050["MWh"], 0
            )
            df2050["planning_year"] = 2050

            df = pd.concat([df2030, df2040, df2050])
        else:
            df["start_value"] = np.where(df["GEN_BLD_YRS_2"] < y, df["BuildGen"], 0)
            df["start_MWh"] = np.where(df["GEN_BLD_YRS_2"] < y, df["MWh"], 0)
            df["planning_year"] = y

        df["start_MWh"][df["MWh"] == "NaN"] = "NaN"
        df["end_MWh"] = df["MWh"]
        resource_capacity = pd.DataFrame()

        resource_capacity["resource_name"] = df["GEN_BLD_YRS_1"]
        resource_capacity["zone"] = [x[0] for x in df["GEN_BLD_YRS_1"].str.split("_")]
        resource_capacity["tech_type"] = "Other"
        resource_capacity.loc[
            resource_capacity["resource_name"].str.contains("hydrogen", case=False),
            "tech_type",
        ] = "Hydrogen"
        resource_capacity.loc[
            resource_capacity["resource_name"].str.contains("batter", case=False),
            "tech_type",
        ] = "Battery"
        resource_capacity.loc[
            resource_capacity["resource_name"].str.contains("coal", case=False),
            "tech_type",
        ] = "Coal"
        resource_capacity.loc[
            resource_capacity["resource_name"].str.contains("solar|pv", case=False),
            "tech_type",
        ] = "Solar"
        resource_capacity.loc[
            resource_capacity["resource_name"].str.contains("wind", case=False),
            "tech_type",
        ] = "Wind"
        resource_capacity.loc[
            resource_capacity["resource_name"].str.contains("hydro|water", case=False),
            "tech_type",
        ] = "Hydro"
        resource_capacity.loc[
            resource_capacity["resource_name"].str.contains("distribute", case=False),
            "tech_type",
        ] = "Distributed Solar"
        resource_capacity.loc[
            resource_capacity["resource_name"].str.contains("geothermal", case=False),
            "tech_type",
        ] = "Geothermal"
        resource_capacity.loc[
            resource_capacity["resource_name"].str.contains("nuclear", case=False),
            "tech_type",
        ] = "Nuclear"
        resource_capacity.loc[
            resource_capacity["resource_name"].str.contains("natural", case=False),
            "tech_type",
        ] = "Natural Gas"
        resource_capacity.loc[
            resource_capacity["resource_name"].str.contains("ccs", case=False),
            "tech_type",
        ] = "CCS"
        resource_capacity["model"] = "SWITCH"

        resource_capacity["planning_year"] = df["planning_year"]
        resource_capacity["case"] = i
        resource_capacity["unit"] = "MW"
        resource_capacity["start_value"] = df["start_value"]
        resource_capacity["end_value"] = df["end_value"]
        resource_capacity["start_MWh"] = df["start_MWh"]
        resource_capacity["end_MWh"] = df["end_MWh"]

        resource_capacity_agg = pd.concat([resource_capacity_agg, resource_capacity])

    resource_capacity_agg = resource_capacity_agg.loc[
        resource_capacity_agg["end_value"] > 0
    ]
    resource_capacity_agg.to_csv(
        os.path.join(in_folder, tocompare_folder, "resource_capacity.csv"),
        index=False,
    )


####################################### make  transmission.csv

for i in case_list:
    tocompare_folder = os.path.join("tocompare/26z", i)
    tx_agg = pd.DataFrame()
    for y in year_list:
        transmission2030_new = pd.read_csv(
            os.path.join(
                in_folder, "26z_results/outputs", i, "out" + str(y), "transmission.csv"
            )
        )

        # find the existing transmission capacity
        transmission2030_ex = pd.read_csv(
            os.path.join(
                in_folder, "26z_results/inputs", str(y), i, "transmission_lines.csv"
            )
        )

        transmission2030 = transmission2030_new.merge(transmission2030_ex, how="left")

        df = transmission2030.copy()
        df["model"] = "SWITCH"
        df["line_name"] = df["trans_lz1"] + "_to_" + df["trans_lz2"]
        df["planning_year"] = df["PERIOD"]
        df["case"] = i
        df["unit"] = "MW"
        # for foreight scenario:
        # define a function to find the values of "TxCapacityNameplate" from last period
        # and fill it as the "start_value" of next period for same line.

        if y == "foresight":
            df.loc[df["PERIOD"] == 2030, "start_value"] = df["existing_trans_cap"]

            fill_dict2030 = (
                df.loc[df["PERIOD"] == 2030]
                .groupby("line_name")["TxCapacityNameplate"]
                .last()
                .to_dict()
            )
            df.loc[df["PERIOD"] == 2040, "start_value"] = df["line_name"].map(
                fill_dict2030
            )

            fill_dict2040 = (
                df.loc[df["PERIOD"] == 2040]
                .groupby("line_name")["TxCapacityNameplate"]
                .last()
                .to_dict()
            )
            df.loc[df["PERIOD"] == 2050, "start_value"] = df["line_name"].map(
                fill_dict2040
            )
        else:
            df["start_value"] = df["existing_trans_cap"]

        df["end_value"] = df["TxCapacityNameplate"]
        df = df[
            [
                "model",
                "line_name",
                "planning_year",
                "case",
                "unit",
                "start_value",
                "end_value",
            ]
        ]
        tx_agg = pd.concat([tx_agg, df])

    tx_agg.to_csv(
        os.path.join(in_folder, tocompare_folder, "transmission.csv"), index=False
    )
    df = tx_agg.copy()
    df["value"] = df["end_value"] - df["start_value"]
    df2 = df[["model", "line_name", "planning_year", "case", "unit", "value"]]
    df2.to_csv(
        os.path.join(in_folder, tocompare_folder, "transmission_expansion.csv"),
        index=False,
    )

################################### make  generation.csv

for i in case_list:
    tocompare_folder = os.path.join("tocompare/26z", i)
    generation_agg = pd.DataFrame()
    dispatch_agg = pd.DataFrame()
    for y in year_list:
        ts = pd.read_csv(
            os.path.join(in_folder, "26z_results/inputs", str(y), i, "timeseries.csv")
        )
        dispatch2030 = pd.read_csv(
            os.path.join(
                in_folder, "26z_results/outputs", i, "out" + str(y), "dispatch.csv"
            )
        )
        df = dispatch2030.copy()

        df["model"] = "SWITCH"
        df["zone"] = df["gen_load_zone"]
        df["resource_name"] = df["generation_project"]
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
        df.loc[
            df["resource_name"].str.contains("coal", case=False), "tech_type"
        ] = "Coal"
        df.loc[
            df["resource_name"].str.contains("solar|pv", case=False), "tech_type"
        ] = "Solar"
        df.loc[
            df["resource_name"].str.contains("wind", case=False), "tech_type"
        ] = "Wind"
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

        # df["planning_year"] = y
        df["case"] = i
        df["timestep"] = "all"
        df["unit"] = "MWh"
        dp = df.copy()
        dp["value"] = dp["DispatchGen_MW"]
        from transmission_scripts.TX_util import tp_to_date

        dp = tp_to_date(dp, "timestamp")
        dp["days"] = pd.to_numeric(dp["days"])
        from datetime import date

        def converttodate(dfrow):
            return date.fromisocalendar(dfrow["period"], dfrow["week"], dfrow["days"])

        dp["date"] = dp.apply(lambda dfrow: converttodate(dfrow), axis=1)
        dp["planning_year"] = dp["period"]
        dp = dp.sort_values(by=["week"])
        week_to_report = ts.iloc[0]["timeseries"][6:]
        dp = dp.loc[dp["week"] == int(week_to_report)]
        dp = dp[
            [
                "hour",
                "resource_name",
                "value",
                "zone",
                "planning_year",
                "model",
                "case",
                "week",
                "date",
            ]
        ]
        dp = dp.loc[dp["value"] > 0]
        df["value"] = df["DispatchGen_MW"] * df["tp_weight_in_year_hrs"]
        generation = df.groupby(["resource_name", "period"], as_index=False).agg(
            {
                "value": "sum",
                "model": "first",
                "zone": "first",
                "resource_name": "first",
                "tech_type": "first",
                "case": "first",
                "timestep": "first",
                "unit": "first",
            }
        )
        generation["planning_year"] = generation["period"]

        dispatch_agg = pd.concat([dispatch_agg, dp])
        generation_agg = pd.concat([generation_agg, generation])

    dispatch_agg.to_csv(
        os.path.join(in_folder, tocompare_folder, "dispatch.csv"), index=False
    )
    generation_agg.to_csv(
        os.path.join(in_folder, tocompare_folder, "generation.csv"), index=False
    )

    # generation2030 = generation2030.rename({"DispatchGen_MW": "value"}, axis=1)


################################### make  emission.csv

for i in case_list:
    tocompare_folder = os.path.join("tocompare/26z", i)
    emission_agg = pd.DataFrame()
    for y in year_list:
        emission2030 = pd.read_csv(
            os.path.join(
                in_folder, "26z_results/outputs", i, "out" + str(y), "dispatch.csv"
            )
        )
        df = emission2030.copy()

        df = df.groupby(["gen_load_zone", "period"], as_index=False).agg(
            {
                "DispatchEmissions_tCO2_per_typical_yr": "sum",
            }
        )
        df["planning_year"] = df["period"]
        df = df.rename(
            {"gen_load_zone": "zone", "DispatchEmissions_tCO2_per_typical_yr": "value"},
            axis=1,
        )
        df["model"] = "SWITCH"
        df["case"] = i
        df["unit"] = "kg"
        df["value"] = df["value"] * 1000  # from ton to kg
        df = df[["model", "zone", "planning_year", "case", "unit", "value"]]
        emission_agg = pd.concat([emission_agg, df])

    emission_agg.to_csv(
        os.path.join(in_folder, tocompare_folder, "emissions.csv"), index=False
    )
############################################################################
############################################################################
############################################################################
############################################################################
# find differences-- rhe predetermined generations are not the same for each period
pre2030 = pd.read_csv(
    "/Users/rangrang/Dropbox/PowerGenomeSWITCH_documentation_r1/Jupyter_Notebooks/national_emm_2030/unconstrained_thin/gen_build_predetermined.csv"
)
pre2040 = pd.read_csv(
    "/Users/rangrang/Dropbox/PowerGenomeSWITCH_documentation_r1/Jupyter_Notebooks/national_emm_2040/unconstrained_thin/gen_build_predetermined.csv"
)
pre_all = pre2030.merge(pre2040, on=["GENERATION_PROJECT"], how="left", indicator=True)
unmatch2030 = pre_all[pre_all["_merge"] == "left_only"]
unmatch2040 = pre_all[pre_all["_merge"] == "right_only"]
unmatch = unmatch2030[["GENERATION_PROJECT", "build_year_x", "build_year_y"]]
unmatch.to_csv(
    "/Users/rangrang/Dropbox/PowerGenomeSWITCH_documentation_r1/Jupyter_Notebooks/predetermined_diff.csv"
)
unmatch2040[["GENERATION_PROJECT", "build_year_x", "build_year_y"]]

############################################################################
############################################################################
# add the ones built at one period(2030) to next period(2040). below need to be editted
# gen_build_predetermined.csv, gen_build_costs.csv, transmission_lines.csv
in_folder = "/Users/rangrang/Library/CloudStorage/GoogleDrive-zhengr@hawaii.edu/My Drive/SWITCH_INPUTS_rr_east"
# USE THIS DIR_LIST IF U ARE ADDING OUTPUTS OF 2030 TO INPUTS 204O

period = 2040
next_period = 2050
scenario = "base_short_1000"

# utility functions for reading files
outputs_dir = os.path.join(
    in_folder, "26z_results/outputs", scenario, "out" + str(period)
)
from_inputs_dir = os.path.join(
    in_folder, "26z_results/inputs", str(next_period), "base_short_origin"
)
to_inputs_dir = os.path.join(
    in_folder, "26z_results/inputs", str(next_period), scenario
)

read_input = lambda f: pd.read_csv(os.path.join(from_inputs_dir, f))
read_output = lambda f: pd.read_csv(os.path.join(outputs_dir, f))

build_gen2030 = read_output("BuildGen.csv")
if period == 2040:
    new_gen2030 = build_gen2030[
        (build_gen2030["GEN_BLD_YRS_2"] > 2031) & (build_gen2030["BuildGen"] != 0)
    ]
if period == 2030:
    new_gen2030 = build_gen2030[
        (build_gen2030["GEN_BLD_YRS_2"] > 2026) & (build_gen2030["BuildGen"] != 0)
    ]
# check if there are any new built battery/storage
if new_gen2030["GEN_BLD_YRS_1"].str.contains("batter|storage").any():
    print("WARNING: There are new-built battery")
# check if there are any new built water/hydro
if new_gen2030["GEN_BLD_YRS_1"].str.contains("water|hydro").any():
    print("WARNING: There are new-built hydro")


# gen_build_predetermined.csv
gen_predetermined2040 = read_input("gen_build_predetermined.csv")
gen_predetermined2040 = gen_predetermined2040.rename(
    {
        "gen_predetermined_cap": "build_gen_predetermined",
        "gen_predetermined_storage_energy_mwh": "build_gen_energy_predetermined",
    },
    axis=1,
)
new_gen2030 = new_gen2030.rename(
    {
        "GEN_BLD_YRS_1": "GENERATION_PROJECT",
        "GEN_BLD_YRS_2": "build_year",
        "BuildGen": "build_gen_predetermined",
    },
    axis=1,
)
new_gen2030["build_gen_energy_predetermined"] = "."
# if no new-built battery, we only need to concat the new builds to predetermined
# if there is new built battery, need to add energy capacity to next period predetermined as well.
build_energy2030 = read_output("BuildStorageEnergy.csv")
new_gen2030 = pd.merge(
    new_gen2030,
    build_energy2030,
    left_on=["GENERATION_PROJECT", "build_year"],
    right_on=["STORAGE_GEN_BLD_YRS_1", "STORAGE_GEN_BLD_YRS_2"],
    how="left",
)
new_gen2030["build_gen_energy_predetermined"] = np.where(
    new_gen2030["BuildStorageEnergy"] != "NaN",
    new_gen2030["BuildStorageEnergy"],
    new_gen2030["build_gen_energy_predetermined"],
)
new_gen2030.build_gen_energy_predetermined = new_gen2030[
    "build_gen_energy_predetermined"
].fillna(".")
new_gen2030 = new_gen2030[
    [
        "GENERATION_PROJECT",
        "build_year",
        "build_gen_predetermined",
        "build_gen_energy_predetermined",
    ]
]

fixed_predetermined2040 = pd.concat([gen_predetermined2040, new_gen2030])
fixed_predetermined2040.build_gen_energy_predetermined = (
    fixed_predetermined2040.build_gen_energy_predetermined.fillna(".")
)
fixed_predetermined2040.to_csv(
    os.path.join(to_inputs_dir, "gen_build_predetermined.csv"), index=False
)

#########################################
# gen_build_costs.csv
gen_build_costs2040 = read_input("gen_build_costs.csv")
new_gen2030 = new_gen2030[["GENERATION_PROJECT", "build_year"]]
new_gen2030["gen_overnight_cost"] = 0
new_gen2030["gen_fixed_om"] = 0
new_gen2030["gen_storage_energy_overnight_cost"] = "."
new_gen2030.loc[
    new_gen2030["GENERATION_PROJECT"].str.contains("batter|storage", case=False),
    "gen_storage_energy_overnight_cost",
] = 0
fixed_gen_build_costs2040 = pd.concat([gen_build_costs2040, new_gen2030])
fixed_gen_build_costs2040.to_csv(
    os.path.join(to_inputs_dir, "gen_build_costs.csv"), index=False
)
#########################################
# transmission_lines.csv
transmission2040 = read_input("transmission_lines.csv")
outputs_tx2030 = read_output("transmission.csv")

fixed_tx2040 = transmission2040.merge(
    outputs_tx2030, on=("TRANSMISSION_LINE", "trans_lz1", "trans_lz2")
)
fixed_tx2040 = fixed_tx2040[
    [
        "TRANSMISSION_LINE",
        "trans_lz1",
        "trans_lz2",
        "trans_length_km_y",
        "trans_efficiency_y",
        "TxCapacityNameplate",
        "trans_dbid_y",
        "trans_derating_factor_y",
        "trans_terrain_multiplier",
        "trans_new_build_allowed",
    ]
]

fixed_tx2040 = fixed_tx2040.rename(
    {
        "trans_length_km_y": "trans_length_km",
        "trans_efficiency_y": "trans_efficiency",
        "TxCapacityNameplate": "existing_trans_cap",
        "trans_dbid_y": "trans_dbid",
        "trans_derating_factor_y": "trans_derating_factor",
    },
    axis=1,
)

fixed_tx2040.to_csv(os.path.join(to_inputs_dir, "transmission_lines.csv"), index=False)
