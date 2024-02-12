import os
import pandas as pd

# Get annual revenue for every line, assuming they are compensated
# via the marginal-cost differential between the zones they serve multiplied
# by the amount of power they move.

# locations to read data from
# note: these could be set in a loop to produce results for multiple scenarios
base_dir = "."
group_in_dir = "inputs_conus_transmission/inputs_main"
group_out_dir = "outputs_conus_transmission/outputs_main"
scen_dir = "decarb_hydrogen/withall"

inputs_dir = os.path.join(base_dir, group_in_dir)
outputs_dir = os.path.join(base_dir, group_out_dir, scen_dir)

# utility functions for reading files
read_input = lambda f: pd.read_csv(os.path.join(inputs_dir, f), na_values=".")
read_output = lambda f: pd.read_csv(os.path.join(outputs_dir, f), na_values=".")

# TODO: change input pipeline so "23-04" is not converted to "23-Apr"
# in the TRANSMISSION_LINES set
trans_lines = read_output("transmission.csv")
# get capacity added in this period (good flag for whether we expect rents)
trans_lines = trans_lines.merge(
    read_output("BuildTx.csv").rename(
        {"TRANS_BLD_YRS_1": "TRANSMISSION_LINE", "TRANS_BLD_YRS_2": "PERIOD"}, axis=1
    )
)
# make a dict to lookup the official line name for any pair of connected zones
trans_line_lookup = {
    zone_pair: r["TRANSMISSION_LINE"]
    for i, r in trans_lines.iterrows()
    for zone_pair in [
        (r["trans_lz1"], r["trans_lz2"]),
        (r["trans_lz2"], r["trans_lz1"]),
    ]
}

# get hourly marginal costs in each zone, plus timepoints and weights
energy_sources = (
    read_output("energy_sources.csv").rename({"timepoint_label": "timestamp"}, axis=1)
    # TODO: eliminate this dummy load zone from the model input data
    .query('load_zone != "loadzone"')
    # get timepoint and timeseries based on timestamp
    .merge(
        read_input("timepoints.csv").rename({"timepoint_id": "timepoint"}, axis=1),
        how="outer",  # outer join so we'll see NaNs if some aren't matched
    )
    # get ts_scale_to_period
    .merge(
        read_input("timeseries.csv")[["timeseries", "ts_scale_to_period"]], how="outer"
    )
)

# transmission dispatch information
disp_trans = read_output("DispatchTx.csv").rename(
    {
        "TRANS_TIMEPOINTS_1": "zone_from",
        "TRANS_TIMEPOINTS_2": "zone_to",
        "TRANS_TIMEPOINTS_3": "timepoint",
    },
    axis=1,
)
# lookup TRANSMISSION_LINE for each from/to pair
disp_trans["TRANSMISSION_LINE"] = disp_trans.apply(
    lambda r: trans_line_lookup[r["zone_from"], r["zone_to"]], axis=1
)
# lookup efficiency
disp_trans = disp_trans.merge(
    trans_lines[["TRANSMISSION_LINE", "trans_efficiency"]], how="outer"
)

# get marginal cost of power at each end of each from/to pair during each timepoint;
# store this in mc_from or mc_to
for end in ["from", "to"]:
    disp_trans = disp_trans.merge(
        # this is a little tricky: we rename the energy_sources columns to match the
        # one we want to lookup (zone_from or zone_to) and the column we want to store
        # the marginal cost in (mc_from or mc_to) before doing the merge. Then the
        # merge automatically matches the right column and stores the result in the
        # right column
        energy_sources[
            ["load_zone", "timepoint", "marginal_cost", "ts_scale_to_period"]
        ].rename({"load_zone": f"zone_{end}", "marginal_cost": f"mc_{end}"}, axis=1),
        how="outer",
    )

# Calculate annual revenue contributed in each timepoint for each from/to pair.
# This includes the weights, so the simple sum across all timepoints will be
# correctly weighted total for the year.
disp_trans["annual_revenue"] = disp_trans.eval(
    # revenue from selling power on "to" end = DispatchTX * trans_efficiency * mc_to
    # cost of buying power on "from" end     = DispatchTX * mc_from
    "DispatchTx * (trans_efficiency * mc_to - mc_from) * ts_scale_to_period / 10"
)

# save total revenue to trans_lines
trans_lines = trans_lines.merge(
    disp_trans.groupby("TRANSMISSION_LINE")["annual_revenue"].sum().reset_index()
)

try:
    # give a nicer display for dataframes if possible
    from IPython.display import display
except ImportError:
    display = print

cols = [
    "TRANSMISSION_LINE",
    "TxCapacityNameplate",
    "BuildTx",
    "annual_revenue",
    "TotalAnnualCost",
]
# find lines with positive rents (shouldn't be any, since they should all be
# expanded until they are breaking even)
print("\nlines with positive rents (underbuilt):")
display(trans_lines.query("annual_revenue > TotalAnnualCost + 0.001")[cols])

print("\nlines with negative rents (overbuilt):")
display(trans_lines.query("annual_revenue < TotalAnnualCost - 0.001")[cols])

print("\nlines at breakeven level:")
display(
    trans_lines.query(
        "TotalAnnualCost + 0.001 >= annual_revenue >= TotalAnnualCost - 0.001"
    )[cols]
)

trans_lines["rents"] = trans_lines["annual_revenue"] - trans_lines["TotalAnnualCost"]
trans_lines.to_csv(
    os.path.join("/Users/rangrang/Desktop", "tx_rent_test.csv"), index=False
)
