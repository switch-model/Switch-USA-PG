"""
This module imposes limits on additions on each transmission path during each
period. The limit is expressed in MW and stored in
trans_path_expansion_limit.csv.

This module defines the following parameters and expression:

`trans_expansion_limit_mw_km[p in PERIODS]`: Can be provided by the user in
"trans_expansion_limit.csv". Setting this to zero will prevent any new
transmission; setting it to a non-zero value limits total transmission expansion
in period `p` to the specified amount, in MW * km (MW of transfer capability *
km length of corridor). If this column is omitted or a value is missing ("."),
it will be treated as an infinite limit.

`trans_path_expansion_limit_mw[tx in TRANSMISSION_LINES, p in PERIODS]`: Can be
provided by the user in "trans_path_expansion_limit.csv". Expressed in MW for
each corridor. Setting this to zero will prevent any new transmission; setting
it to 400 would allow new construction along path `tx` in period `p` of 400 MW.
If this column is omitted or a value is missing ("."), it will be treated as an
infinite limit.
"""

import os
from pyomo.environ import *


def define_components(m):
    # Maximum amount of transmission that can be added per period on each path
    # in each period, expressed in MW.
    m.trans_path_expansion_limit_mw = Param(
        m.TRANSMISSION_LINES, m.PERIODS, within=NonNegativeReals, default=float("inf")
    )

    # Enforce trans_path_expansion_limit_mw if provided
    m.Limit_Transmission_Path_Expansion_MW = Constraint(
        m.TRANSMISSION_LINES,
        m.PERIODS,
        rule=lambda m, tx, p: (
            Constraint.Skip
            if m.trans_path_expansion_limit_mw[tx, p] == float("inf")
            else (m.BuildTx[tx, p] <= m.trans_path_expansion_limit_mw[tx, p])
        ),
    )


def load_inputs(m, switch_data, inputs_dir):
    """
    Import the cap(s) on new transmission in each period.

    trans_path_expansion_limit.csv
        TRANSMISSION_LINE, PERIOD, trans_path_expansion_limit_mw
    """

    switch_data.load_aug(
        filename=os.path.join(inputs_dir, "trans_path_expansion_limit.csv"),
        optional=True,
        param=(m.trans_path_expansion_limit_mw,),
    )
