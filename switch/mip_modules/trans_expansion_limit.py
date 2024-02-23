"""
This module imposes a limit on total transmission additions during each period.
The limit is expressed as a fraction relative to the total amount of
transmission (MW * km) already existing in the system before the start of the
study. This module defines the following parameters and expression:

`trans_expansion_baseline_mw_km`: Total MW-km of transmission in the power
system at the start of the study. Calculated from transmission corridor lengths
and capacities given in transmission_lines.csv.

`TxTotalMWkmAdded[p in PERIODS]`: Expression showing total MW-km of transmission
added in period `p`, calculated as the sum of `BuildTx[tx, p] *
trans_length_km[tx]` for all corridors `Tx` in `TRANSMISSION_LINES`.

`trans_expansion_limit_fraction[p in PERIODS]`: Can be provided by the user in
"trans_expansion_limit.csv". Expressed as a fraction of existing transmission
(`trans_expansion_baseline_mw_km`, defined below). Setting this to zero will
prevent any new transmission; setting it to 1 would allow new construction in
period `p` of as much transmission as already exists at the start of the study.
The limit is set on the basis of total MW-km throughout the system, so it will
not limit transmission on individual corridors based on the quantity already in
place on that corridor. If this column is omitted or a value is missing ("."),
it will be treated as an infinite limit.

`trans_expansion_limit_mw_km[p in PERIODS]`: Can be provided by the user in
"trans_expansion_limit.csv". Setting this to zero will prevent any new
transmission; setting it to a non-zero value limits total transmission expansion
in period `p` to the specified amount, in MW * km (MW of transfer capability *
km length of corridor). If this column is omitted or a value is missing ("."),
it will be treated as an infinite limit.

`trans_expansion_limit_path_fraction[p in PERIODS]`: Can be provided by the user
in "trans_expansion_limit.csv". Expressed as a fraction of existing transmission
(MW) on each corridor. Setting this to zero will prevent any new transmission;
setting it to 1 would allow new construction along each path in period `p` of as
much transmission as already exists on that path at the start of the study. If
this column is omitted or a value is missing ("."), it will be treated as an
infinite limit.

If multiple limits are specified, all of them will be enforced, i.e., the lower
one(s) will take effect.

It can be helpful to run Switch with `--input-alias
trans_expansion_limit.csv=none --save-expression TxTotalMWkmAdded` flags to
report how much transmission would be built without a cap. This value will be
saved in <outputs-dir>/TxTotalMWkmAdded.csv. Then you can construct a
<inputs-dir>/trans_expansion_limit.csv file with the trans_expansion_limit_mw_km
column equal to some fraction of the baseline additions, e.g., 0.5 *
TxTotalMWkmAdded.

"""

import os
from pyomo.environ import *


def define_components(m):
    # Maximum amount of transmission that can be added per period, as a fraction
    # relative to the total capacity (MW-km) in place at the end of the prior
    # period
    m.trans_expansion_limit_fraction = Param(
        m.PERIODS, within=NonNegativeReals, default=float("inf")
    )

    # Maximum amount of transmission that can be added per period, expressed in
    # units of MW * km
    m.trans_expansion_limit_mw_km = Param(
        m.PERIODS, within=NonNegativeReals, default=float("inf")
    )

    # Maximum amount of transmission that can be added per period on each path,
    # as a fraction relative to the total capacity (MW) in place on that path at
    # the end of the prior period
    m.trans_expansion_limit_path_fraction = Param(
        m.PERIODS, within=NonNegativeReals, default=float("inf")
    )

    # Total MW-km of transmission in place at start
    m.trans_expansion_baseline_mw_km = Param(
        within=NonNegativeReals,
        rule=lambda m: sum(
            m.existing_trans_cap[tx] * m.trans_length_km[tx]
            for tx in m.TRANSMISSION_LINES
        ),
    )

    # Total MW-km of transmission added in each period
    m.TxTotalMWkmAdded = Expression(
        m.PERIODS,
        rule=lambda m, p: sum(
            m.BuildTx[tx, p] * m.trans_length_km[tx] for tx in m.TRANSMISSION_LINES
        ),
    )

    # Enforce trans_expansion_limit_fraction if provided
    def rule(m, p):
        if m.trans_expansion_limit_fraction[p] == float("inf"):
            return Constraint.Skip
        else:
            prev_cap = m.trans_expansion_baseline_mw_km + sum(
                m.TxTotalMWkmAdded[p]
                for _p in m.CURRENT_AND_PRIOR_PERIODS_FOR_PERIOD[p]
                if _p != p
            )
            return (
                m.TxTotalMWkmAdded[p] <= m.trans_expansion_limit_fraction[p] * prev_cap
            )

    m.Limit_Transmission_Expansion_Fraction = Constraint(
        m.PERIODS,
        rule=rule,
    )

    # Enforce trans_expansion_limit_mw_km if provided
    m.Limit_Transmission_Expansion_MWkm = Constraint(
        m.PERIODS,
        rule=lambda m, p: (
            Constraint.Skip
            if m.trans_expansion_limit_mw_km[p] == float("inf")
            else (m.TxTotalMWkmAdded[p] <= m.trans_expansion_limit_mw_km[p])
        ),
    )

    # Enforce trans_expansion_limit_path_fraction if provided
    def rule(m, p, tx):
        if m.trans_expansion_limit_path_fraction[p] == float("inf"):
            return Constraint.Skip
        else:
            if p == m.PERIODS.first():
                prev_cap = m.existing_trans_cap[tx]
            else:
                prev_cap = m.TxCapacityNameplate[tx, m.PERIODS.prev(p)]
            return (
                m.BuildTx[tx, p] <= m.trans_expansion_limit_path_fraction[p] * prev_cap
            )

    m.Limit_Transmission_Expansion_Path_Fraction = Constraint(
        m.PERIODS,
        m.TRANSMISSION_LINES,
        rule=rule,
    )


def load_inputs(m, switch_data, inputs_dir):
    """
    Import the cap(s) on new transmission in each period.

    trans_expansion_limit.csv
        PERIOD, trans_expansion_limit_fraction, trans_expansion_limit_mw_km, trans_expansion_limit_path_fraction
    """

    switch_data.load_aug(
        filename=os.path.join(inputs_dir, "trans_expansion_limit.csv"),
        optional=True,
        param=(
            m.trans_expansion_limit_fraction,
            m.trans_expansion_limit_mw_km,
            m.trans_expansion_limit_path_fraction,
        ),
    )
