"""
This module requires minimizing pollution exposure subject to a limit on total
expenditure.

This module defines the following parameters and expressions:

"""

import os
from pyomo.environ import *
from switch_model.utilities import unique_list


def define_arguments(argparser):
    argparser.add_argument(
        "--no-minimize-pollution-exposure",
        action="store_true",
        default=False,
        help="Don't minimize pollution exposure across groups; "
        "just minimize costs and report results. You may also want to set "
        "`--save-expression GroupExposure` for benchmarking.",
    )


def define_components(m):
    # Exposure to pollution for each group per MWh produced by each generator
    # in each period

    # Indexing set for the exporure data: (generator, period, group) combination
    # (These are all the index columns from exposure_coefficients.csv.)
    m.GEN_PERIOD_EXP_GROUPS = Set(
        dimen=3,  # within=m.GENERATION_PROJECTS * m.PERIODS * Any
        within=Any * m.PERIODS * Any,
    )

    # exposure coefficient for each (generator, period, group) combination
    m.group_exposure_coefficient = Param(
        m.GEN_PERIOD_EXP_GROUPS, default=0, within=NonNegativeReals
    )

    # list of all groups that are ever exposed to pollution
    m.EXPOSURE_GROUPS = Set(
        initialize=lambda m: unique_list(
            grp for (gen, p, grp) in m.GEN_PERIOD_EXP_GROUPS
        )
    )

    m.GEN_PROJECTS_FOR_EXPOSURE_GROUP_IN_PERIOD = Set(
        m.EXPOSURE_GROUPS,
        m.PERIODS,
        initialize=lambda m, grp, p: [
            _gen
            for (_gen, _p, _grp) in m.GEN_PERIOD_EXP_GROUPS
            if (_grp, _p) == (grp, p)
        ],
    )

    # total exposure for each group in each period
    m.GroupExposure = Expression(
        m.EXPOSURE_GROUPS,
        m.PERIODS,
        rule=lambda m, grp, p: sum(
            m.DispatchGen[gen, tp]
            * m.group_exposure_coefficient[gen, p, grp]
            * m.tp_weight_in_year[tp]
            for gen in m.GEN_PROJECTS_FOR_EXPOSURE_GROUP_IN_PERIOD[grp, p]
            if gen in m.GENERATION_PROJECTS
            for tp in m.TPS_FOR_GEN_IN_PERIOD[gen, p]
            if (gen, p, grp) in m.GEN_PERIOD_EXP_GROUPS
            and m.group_exposure_coefficient[gen, p, grp] != 0
        ),
    )

    # cap on total exposure of each group in each period
    m.MaxGroupExposure = Var(m.PERIODS)

    # respect the cap
    m.Enforce_MaxGroupExposure = Constraint(
        m.EXPOSURE_GROUPS,
        m.PERIODS,
        rule=lambda m, grp, p: m.GroupExposure[grp, p] <= m.MaxGroupExposure[p],
    )


def define_dynamic_components(m):
    if not m.options.no_minimize_pollution_exposure:

        # budget limit
        # directory from which to read total_cost.txt from previous model
        m.system_cost_baseline_dir = Param(within=Any, default="")
        # amount to scale previous cost by to get the budget limit
        m.system_cost_multiplier = Param(within=Reals, default=1)
        # amount to add to the previous cost to get the budget limit
        m.system_cost_adder = Param(within=Reals, default=0)

        def init(m):
            if m.system_cost_multiplier == 0:
                baseline_cost = 0
            else:
                with open(
                    # have to wrap string parameters in value() for some reason
                    os.path.join(value(m.system_cost_baseline_dir), "total_cost.txt")
                ) as f:
                    baseline_cost = float(f.read().strip())
            return baseline_cost * m.system_cost_multiplier + m.system_cost_adder

        m.system_cost_limit = Param(initialize=init)

        # enforce budget limit (on NPV basis) (not known until define_dynamic_components)
        m.Enforce_System_Cost_Limit = Constraint(
            rule=lambda m: m.SystemCost <= m.system_cost_limit
        )

        # minimize the cap (duration-weighted sum of peak exposure across periods)
        m.Mini_Max_Group_Exposure = Objective(
            rule=lambda m: sum(
                m.MaxGroupExposure[p] * m.period_length_years[p] for p in m.PERIODS
            )
        )

        # turn off the standard objective
        m.Minimize_System_Cost.deactivate()


def load_inputs(m, switch_data, inputs_dir):
    """
    Import the budget cap and exposure data.

    system_cost_limit.csv
        system_cost_baseline_dir, system_cost_multiplier, system_cost_adder

    group_exposure_coefficients.csv
        GENERATION_PROJECT, PERIOD, EXPOSURE_GROUP, group_exposure_coefficient
    """

    if not m.options.no_minimize_pollution_exposure:
        switch_data.load_aug(
            filename=os.path.join(inputs_dir, "system_cost_limit.csv"),
            param=(
                m.system_cost_baseline_dir,
                m.system_cost_multiplier,
                m.system_cost_adder,
            ),
        )

    switch_data.load_aug(
        filename=os.path.join(inputs_dir, "group_exposure_coefficients.csv"),
        index=m.GEN_PERIOD_EXP_GROUPS,
        param=(m.group_exposure_coefficient,),
    )
