import os
from pyomo.environ import *
from switch_model.financials import capital_recovery_factor as crf

"""
Calculate capital recovery for all generation projects based on the economic
life specified in gen_amortization_period in gen_info.csv instead of
gen_max_age. The plant will still retire when it reaches gen_max_age years old,
and capital recovery will occur at the adjusted rate until gen_max_age.
"""


def define_components(m):
    # custom amortization period per generator
    m.gen_amortization_period = Param(
        m.GENERATION_PROJECTS,
        within=PositiveReals,
        default=lambda m, g: m.gen_max_age[g],
    )

    # Extra CRF needed based on the difference between the default crf (using
    # gen_max_age) and the custom crf (using gen_amortization_period)
    m.gen_amortization_adjustment = Param(
        m.GENERATION_PROJECTS,
        within=Reals,
        initialize=lambda m, g: crf(m.interest_rate, m.gen_amortization_period[g])
        / crf(m.interest_rate, m.gen_max_age[g])
        - 1,
    )

    m.GenAmortizationAdjustment = Expression(
        m.GENERATION_PROJECTS,
        m.PERIODS,
        rule=lambda m, g, p: sum(
            m.BuildGen[g, bld_yr]
            * m.gen_capital_cost_annual[g, bld_yr]
            * m.gen_amortization_adjustment[g]
            for bld_yr in m.BLD_YRS_FOR_GEN_PERIOD[g, p]
        ),
    )
    # Summarize costs for the objective function (total $/year).
    m.TotalGenAmortizationAdjustment = Expression(
        m.PERIODS,
        rule=lambda m, p: sum(
            m.GenAmortizationAdjustment[g, p] for g in m.GENERATION_PROJECTS
        ),
    )
    m.Cost_Components_Per_Period.append("TotalGenAmortizationAdjustment")


def load_inputs(mod, switch_data, inputs_dir):
    """
    Read custom amortization period from gen_info.csv.

    Data read:

    gen_info.csv
        gen_amortization_period

    """

    switch_data.load_aug(
        filename=os.path.join(inputs_dir, "gen_info.csv"),
        param=(mod.gen_amortization_period,),
    )
