"""
Implement reuirement of minimum installed capacity for certain type of energy source in some 
areas. -- etc, offshore wind 


min_cap_requirement.csv shows the requirement capacity.
min_cap_generators.csv has the list of qualified generators.

"""
import os
from pyomo.environ import (
    Set,
    Param,
    Expression,
    Constraint,
    Suffix,
    NonNegativeReals,
    Reals,
    Any,
    Var,
)

# from switch_model.utilities import unique_list
import switch_model.reporting as reporting


def define_components(m):

    m.MIN_CAP_RULES = Set(dimen=2, within=Any * m.PERIODS)

    # MIN cap specified for each (program, period) combination
    m.min_MW = Param(m.MIN_CAP_RULES, within=Reals)

    # # give a more helpful error message if data is missing
    # m.min_data_check(
    #     "REGIONAL_CO2_RULES",
    #     "zone_co2_cap",
    # )

    m.MIN_CAP_GEN = Param(m.MIN_CAP_RULES, within=m.GENERATION_PROJECTS)

    # enforce constraint on total installed capacity in each program in each period
    def rule(m, pr, p):

        build_capacity = sum(m.GenCapacity[g, p] for g in m.GENS_QUALIFIED[pr, p])
        min_capacity_requirement = m.min_MW[pr, p]

        # define and return the constraint
        return build_capacity >= min_capacity_requirement

    m.Enforce_Min_Capacity = Constraint(m.MIN_CAP_RULES, rule=rule)


def load_inputs(model, switch_data, inputs_dir):
    """
    Expected input files:
    min_cap_generators.csv
        MIN_CAP_PROGRAM,PERIOD,MIN_CAP_GEN

    min_cap_requirement.csv
        MIN_CAP_PROGRAM,PERIOD,min_MW,program_description

    """
    switch_data.load_aug(
        filename=os.path.join(inputs_dir, "min_cap_generators.csv"),
        index=model.MIN_CAP_RULES,
        params=(model.MIN_CAP_GEN,),
    )
    switch_data.load_aug(
        filename=os.path.join(inputs_dir, "min_cap_requirement.csv"),
        index=model.MIN_CAP_RULES,
        params=(model.min_MW),
    )
