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

from switch_model.utilities import unique_list


def define_components(m):

    # (program, period) combinations with minimum capacity rules in effect
    m.MIN_CAP_RULES = Set(dimen=2, within=Any * m.PERIODS)
    # minimum capacity specified for each (program, period) combination
    m.min_cap_mw = Param(m.MIN_CAP_RULES, within=Reals)
    # set of all minimum-capacity programs
    m.MIN_CAP_PROGRAMS = Set(
        initialize=lambda m: unique_list(pr for pr, pe in m.MIN_CAP_RULES)
    )

    # set of all valid program/generator combinations (i.e., gens participating
    # in each program)
    m.MIN_CAP_PROGRAM_GENS = Set(within=m.MIN_CAP_PROGRAMS * m.GENERATION_PROJECTS)
    m.GENS_IN_MIN_CAP_PROGRAM = Set(
        m.MIN_CAP_PROGRAMS,
        within=m.GENERATION_PROJECTS,
        initialize=lambda m, pr: unique_list(
            _g for (_pr, _g) in m.MIN_CAP_PROGRAM_GENS if _pr == pr
        ),
    )

    # enforce constraint on total installed capacity in each program in each period
    def rule(m, pr, pe):

        build_capacity = sum(
            m.GenCapacity[g, pe] for g in m.GENS_IN_MIN_CAP_PROGRAM[pr]
        )
        min_capacity_requirement = m.min_cap_mw[pr, pe]

        # define and return the constraint
        return build_capacity >= min_capacity_requirement

    m.Enforce_Min_Capacity = Constraint(m.MIN_CAP_RULES, rule=rule)


def load_inputs(model, switch_data, inputs_dir):
    """
    Expected input files:
    min_cap_generators.csv
        MIN_CAP_PROGRAM,PERIOD,MIN_CAP_GEN

    min_cap_requirements.csv
        MIN_CAP_PROGRAM,PERIOD,min_cap_mw

    """
    switch_data.load_aug(
        filename=os.path.join(inputs_dir, "min_cap_requirements.csv"),
        optional=True,  # also enables empty files
        index=model.MIN_CAP_RULES,
        param=(model.min_cap_mw,),
    )
    switch_data.load_aug(
        filename=os.path.join(inputs_dir, "min_cap_generators.csv"),
        optional=True,  # also enables empty files
        set=model.MIN_CAP_PROGRAM_GENS,
    )
