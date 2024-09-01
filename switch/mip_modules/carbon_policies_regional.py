"""
Implement regional carbon caps.

carbon_policies_regional.csv shows the caps:

CO2_PROGRAM: name of the program (there may be multiple)
PERIOD: period when the cap applies
LOAD_ZONE: zone participating in the program
carbon_cap_tco2_per_yr: cap for this zone within this program (tCO2/y); carbon can be
traded between zones in the same program

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
import switch_model.reporting as reporting


def define_components(m):
    # read in the data, especially carbon_cap_tco2_per_yr
    # add a constraint that total CO2 emissions in all the zones in each program
    # <= total cap for all zones in that program

    # indexing set for the zonal cap: (program, period, zone) combination
    # (These are all the index columns from carbon_policies_regional.csv.)
    m.REGIONAL_CO2_RULES = Set(dimen=3, within=Any * m.PERIODS * m.LOAD_ZONES)

    # cap specified for each (program, period, zone) combination
    m.carbon_cap_tco2_per_yr = Param(
        m.REGIONAL_CO2_RULES,
        within=Reals,
        default=float("inf"),
    )

    # cost for any emissions above the limit in each (program, period, zone)
    m.carbon_cost_dollar_per_tco2 = Param(
        m.REGIONAL_CO2_RULES,
        within=Reals,
        default=float("inf"),
    )

    # how much to relax the constraint in each program, period, zone
    m.AnnualCapViolation = Var(
        m.REGIONAL_CO2_RULES,
        within=NonNegativeReals,
        bounds=lambda m, pr, pe, z: (
            0,
            # never exceed the cap if no price is specified
            (0 if m.carbon_cost_dollar_per_tco2[pr, pe, z] == float("inf") else None),
        ),
    )

    # names of all the CO2 programs and periods when they are in effect;
    # each unique pair of values in the first two columns of
    # carbon_policies_regional.csv is a (program, period) combo
    m.CO2_PROGRAM_PERIODS = Set(
        dimen=2,
        initialize=lambda m: unique_list(
            (pr, pe) for (pr, pe, lz) in m.REGIONAL_CO2_RULES
        ),
    )

    # set of zones that participate in a particular CO2 program in a particular period
    m.ZONES_IN_CO2_PROGRAM_PERIOD = Set(
        m.CO2_PROGRAM_PERIODS,
        within=m.LOAD_ZONES,
        initialize=lambda m, pr, pe: [
            _z for (_pr, _pe, _z) in m.REGIONAL_CO2_RULES if (_pr, _pe) == (pr, pe)
        ],
    )

    # enforce constraint on total emissions in each program in each period
    def rule(m, pr, pe):
        # sum of zone co2 caps for this program/period

        cap = sum(
            m.carbon_cap_tco2_per_yr[pr, pe, z]
            for z in m.ZONES_IN_CO2_PROGRAM_PERIOD[pr, pe]
        )

        exceedance = sum(
            m.AnnualCapViolation[pr, pe, z]
            for z in m.ZONES_IN_CO2_PROGRAM_PERIOD[pr, pe]
        )

        # sum of annual emissions for gens in this program in this period
        emissions = sum(
            m.DispatchEmissions[g, tp, f] * m.tp_weight_in_year[tp]
            for z in m.ZONES_IN_CO2_PROGRAM_PERIOD[pr, pe]
            # all active fuel-using gens in zone z in period pe
            for g in m.GENS_IN_ZONE[z]
            if g in m.FUEL_BASED_GENS
            for tp in m.TPS_FOR_GEN_IN_PERIOD[g, pe]
            for f in m.FUELS_FOR_GEN[g]
        )
        if cap == float("inf"):
            return Constraint.Skip
        else:
            # scale the cap down into the same range as other vars to improve
            # numerical stability (may not be needed, since it has an easy
            # feasibility fix by just raising the exceedance)
            # define and return the constraint
            return emissions * 0.001 <= (cap + exceedance) * 0.001

    m.Enforce_Regional_Carbon_Cap = Constraint(m.CO2_PROGRAM_PERIODS, rule=rule)

    # could make sure the dual is defined and calculate the dual of this
    # constraint to get the clearing price for carbon in each program if wanted
    # (search for 'dual' in switch_model.policies.carbon_policies for that code)

    m.EmissionsCost = Expression(
        m.PERIODS,
        rule=lambda m, pe: sum(
            m.AnnualCapViolation[_pr, _pe, _z]
            * m.carbon_cost_dollar_per_tco2[_pr, _pe, _z]
            for (_pr, _pe) in m.CO2_PROGRAM_PERIODS
            if _pe == pe
            for _z in m.ZONES_IN_CO2_PROGRAM_PERIOD[_pr, _pe]
            # assume no relaxation if cost per tonne is infinite (omitted)
            if m.carbon_cost_dollar_per_tco2[_pr, _pe, _z] != float("inf")
        ),
        doc="Calculates the carbon cost for generation-related emissions.",
    )
    m.Cost_Components_Per_Period.append("EmissionsCost")


def load_inputs(m, switch_data, inputs_dir):
    """
    Expected input files:
    carbon_policies_regional.csv
        CO2_PROGRAM, PERIODS, LOAD_ZONES, carbon_cap_tco2_per_yr

    """
    switch_data.load_aug(
        filename=os.path.join(inputs_dir, "carbon_policies_regional.csv"),
        optional=True,  # also enables empty files
        index=m.REGIONAL_CO2_RULES,
        param=(m.carbon_cap_tco2_per_yr, m.carbon_cost_dollar_per_tco2),
    )


# could export annual emissions, cap and costs for each program if wanted,
# based on code in switch_model.policies.carbon_policies.post_solve
