"""
Implement regional carbon caps.

carbon_policies_regional.csv shows the caps:

CO2_PROGRAM: name of the program (there may be multiple)
PERIOD: period when the cap applies
LOAD_ZONE: zone participating in the program
zone_co2_cap: cap for this zone within this program (tCO2/y); carbon can be
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
    # read in the data, especially zone_co2_cap
    # add a constraint that total CO2 emissions in all the zones in each program
    # <= total cap for all zones in that program

    # indexing set for the zonal cap: (program, period, zone) combination
    # (These are all the index columns from carbon_policies_regional.csv.)
    m.REGIONAL_CO2_RULES = Set(dimen=3, within=Any * m.PERIODS * m.LOAD_ZONES)

    # cap specified for each (program, period, zone) combination
    m.zone_co2_cap = Param(
        m.REGIONAL_CO2_RULES,
        within=Reals,
        default=float("inf"),
    )

    # give a more helpful error message if data is missing
    m.min_data_check(
        "REGIONAL_CO2_RULES",
        "zone_co2_cap",
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
            m.zone_co2_cap[pr, pe, z] for z in m.ZONES_IN_CO2_PROGRAM_PERIOD[pr, pe]
        )

        # sum of annual emissions for gens in this program in this period
        emissions = sum(
            m.DispatchEmissions[g, tp, f] * m.tp_weight_in_year[tp]
            for z in m.ZONES_IN_CO2_PROGRAM_PERIOD[pr, pe]
            # need all active fuel-using gens in zone z in period pe
            for g in m.GENS_IN_ZONE[z]
            if g in m.FUEL_BASED_GENS
            for tp in m.TPS_FOR_GEN_IN_PERIOD[g, pe]
            for f in m.FUELS_FOR_GEN[g]
        )
        if cap == float("inf"):
            Constraint.Skip
        else:
            # define and return the constraint
            return emissions <= cap

    m.Enforce_Regional_Carbon_Cap = Constraint(m.CO2_PROGRAM_PERIODS, rule=rule)

    # could make sure the dual is defined and calculate the dual of this
    # constraint to get the clearing price for carbon in each program if wanted
    # (search for 'dual' in switch_model.policies.carbon_policies for that code)


def load_inputs(model, switch_data, inputs_dir):
    """
    Expected input files:
    carbon_policies_regional.csv
        CO2_PROGRAM, PERIODS, LOAD_ZONES, zone_co2_cap

    """
    switch_data.load_aug(
        filename=os.path.join(inputs_dir, "carbon_policies_regional.csv"),
        optional=True,
        index=model.REGIONAL_CO2_RULES,
        params=(model.zone_co2_cap),
    )


# could export annual emissions, cap and costs for each program if wanted,
# based on code in switch_model.policies.carbon_policies.post_solve
