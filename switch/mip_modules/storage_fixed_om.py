import os
from pyomo.environ import *

# TODO: merge this into standard storage module (and add variable O&M and a test case)


def define_components(mod):
    """
    gen_storage_energy_fixed_om[(g, bld_yr) in STORAGE_GEN_BLD_YRS] is the
    annual fixed operations & maintenance cost per MWh of energy capacity
    installed. This is charged every year over the life of the storage project,
    whether it is operated or not. It should be in units of real dollars per
    year per MWh of capacity. This should only be defined for storage
    technologies; it will be ignored for non-storage generators. Note that this
    shows the cost per unit of energy capacity (i.e., batteries) and
    gen_fixed_om shows the cost per unit of power capacity (i.e. inverters).
    """

    mod.gen_storage_energy_fixed_om = Param(
        mod.GEN_BLD_YRS, within=NonNegativeReals, default=0.0
    )

    # Add storage-energy-related fixed O&M costs to the objective function.
    mod.TotalStorageEnergyFixedOMCosts = Expression(
        mod.PERIODS,
        rule=lambda m, p: sum(
            # apply appropriate cost to all vintages (bld_yr) of storage that
            # are active in the current period (p)
            m.BuildStorageEnergy[g, bld_yr] * m.gen_storage_energy_fixed_om[g, bld_yr]
            for g in m.STORAGE_GENS
            for bld_yr in m.BLD_YRS_FOR_GEN_PERIOD[g, p]
        ),
    )
    mod.Cost_Components_Per_Period.append("TotalStorageEnergyFixedOMCosts")

    # Some projects are retired before the first study period, so they
    # don't appear in the objective function or any constraints.
    # In this case, pyomo may leave the variable value undefined even
    # after a solve, instead of assigning a value within the allowed
    # range. This causes errors in the Progressive Hedging code, which
    # expects every variable to have a value after the solve. So as a
    # starting point we assign an appropriate value to all the existing
    # projects here.
    def BuildStorageEnergy_assign_default_value(m, g, bld_yr):
        if (g, bld_yr) in m.build_gen_energy_predetermined:
            m.BuildStorageEnergy[g, bld_yr] = m.build_gen_energy_predetermined[
                g, bld_yr
            ]
        elif g in m.STORAGE_GENS and m.gen_storage_energy_to_power_ratio[g] == float(
            "inf"
        ):
            raise ValueError(
                f"For storage generator g='{g}', gen_build_predetermined[g, {bld_yr}] "
                f"has been specified, but not "
                f"gen_build_energy_predetermined[g, {bld_yr}] or "
                f"gen_storage_energy_to_power_ratio[g]."
            )

    mod.BuildStorageEnergy_assign_default_value = BuildAction(
        mod.PREDETERMINED_GEN_BLD_YRS, rule=BuildStorageEnergy_assign_default_value
    )

    # TODO: expand m.PREDETERMINED_GEN_BLD_YRS to include generators with
    # energy specified but not power, and in these cases, raise an error if
    # energy_to_power_ratio is not specified.


def load_inputs(mod, switch_data, inputs_dir):
    """
    Import storage parameters. Optional columns are noted with a *.

    gen_build_costs.csv
        GENERATION_PROJECT, build_year, ...
        gen_storage_energy_fixed_om*
    """

    switch_data.load_aug(
        filename=os.path.join(inputs_dir, "gen_build_costs.csv"),
        param=(mod.gen_storage_energy_fixed_om,),
    )
