import os
from pyomo.environ import *


def define_components(m):
    """
    gen_fixed_om_by_period[(g, p) in m.GEN_PERIODS]

    gen_storage_energy_fixed_om_by_period[(g, p) in m.GEN_PERIODS]

    gen_variable_om_by_period[(g, p) in m.GEN_PERIODS]

    note: there's no gen_storage_energy_variable_om_by_period because that
    would be the same as gen_variable_om_by_period.

    is the
    annual fixed operations & maintenance cost per MWh of energy capacity
    installed. This is charged every year over the life of the storage project,
    whether it is operated or not. It should be in units of real dollars per
    year per MWh of capacity. This should only be defined for storage
    technologies; it will be ignored for non-storage generators. Note that this
    shows the cost per unit of energy capacity (i.e., batteries) and
    gen_fixed_om shows the cost per unit of power capacity (i.e. inverters).
    """

    m.gen_fixed_om_by_period = Param(m.GEN_PERIODS, within=Reals, default=0.0)
    m.gen_storage_energy_fixed_om_by_period = Param(
        m.GEN_PERIODS, within=Reals, default=0.0
    )
    m.gen_variable_om_by_period = Param(m.GEN_PERIODS, within=Reals, default=0.0)

    m.Gen_Fixed_OM_by_Period = Expression(
        m.PERIODS,
        rule=lambda m, p: (
            sum(
                m.GenCapacity[g, p] * m.gen_fixed_om_by_period[g, p]
                for g in m.GENS_IN_PERIOD[p]
            )
            + sum(
                m.StorageEnergyCapacity[g, p]
                * m.gen_storage_energy_fixed_om_by_period[g, p]
                for g in m.STORAGE_GENS
                if p in m.PERIODS_FOR_GEN[g]
            )
            if hasattr(m, "STORAGE_GENS")
            else 0
        ),
    )
    m.Cost_Components_Per_Period.append("Gen_Fixed_OM_by_Period")

    m.Gen_Variable_OM_by_Period = Expression(
        m.TIMEPOINTS,
        rule=lambda m, tp: sum(
            m.DispatchGen[g, tp] * m.gen_variable_om_by_period[g, m.tp_period[tp]]
            for g in m.GENS_IN_PERIOD[m.tp_period[tp]]
        ),
    )
    m.Cost_Components_Per_TP.append("Gen_Variable_OM_by_Period")


def load_inputs(m, switch_data, inputs_dir):
    """
    Import O&M parameters. All columns are optional.

    gen_om_by_period.csv
        GENERATION_PROJECT,
        PERIOD,
        gen_fixed_om_by_period,
        gen_variable_om_by_period,
        gen_storage_energy_fixed_om_by_period,
    """

    switch_data.load_aug(
        filename=os.path.join(inputs_dir, "gen_om_by_period.csv"),
        optional=True,
        param=(
            m.gen_fixed_om_by_period,
            m.gen_variable_om_by_period,
            m.gen_storage_energy_fixed_om_by_period,
        ),
    )
