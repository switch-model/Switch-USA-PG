import os
from pyomo.environ import *


def define_components(m):
    """ """
    # maximum fraction of committed capacity that can be ramped up or down per
    # hour

    m.gen_ramp_limit_up = Param(
        m.GENERATION_PROJECTS, within=NonNegativeReals, default=1
    )
    m.gen_ramp_limit_down = Param(
        m.GENERATION_PROJECTS, within=NonNegativeReals, default=1
    )

    # TODO: make this deal more correctly with multi-hour timepoints
    # (only one mandatory shutdown, not n_hours * min_load, and assume it can
    # reach min load in first hour, then ramp from there at standard rate per hour)
    def ramp_up_rule(m, g, tp):
        if m.gen_ramp_limit_up[g] >= 1:
            return Constraint.Skip
        # very similar to GenX: https://github.com/GenXProject/GenX.jl/blob/62c0c0c2501b6c0dcff1416870730ae0428a651c/src/model/resources/thermal/thermal_commit.jl#L187
        max_ramp_up = (
            # capacity that was previously on and is still on now (wasn't just
            # started up) respects the ramp limits
            m.gen_ramp_limit_up[g] * (m.CommitGen[g, tp] - m.StartupGenCapacity[g, tp])
            # capacity starting up this hour can at least jump to min load
            + m.StartupGenCapacity[g, tp]
            * max(m.gen_min_load_fraction_TP[g, tp], m.gen_ramp_limit_up[g])
            # capacity stopping this hour must withdraw at least min load
            - m.ShutdownGenCapacity[g, tp] * m.gen_min_load_fraction_TP[g, tp]
        ) * m.tp_duration_hrs[tp]
        ramp_up = m.DispatchGen[g, tp] - m.DispatchGen[g, m.tp_previous[tp]]
        return ramp_up <= max_ramp_up

    m.Max_Ramp_Up = Constraint(m.GEN_TPS, rule=ramp_up_rule)

    def ramp_down_rule(m, g, tp):
        if m.gen_ramp_limit_down[g] >= 1:
            return Constraint.Skip
        max_ramp_down = (
            # capacity that was previously on and is still on now (wasn't just
            # started up) respects the ramp limits
            m.gen_ramp_limit_down[g]
            * (m.CommitGen[g, tp] - m.StartupGenCapacity[g, tp])
            # capacity starting up this hour must at least jump to min load
            - m.StartupGenCapacity[g, tp] * m.gen_min_load_fraction_TP[g, tp]
            # capacity stopping this hour can withdraw at least min load
            + m.ShutdownGenCapacity[g, tp]
            * max(m.gen_min_load_fraction_TP[g, tp], m.gen_ramp_limit_down[g])
        ) * m.tp_duration_hrs[tp]
        ramp_down = m.DispatchGen[g, m.tp_previous[tp]] - m.DispatchGen[g, tp]
        return ramp_down <= max_ramp_down

    m.Max_Ramp_Down = Constraint(m.GEN_TPS, rule=ramp_down_rule)


def load_inputs(m, switch_data, inputs_dir):
    """
    Import O&M parameters. All columns are optional.

    gen_info.csv
        GENERATION_PROJECT,
        gen_ramp_limit_up,
        gen_ramp_limit_down,
    """

    switch_data.load_aug(
        filename=os.path.join(inputs_dir, "gen_info.csv"),
        optional=True,
        param=(
            m.gen_ramp_limit_up,
            m.gen_ramp_limit_down,
        ),
    )
