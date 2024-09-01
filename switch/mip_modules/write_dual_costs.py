"""
Report rents or stranded costs for all aspects of the power system.

This can be useful for inferring whether average marginal cost pricing will
produce enough revenue to cover all costs, or whether there will be surplus
or shortfall on the production side.
"""

import os, time
from pyomo.environ import Set, Param, Expression, Constraint, Suffix, Var, value
from pyomo.repn import generate_standard_repn

from switch_model.reporting import write_table

# from switch_model.balancing.demand_response.iterative import write_dual_costs


def define_arguments(argparser):
    argparser.add_argument(
        "--write-all-duals",
        action="store_true",
        default=False,
        help="Report duals for all constraints and bounds in dual_costs.csv, "
        "not just those with non-zero total value.",
    )


def define_components(m):
    # Make sure the model has dual and rc suffixes
    if not hasattr(m, "dual"):
        m.dual = Suffix(direction=Suffix.IMPORT)
    if not hasattr(m, "rc"):
        m.rc = Suffix(direction=Suffix.IMPORT)


# cloned from switch_model.balancing.demand_response.iterative.write_dual_costs
# with minor changes 2023-11-17
def write_dual_costs(m):
    outputs_dir = m.options.outputs_dir
    # tag = filename_tag(m, include_iter_num)

    # with open(os.path.join(outputs_dir, "producer_surplus{t}.csv".format(t=tag)), 'w') as f:
    #     for g, per in m.Max_Build_Potential:
    #         const = m.Max_Build_Potential[g, per]
    #         surplus = const.upper() * m.dual[const]
    #         if surplus != 0.0:
    #             f.write(','.join([const.name, str(surplus)]) + '\n')
    #     # import pdb; pdb.set_trace()
    #     for g, year in m.BuildGen:
    #         var = m.BuildGen[g, year]
    #         if var.ub is not None and var.ub > 0.0 and value(var) > 0.0 and var in m.rc and m.rc[var] != 0.0:
    #             surplus = var.ub * m.rc[var]
    #             f.write(','.join([var.name, str(surplus)]) + '\n')

    outfile = os.path.join(outputs_dir, "dual_costs.csv")
    dual_data = []
    start_time = time.time()
    print(f"Writing {outfile} ... ", end=" ")

    def add_dual(const, lbound, ubound, duals, prefix="", offset=0.0):
        if const in duals:
            dual = duals[const]
            if dual >= 0.0:
                direction = ">="
                bound = lbound
            else:
                direction = "<="
                bound = ubound
            if bound is None:
                # Variable is unbounded; dual should be 0.0 or possibly a tiny non-zero value.
                if not (-1e-5 < dual < 1e-5):
                    # Weird case; generate warning and weird data
                    print(
                        f"WARNING: {const.name} has no {'lower' if dual > 0 else 'upper'} "
                        f"bound but has a non-zero dual value {dual}."
                    )
                    dual_data.append(
                        (
                            prefix + const.parent_component().name,
                            str(const.index()),
                            direction,
                            f"None + {offset}",
                            dual,
                            f"None + {dual * offset}",
                        )
                    )
            else:
                total_cost = dual * (bound + offset)
                if total_cost != 0.0 or m.options.write_all_duals:
                    dual_data.append(
                        (
                            prefix + const.parent_component().name,
                            str(const.index()),
                            direction,
                            (bound + offset),
                            dual,
                            total_cost,
                        )
                    )

    for comp in m.component_objects(ctype=Var):
        for idx in comp:
            var = comp[idx]
            if var.value is not None:  # ignore vars that weren't used in the model
                if var.is_integer() or var.is_binary():
                    # integrality constraint sets upper and lower bounds
                    add_dual(var, value(var), value(var), m.rc, prefix="integer: ")
                else:
                    add_dual(var, var.lb, var.ub, m.rc)
    for comp in m.component_objects(ctype=Constraint):
        for idx in comp:
            constr = comp[idx]
            if constr.active:
                offset = 0.0
                # cancel out any constants that were stored in the body instead of the bounds
                # (see https://groups.google.com/d/msg/pyomo-forum/-loinAh0Wx4/IIkxdfqxAQAJ)
                # (might be faster to do this once during model setup instead of every time)
                standard_constraint = generate_standard_repn(constr.body)
                if standard_constraint.constant is not None:
                    offset = -standard_constraint.constant
                add_dual(
                    constr,
                    value(constr.lower),
                    value(constr.upper),
                    m.dual,
                    offset=offset,
                )

    dual_data.sort(
        key=lambda r: (
            not r[0].startswith("DR_Convex_"),
            isinstance(r[3], str) or r[3] >= 0,
            r[0],
            r[1],
            r[2],
        )
    )

    write_table(
        m,
        range(len(dual_data)),
        output_file=outfile,
        headings=("constraint", "index", "direction", "bound", "dual", "total_cost"),
        values=lambda m, i: dual_data[i],
    )

    # with open(outfile, "w") as f:
    #     f.write(
    #         ",".join(["constraint", "direction", "bound", "dual", "total_cost"]) + "\n"
    #     )
    #     f.writelines(",".join(map(str, r)) + "\n" for r in dual_data)

    print("time taken: {dur:.2f}s".format(dur=time.time() - start_time))


def post_solve(m, outputs_dir):
    write_dual_costs(m)
