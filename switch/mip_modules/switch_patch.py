from pyomo.environ import *
import inspect
import switch_model.solve


def define_components(m):
    # allow negative variable O&M for current policies case
    m.gen_variable_om.domain = Reals


def patch_switch():

    # patch switch_model.solve (2.0.7) to work without suffixes
    old_code = """
    suffixes = [c.name for c in model.component_objects(ctype=Suffix)]
    if suffixes == []:
        # use None instead of an empty list for compatibility with appsi_highs
        # and maybe some other solvers
        suffixes = None
    solver_args["suffixes"] = suffixes"""
    new_code = """
    suffixes = [c.name for c in model.component_objects(ctype=Suffix)]
    if suffixes:
        # don't assign at all if no suffixes are defined, since appsi_highs
        # (and maybe others) want None instead of an empty list, but cplex and
        # gurobi crash with None
        solver_args["suffixes"] = suffixes"""

    solve_code = inspect.getsource(switch_model.solve.solve)
    if old_code in solve_code:
        # create and inject a new version of the code
        solve_code = solve_code.replace(old_code, new_code)
        switch_model.solve.replace_method(switch_model.solve, "solve", solve_code)


patch_switch()
