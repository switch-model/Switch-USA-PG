from __future__ import division

demand_module = None  # will be set via command-line options


def calibrate_model(m):
    """
    Calibrate the demand system and add it to the model.
    """

    # base_data consists of a list of tuples showing (load_zone, timeseries, base_load (list) and base_price)
    # note: the constructor below assumes list comprehensions will preserve the order of the underlying list
    # (which is guaranteed according to http://stackoverflow.com/questions/1286167/is-the-order-of-results-coming-from-a-list-comprehension-guaranteed)

    # calculate the average-cost price for the current study period
    # TODO: store monthly retail prices in system_load, and find annual average prices
    # that correspond to the load forecasts for each period, then store scale factors
    # in system_load_scale to convert 2007-08 monthly prices into monthly prices for other
    # years (same technique as rescaling the loads, but only adjusting the mean), then
    # report base prices for each timepoint along with the loads in loads.csv.
    # For now, we just assume the base price was $180/MWh, which is HECO's average price in
    # 2007 according to EIA form 826.
    # TODO: add in something for the fixed costs, to make marginal cost commensurate with the base_price
    # baseCosts = [m.dual[m.EnergyBalance[z, tp]] for z in m.LOAD_ZONES for tp in m.TIMEPOINTS]
    base_price = 125  # average retail price for 2022 ($/MWh). From https://www.statista.com/statistics/183700/us-average-retail-electricity-price-since-1990/#:~:text=The%20retail%20price%20for%20electricity,growth%20of%20over%2012%20percent.
    m.base_data = [
        (
            z,
            ts,
            [m.zone_demand_mw[z, tp] for tp in m.TPS_IN_TS[ts]],
            [base_price] * len(m.TPS_IN_TS[ts]),
        )
        for z in m.LOAD_ZONES
        for ts in m.TIMESERIES
    ]

    # make a dict of base_data, indexed by load_zone and timepoint, for later reference
    m.base_data_dict = {
        (z, tp): (m.zone_demand_mw[z, tp], base_price)
        for z in m.LOAD_ZONES
        for tp in m.TIMEPOINTS
    }

    # calibrate the demand module
    demand_module.calibrate(m, m.base_data)


def calibrate(base_data, dr_elasticity_scenario=3):
    """Accept a list of tuples showing [base hourly loads], and [base hourly prices] for each
    location (load_zone) and date (time_series). Store these for later reference by bid().
    """
    # import numpy; we delay till here to avoid interfering with unit tests
    global np
    import numpy as np

    global base_load_dict, base_price_dict, elasticity_scenario
    # build dictionaries (indexed lists) of base loads and prices
    # store the load and price vectors as numpy arrays (vectors) for faste calculation later
    base_load_dict = {
        (z, ts): np.array(base_loads, float)
        for (z, ts, base_loads, base_prices) in base_data
    }
    base_price_dict = {
        (z, ts): np.array(base_prices, float)
        for (z, ts, base_loads, base_prices) in base_data
    }
    elasticity_scenario = dr_elasticity_scenario


def bid(load_zone, time_series, prices):
    """Accept a vector of current prices, for a particular location (load_zone) and day (time_series).
    Return a tuple showing hourly load levels and willingness to pay for those loads (relative to the
    loads achieved at the base_price).

    This version assumes that part of the load is price elastic with constant elasticity of 0.1 and no
    substitution between hours (this part is called "elastic load" below), and the rest of the load is inelastic
    in total volume, but schedules itself to the cheapest hours (this part is called "shiftable load")."""

    elasticity = 0.1
    shiftable_share = 0.1 * elasticity_scenario  # 1-3

    # convert prices to a numpy vector, and make non-zero
    # to avoid errors when raising to a negative power
    p = np.maximum(1.0, np.array(prices, float))

    # get vectors of base loads and prices for this location and date
    bl = base_load_dict[load_zone, time_series]
    bp = base_price_dict[load_zone, time_series]

    # spread shiftable load among all minimum-cost hours,
    # shaped like the original load during those hours (so base prices result in base loads)
    mins = p == np.min(p)

    shiftable_load = np.zeros(len(p))
    shiftable_load[mins] = bl[mins] * shiftable_share * np.sum(bl) / sum(bl[mins])

    # the shiftable load is inelastic, so wtp is the same high number, regardless of when the load is served
    # so _relative_ wtp is always zero
    shiftable_load_wtp = 0

    elastic_base_load = (1.0 - shiftable_share) * bl
    elastic_load = elastic_base_load * (p / bp) ** (-elasticity)
    # _relative_ consumer surplus for the elastic load is the integral
    # of the load (quantity) function from p to bp; note: the hours are independent.
    # if p < bp, consumer surplus decreases as we move from p to bp, so cs_p - cs_p0
    # (given by this integral) is positive.
    elastic_load_cs_diff = np.sum(
        (1 - (p / bp) ** (1 - elasticity)) * bp * elastic_base_load / (1 - elasticity)
    )
    # _relative_ amount actually paid for elastic load under current price, vs base price
    base_elastic_load_paid = np.sum(bp * elastic_base_load)
    elastic_load_paid = np.sum(p * elastic_load)
    elastic_load_paid_diff = elastic_load_paid - base_elastic_load_paid

    demand = shiftable_load + elastic_load
    wtp = shiftable_load_wtp + elastic_load_cs_diff + elastic_load_paid_diff

    return (demand, wtp)
