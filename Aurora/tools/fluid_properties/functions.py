# -*- coding: utf-8

"""Module for fluid property functions.


This file is part of project TESPy (github.com/oemof/Aurora). It's copyrighted
by the contributors recorded in the version control history of the file,
available from its original location
Aurora/tools/fluid_properties/functions.py

SPDX-License-Identifier: MIT
"""

from .helpers import _check_mixing_rule
from .helpers import get_number_of_fluids
from .helpers import get_pure_fluid
from .helpers import inverse_temperature_mixture
from .mixtures import EXERGY_CHEMICAL
from .mixtures import H_MIX_PT_DIRECT
from .mixtures import S_MIX_PT_DIRECT
from .mixtures import T_MIX_PH_REVERSE
from .mixtures import T_MIX_PS_REVERSE
from .mixtures import V_MIX_PT_DIRECT
from .mixtures import VISCOSITY_MIX_PT_DIRECT


def isentropic(p_1, h_1, p_2, fluid_data, mixing_rule=None, T0=None):
    if get_number_of_fluids(fluid_data) == 1:
        pure_fluid = get_pure_fluid(fluid_data)
        return pure_fluid["wrapper"].isentropic(p_1, h_1, p_2)
    else:
        s_1 = s_mix_ph(p_1, h_1, fluid_data, mixing_rule)
        T_2 = T_mix_ps(p_2, s_1, fluid_data, mixing_rule)
        return h_mix_pT(p_2, T_2, fluid_data, mixing_rule)


def calc_physical_exergy(h, s, p, pamb, Tamb, fluid_data, mixing_rule=None, T0=None):
    r"""
    Calculate specific physical exergy.

    Physical exergy is allocated to a thermal and a mechanical share according
    to :cite:`Morosuk2019`.

    Parameters
    ----------
    pamb : float
        Ambient pressure p0 / Pa.

    Tamb : float
        Ambient temperature T0 / K.

    Returns
    -------
    e_ph : tuple
        Specific thermal and mechanical exergy
        (:math:`e^\mathrm{T}`, :math:`e^\mathrm{M}`) in J / kg.

        .. math::

            e^\mathrm{T} = \left( h - h \left( p, T_0 \right) \right) -
            T_0 \cdot \left(s - s\left(p, T_0\right)\right)

            e^\mathrm{M}=\left(h\left(p,T_0\right)-h\left(p_0,T_0\right)\right)
            -T_0\cdot\left(s\left(p, T_0\right)-s\left(p_0,T_0\right)\right)

            e^\mathrm{PH} = e^\mathrm{T} + e^\mathrm{M}
    """
    h_T0_p = h_mix_pT(p, Tamb, fluid_data, mixing_rule)
    s_T0_p = s_mix_pT(p, Tamb, fluid_data, mixing_rule)
    ex_therm = (h - h_T0_p) - Tamb * (s - s_T0_p)
    h0 = h_mix_pT(pamb, Tamb, fluid_data, mixing_rule)
    s0 = s_mix_pT(pamb, Tamb, fluid_data, mixing_rule)
    ex_mech = (h_T0_p - h0) - Tamb * (s_T0_p - s0)
    return ex_therm, ex_mech


def calc_chemical_exergy(pamb, Tamb, fluid_data, Chem_Ex, mixing_rule=None, T0=None):
    if get_number_of_fluids(fluid_data) == 1:
        pure_fluid = get_pure_fluid(fluid_data)
        fluid_aliases = pure_fluid["wrapper"]._aliases
        y = [Chem_Ex[k][Chem_Ex[k][4]] for k in fluid_aliases if k in Chem_Ex]
        return y[0] / pure_fluid["wrapper"]._molar_mass * 1e3
    else:
        _check_mixing_rule(mixing_rule, EXERGY_CHEMICAL, "chemical exergy")
        return EXERGY_CHEMICAL[mixing_rule](pamb, Tamb, fluid_data, Chem_Ex)


def T_mix_ph(p, h, fluid_data, mixing_rule=None, T0=None, T_min=275.9):
    if get_number_of_fluids(fluid_data) == 1:
        pure_fluid = get_pure_fluid(fluid_data)
        return pure_fluid["wrapper"].T_ph(p, h)
    else:
        _check_mixing_rule(mixing_rule, T_MIX_PH_REVERSE, "temperature (from enthalpy)")
        kwargs = {
            "p": p, "target_value": h, "fluid_data": fluid_data, "T0": T0,
            "f": T_MIX_PH_REVERSE[mixing_rule], 'wrapper_func': T_mix_ph, 'T_min': T_min
        }
        # !!!!!!!!!!
        return inverse_temperature_mixture(**kwargs)


def dT_mix_pdh(p, h, fluid_data, mixing_rule=None, T0=None):
    d = 1e-1
    upper = T_mix_ph(p, h + d, fluid_data, mixing_rule=mixing_rule, T0=T0)
    lower = T_mix_ph(p, h - d, fluid_data, mixing_rule=mixing_rule, T0=upper)
    return (upper - lower) / (2 * d)


def dT_mix_dph(p, h, fluid_data, mixing_rule=None, T0=None):
    d = 1e-1
    upper = T_mix_ph(p + d, h, fluid_data, mixing_rule=mixing_rule, T0=T0)
    lower = T_mix_ph(p - d, h, fluid_data, mixing_rule=mixing_rule, T0=upper)
    return (upper - lower) / (2 * d)


def dT_mix_ph_dfluid(p, h, fluid, fluid_data, mixing_rule=None, T0=None):
    d = 1e-5
    fluid_data[fluid]["mass_fraction"] += d
    upper = T_mix_ph(p, h, fluid_data, mixing_rule=mixing_rule, T0=T0)
    fluid_data[fluid]["mass_fraction"] -= 2 * d
    lower = T_mix_ph(p, h, fluid_data, mixing_rule=mixing_rule, T0=upper)
    fluid_data[fluid]["mass_fraction"] += d
    return (upper - lower) / (2 * d)


def d2T_mix_p_d2h(p, h, fluid_data, mixing_rule=None, T0=None):
    d = 1e-3
    upper = T_mix_ph(p, h + 2 * d, fluid_data, mixing_rule=mixing_rule, T0=T0)
    mider = T_mix_ph(p, h, fluid_data, mixing_rule=mixing_rule, T0=upper)
    lower = T_mix_ph(p, h - 2 * d, fluid_data, mixing_rule=mixing_rule, T0=mider)
    return (upper + lower - 2 * mider) / (4 * d**2)


def d2T_mix_d2p_h(p, h, fluid_data, mixing_rule=None, T0=None):
    d = 1e-3
    upper = T_mix_ph(p + 2 * d, h, fluid_data, mixing_rule=mixing_rule, T0=T0)
    mider = T_mix_ph(p, h, fluid_data, mixing_rule=mixing_rule, T0=upper)
    lower = T_mix_ph(p - 2 * d, h, fluid_data, mixing_rule=mixing_rule, T0=mider)
    return (upper + lower - 2 * mider) / (4 * d ** 2)


def d2T_mix_ph_d2fluid(p, h, fluid, fluid_data, mixing_rule=None, T0=None):
    d = 1e-5
    origin_value = fluid_data[fluid]["mass_fraction"]
    mider = T_mix_ph(p, h, fluid_data, mixing_rule=mixing_rule, T0=T0)
    fluid_data[fluid]["mass_fraction"] = origin_value + 2 * d
    upper = T_mix_ph(p, h, fluid_data, mixing_rule=mixing_rule, T0=mider)
    fluid_data[fluid]["mass_fraction"] = origin_value - 2 * d
    lower = T_mix_ph(p, h, fluid_data, mixing_rule=mixing_rule, T0=upper)
    fluid_data[fluid]["mass_fraction"] = origin_value
    return (upper + lower - 2 * mider) / (4 * d ** 2)


def d2T_mix_dpdh(p, h, fluid_data, mixing_rule=None, T0=None):
    d1 = 1e-3
    d2 = 1e-3
    y11 = T_mix_ph(p + d1, h + d2, fluid_data, mixing_rule=mixing_rule, T0=T0)
    y22 = T_mix_ph(p - d1, h - d2, fluid_data, mixing_rule=mixing_rule, T0=y11)
    y12 = T_mix_ph(p + d1, h - d2, fluid_data, mixing_rule=mixing_rule, T0=y22)
    y21 = T_mix_ph(p - d1, h + d2, fluid_data, mixing_rule=mixing_rule, T0=y12)
    return (y11 + y22 - y12 - y21) / (4 * d1 * d2)


def d2T_mix_dp_h_dfluid(p, h, fluid, fluid_data, mixing_rule=None, T0=None):
    d1 = 1e-3
    d2 = 1e-5
    origin_value = fluid_data[fluid]["mass_fraction"]
    fluid_data[fluid]["mass_fraction"] = origin_value + d2
    y11 = T_mix_ph(p + d1, h, fluid_data, mixing_rule=mixing_rule, T0=T0)
    fluid_data[fluid]["mass_fraction"] = origin_value - d2
    y22 = T_mix_ph(p - d1, h, fluid_data, mixing_rule=mixing_rule, T0=y11)
    y12 = T_mix_ph(p + d1, h, fluid_data, mixing_rule=mixing_rule, T0=y22)
    fluid_data[fluid]["mass_fraction"] = origin_value + d2
    y21 = T_mix_ph(p - d1, h, fluid_data, mixing_rule=mixing_rule, T0=y12)
    fluid_data[fluid]["mass_fraction"] = origin_value
    return (y11 + y22 - y12 - y21) / (4 * d1 * d2)


def d2T_mix_p_dh_dfluid(p, h, fluid, fluid_data, mixing_rule=None, T0=None):
    d1 = 1e-3
    d2 = 1e-5
    origin_value = fluid_data[fluid]["mass_fraction"]
    fluid_data[fluid]["mass_fraction"] = origin_value + d2
    y11 = T_mix_ph(p, h + d1, fluid_data, mixing_rule=mixing_rule, T0=T0)
    fluid_data[fluid]["mass_fraction"] = origin_value - d2
    y22 = T_mix_ph(p, h - d1, fluid_data, mixing_rule=mixing_rule, T0=y11)
    y12 = T_mix_ph(p, h + d1, fluid_data, mixing_rule=mixing_rule, T0=y22)
    fluid_data[fluid]["mass_fraction"] = origin_value + d2
    y21 = T_mix_ph(p, h - d1, fluid_data, mixing_rule=mixing_rule, T0=y12)
    fluid_data[fluid]["mass_fraction"] = origin_value
    return (y11 + y22 - y12 - y21) / (4 * d1 * d2)


def d2T_mix_ph_dfluid1_dfluid2(p, h, fluid1, fluid2, fluid_data, mixing_rule=None, T0=None):
    d1 = 1e-5
    d2 = 1e-5
    origin_value1 = fluid_data[fluid1]["mass_fraction"]
    origin_value2 = fluid_data[fluid2]["mass_fraction"]

    fluid_data[fluid1]["mass_fraction"] = origin_value1 + d1
    fluid_data[fluid2]["mass_fraction"] = origin_value2 + d2
    y11 = T_mix_ph(p, h, fluid_data, mixing_rule=mixing_rule, T0=T0)

    fluid_data[fluid1]["mass_fraction"] = origin_value1 - d1
    fluid_data[fluid2]["mass_fraction"] = origin_value2 - d2
    y22 = T_mix_ph(p, h, fluid_data, mixing_rule=mixing_rule, T0=y11)

    fluid_data[fluid1]["mass_fraction"] = origin_value1 + d1
    fluid_data[fluid2]["mass_fraction"] = origin_value2 - d2
    y12 = T_mix_ph(p, h, fluid_data, mixing_rule=mixing_rule, T0=y22)

    fluid_data[fluid1]["mass_fraction"] = origin_value1 - d1
    fluid_data[fluid2]["mass_fraction"] = origin_value2 + d2
    y21 = T_mix_ph(p, h, fluid_data, mixing_rule=mixing_rule, T0=y12)

    return (y11 + y22 - y12 - y21) / (4 * d1 * d2)


def h_mix_pT(p, T, fluid_data, mixing_rule=None):
    if get_number_of_fluids(fluid_data) == 1:
        pure_fluid = get_pure_fluid(fluid_data)
        return pure_fluid["wrapper"].h_pT(p, T)
    else:
        _check_mixing_rule(mixing_rule, H_MIX_PT_DIRECT, "enthalpy")
        return H_MIX_PT_DIRECT[mixing_rule](p, T, fluid_data)


def h_mix_pQ(p, Q, fluid_data, mixing_rule=None):
    if get_number_of_fluids(fluid_data) == 1:
        pure_fluid = get_pure_fluid(fluid_data)
        return pure_fluid["wrapper"].h_pQ(p, Q)
    else:
        msg = "Saturation function cannot be called on mixtures."
        raise ValueError(msg)


def dh_mix_dpQ(p, Q, fluid_data, mixing_rule=None):
    d = 0.1
    upper = h_mix_pQ(p + d, Q, fluid_data)
    lower = h_mix_pQ(p - d, Q, fluid_data)
    return (upper - lower) / (2 * d)


def d2h_mix_d2p_Q(p, Q, fluid_data, mixing_rule=None):
    d = 0.1
    upper = h_mix_pQ(p + d, Q, fluid_data)
    mider = h_mix_pQ(p, Q, fluid_data)
    lower = h_mix_pQ(p - d, Q, fluid_data)
    return (upper + lower - 2 * mider) / (4 * d**2)


def h_mix_pv(p, v, fluid_data, mixing_rule=None):
    pass


def p_mix_hT(h, T, fluid_data, mixing_rule=None):
    if get_number_of_fluids(fluid_data) == 1:
        pure_fluid = get_pure_fluid(fluid_data)
        return pure_fluid["wrapper"].p_hT(h, T)
    pass


def p_mix_hv(h, v, fluid_data, mixing_rule=None):
    pass


def p_mix_hQ(h, Q, fluid_data, mixing_rule=None):
    if get_number_of_fluids(fluid_data) == 1:
        pure_fluid = get_pure_fluid(fluid_data)
        return pure_fluid["wrapper"].p_hQ(h, Q)
    else:
        msg = "Saturation function cannot be called on mixtures."
        raise ValueError(msg)


def Q_mix_ph(p, h, fluid_data, mixing_rule=None):
    if get_number_of_fluids(fluid_data) == 1:
        pure_fluid = get_pure_fluid(fluid_data)
        return pure_fluid["wrapper"].Q_ph(p, h)
    else:
        msg = "Saturation function cannot be called on mixtures."
        raise ValueError(msg)


def phase_mix_ph(p, h, fluid_data, mixing_rule=None):
    if get_number_of_fluids(fluid_data) == 1:
        pure_fluid = get_pure_fluid(fluid_data)
        return pure_fluid["wrapper"].phase_ph(p, h)
    else:
        msg = "State function cannot be called on mixtures."
        raise ValueError(msg)


def p_critical_fluids(c, fluid_data):
    if get_number_of_fluids(fluid_data) == 1:
        pure_fluid = get_pure_fluid(fluid_data)
        return pure_fluid["wrapper"]._p_crit
    else:
        msg = f"There is no p_critical on mixtures in connection: {c}."
        raise ValueError(msg)

def p_crit_(fluid_data):
    if get_number_of_fluids(fluid_data) == 1:
        pure_fluid = get_pure_fluid(fluid_data)
        return pure_fluid["wrapper"]._p_crit
    else:
        msg = "There is no p_critical on mixtures."
        raise ValueError(msg)


def p_sat_T(T, fluid_data, mixing_rule=None):
    if get_number_of_fluids(fluid_data) == 1:
        pure_fluid = get_pure_fluid(fluid_data)
        return pure_fluid["wrapper"].p_sat(T)
    else:
        msg = "Saturation function cannot be called on mixtures."
        raise ValueError(msg)


def T_sat_p(p, fluid_data, mixing_rule=None):
    if get_number_of_fluids(fluid_data) == 1:
        pure_fluid = get_pure_fluid(fluid_data)
        return pure_fluid["wrapper"].T_sat(p)
    else:
        msg = "Saturation function cannot be called on mixtures."
        raise ValueError(msg)


def dT_sat_dp(p, fluid_data, mixing_rule=None):
    d = 0.01
    upper = T_sat_p(p + d, fluid_data)
    lower = T_sat_p(p - d, fluid_data)
    return (upper - lower) / (2 * d)


def d2T_sat_d2p(p, fluid_data, mixing_rule=None):
    d = 0.01
    upper = T_sat_p(p + d, fluid_data)
    mider = T_sat_p(p, fluid_data)
    lower = T_sat_p(p - d, fluid_data)
    return (upper + lower - 2 * mider) / (4 * d ** 2)


def s_mix_ph(p, h, fluid_data, mixing_rule=None, T0=None):
    if get_number_of_fluids(fluid_data) == 1:
        pure_fluid = get_pure_fluid(fluid_data)
        return pure_fluid["wrapper"].s_ph(p, h)
    else:
        T = T_mix_ph(p, h , fluid_data, mixing_rule, T0)
        return s_mix_pT(p, T, fluid_data, mixing_rule)



def s_mix_pT(p, T, fluid_data, mixing_rule=None):
    if get_number_of_fluids(fluid_data) == 1:
        pure_fluid = get_pure_fluid(fluid_data)
        return pure_fluid["wrapper"].s_pT(p, T)
    else:
        _check_mixing_rule(mixing_rule, S_MIX_PT_DIRECT, "entropy")
        return S_MIX_PT_DIRECT[mixing_rule](p, T, fluid_data)


def T_mix_ps(p, s, fluid_data, mixing_rule=None, T0=None):
    if get_number_of_fluids(fluid_data) == 1:
        pure_fluid = get_pure_fluid(fluid_data)
        return pure_fluid["wrapper"].T_ps(p, s)
    else:
        _check_mixing_rule(mixing_rule, T_MIX_PS_REVERSE, "temperature (from entropy)")
        kwargs = {
            "p": p, "target_value": s, "fluid_data": fluid_data, "T0": T0,
            "f": T_MIX_PS_REVERSE[mixing_rule], 'wrapper_func': T_mix_ps
        }
        return inverse_temperature_mixture(**kwargs)


def v_mix_ph(p, h, fluid_data, mixing_rule=None, T0=None):
    if get_number_of_fluids(fluid_data) == 1:
        pure_fluid = get_pure_fluid(fluid_data)
        return 1 / pure_fluid["wrapper"].d_ph(p, h)
    else:
        T = T_mix_ph(p, h, fluid_data, mixing_rule, T0)
        return v_mix_pT(p, T, fluid_data, mixing_rule)


def dv_mix_dph(p, h, fluid_data, mixing_rule=None, T0=None):
    d = 1e-1
    upper = v_mix_ph(p + d, h, fluid_data, mixing_rule=mixing_rule, T0=T0)
    lower = v_mix_ph(p - d, h, fluid_data, mixing_rule=mixing_rule, T0=T0)
    return (upper - lower) / (2 * d)


def dv_mix_pdh(p, h, fluid_data, mixing_rule=None, T0=None):
    d = 1e-1
    upper = v_mix_ph(p, h + d, fluid_data, mixing_rule=mixing_rule, T0=T0)
    lower = v_mix_ph(p, h - d, fluid_data, mixing_rule=mixing_rule, T0=T0)
    return (upper - lower) / (2 * d)


def dv_mix_ph_dfluid(p, h, fluid, fluid_data, mixing_rule=None, T0=None):
    d = 1e-5
    fluid_data[fluid]["mass_fraction"] += d
    upper = v_mix_ph(p, h, fluid_data, mixing_rule=mixing_rule, T0=T0)
    fluid_data[fluid]["mass_fraction"] -= 2 * d
    lower = v_mix_ph(p, h, fluid_data, mixing_rule=mixing_rule, T0=T0)
    fluid_data[fluid]["mass_fraction"] += d
    return (upper - lower) / (2 * d)


def d2v_mix_d2p_h(p, h, fluid_data, mixing_rule=None, T0=None):
    d = 1e-1
    upper = v_mix_ph(p + 2 * d, h, fluid_data, mixing_rule=mixing_rule, T0=T0)
    mider = v_mix_ph(p, h, fluid_data, mixing_rule=mixing_rule, T0=T0)
    lower = v_mix_ph(p - 2 * d, h, fluid_data, mixing_rule=mixing_rule, T0=T0)
    return (upper + lower - 2 * mider) / (4 * d ** 2)


def d2v_mix_p_d2h(p, h, fluid_data, mixing_rule=None, T0=None):
    d = 1e-1
    upper = v_mix_ph(p, h + 2 * d, fluid_data, mixing_rule=mixing_rule, T0=T0)
    mider = v_mix_ph(p, h, fluid_data, mixing_rule=mixing_rule, T0=T0)
    lower = v_mix_ph(p, h - 2 * d, fluid_data, mixing_rule=mixing_rule, T0=T0)
    return (upper + lower - 2 * mider) / (4 * d ** 2)


def d2v_mix_ph_d2fluid(p, h, fluid, fluid_data, mixing_rule=None, T0=None):
    d = 1e-5
    origin_value = fluid_data[fluid]["mass_fraction"]
    mider = v_mix_ph(p, h, fluid_data, mixing_rule=mixing_rule, T0=T0)
    fluid_data[fluid]["mass_fraction"] = origin_value + 2 * d
    upper = v_mix_ph(p, h, fluid_data, mixing_rule=mixing_rule, T0=T0)
    fluid_data[fluid]["mass_fraction"] = origin_value - 2 * d
    lower = v_mix_ph(p, h, fluid_data, mixing_rule=mixing_rule, T0=T0)
    fluid_data[fluid]["mass_fraction"] = origin_value
    return (upper + lower - 2 * mider) / (4 * d ** 2)


def d2v_mix_dp_dh(p, h, fluid_data, mixing_rule=None, T0=None):
    d1 = 1e-1
    d2 = 1e-1
    y11 = v_mix_ph(p + d1, h + d2, fluid_data, mixing_rule=mixing_rule, T0=T0)
    y22 = v_mix_ph(p - d1, h - d2, fluid_data, mixing_rule=mixing_rule, T0=T0)
    y12 = v_mix_ph(p + d1, h - d2, fluid_data, mixing_rule=mixing_rule, T0=T0)
    y21 = v_mix_ph(p - d1, h + d2, fluid_data, mixing_rule=mixing_rule, T0=T0)
    return (y11 + y22 - y12 - y21) / (4 * d1 * d2)


def d2v_mix_dp_h_dfluid(p, h, fluid, fluid_data, mixing_rule=None, T0=None):
    d1 = 1e-1
    d2 = 1e-5
    origin_value = fluid_data[fluid]["mass_fraction"]
    fluid_data[fluid]["mass_fraction"] = origin_value + d2
    y11 = v_mix_ph(p + d1, h, fluid_data, mixing_rule=mixing_rule, T0=T0)
    fluid_data[fluid]["mass_fraction"] = origin_value - d2
    y22 = v_mix_ph(p - d1, h, fluid_data, mixing_rule=mixing_rule, T0=T0)
    y12 = v_mix_ph(p + d1, h, fluid_data, mixing_rule=mixing_rule, T0=T0)
    fluid_data[fluid]["mass_fraction"] = origin_value + d2
    y21 = v_mix_ph(p - d1, h, fluid_data, mixing_rule=mixing_rule, T0=T0)
    fluid_data[fluid]["mass_fraction"] = origin_value
    return (y11 + y22 - y12 - y21) / (4 * d1 * d2)


def d2v_mix_p_dh_dfluid(p, h, fluid, fluid_data, mixing_rule=None, T0=None):
    d1 = 1e-1
    d2 = 1e-5
    origin_value = fluid_data[fluid]["mass_fraction"]
    fluid_data[fluid]["mass_fraction"] = origin_value + d2
    y11 = v_mix_ph(p, h + d1, fluid_data, mixing_rule=mixing_rule, T0=T0)
    fluid_data[fluid]["mass_fraction"] = origin_value - d2
    y22 = v_mix_ph(p, h - d1, fluid_data, mixing_rule=mixing_rule, T0=T0)
    y12 = v_mix_ph(p, h + d1, fluid_data, mixing_rule=mixing_rule, T0=T0)
    fluid_data[fluid]["mass_fraction"] = origin_value + d2
    y21 = v_mix_ph(p, h - d1, fluid_data, mixing_rule=mixing_rule, T0=T0)
    fluid_data[fluid]["mass_fraction"] = origin_value
    return (y11 + y22 - y12 - y21) / (4 * d1 * d2)


def d2v_mix_ph_dfluid1_dfluid2(p, h, fluid1, fluid2, fluid_data, mixing_rule=None, T0=None):
    d1 = 1e-5
    d2 = 1e-5
    origin_value1 = fluid_data[fluid1]["mass_fraction"]
    origin_value2 = fluid_data[fluid2]["mass_fraction"]

    fluid_data[fluid1]["mass_fraction"] = origin_value1 + d1
    fluid_data[fluid2]["mass_fraction"] = origin_value2 + d2
    y11 = v_mix_ph(p, h, fluid_data, mixing_rule=mixing_rule, T0=T0)

    fluid_data[fluid1]["mass_fraction"] = origin_value1 - d1
    fluid_data[fluid2]["mass_fraction"] = origin_value2 - d2
    y22 = v_mix_ph(p, h, fluid_data, mixing_rule=mixing_rule, T0=T0)

    fluid_data[fluid1]["mass_fraction"] = origin_value1 + d1
    fluid_data[fluid2]["mass_fraction"] = origin_value2 - d2
    y12 = v_mix_ph(p, h, fluid_data, mixing_rule=mixing_rule, T0=T0)

    fluid_data[fluid1]["mass_fraction"] = origin_value1 - d1
    fluid_data[fluid2]["mass_fraction"] = origin_value2 + d2
    y21 = v_mix_ph(p, h, fluid_data, mixing_rule=mixing_rule, T0=T0)

    return (y11 + y22 - y12 - y21) / (4 * d1 * d2)


def v_mix_pT(p, T, fluid_data, mixing_rule=None):
    if get_number_of_fluids(fluid_data) == 1:
        pure_fluid = get_pure_fluid(fluid_data)
        return 1 / pure_fluid["wrapper"].d_pT(p, T)
    else:
        _check_mixing_rule(mixing_rule, V_MIX_PT_DIRECT, "specific volume")
        return V_MIX_PT_DIRECT[mixing_rule](p, T, fluid_data)


def viscosity_mix_ph(p, h, fluid_data, mixing_rule=None, T0=None):
    if get_number_of_fluids(fluid_data) == 1:
        pure_fluid = get_pure_fluid(fluid_data)
        return pure_fluid["wrapper"].viscosity_ph(p, h)
    else:
        T = T_mix_ph(p, h , fluid_data, mixing_rule, T0)
        return viscosity_mix_pT(p, T, fluid_data, mixing_rule)


def viscosity_mix_pT(p, T, fluid_data, mixing_rule=None):
    if get_number_of_fluids(fluid_data) == 1:
        pure_fluid = get_pure_fluid(fluid_data)
        return pure_fluid["wrapper"].viscosity_pT(p, T)
    else:
        _check_mixing_rule(mixing_rule, V_MIX_PT_DIRECT, "viscosity")
        return VISCOSITY_MIX_PT_DIRECT[mixing_rule](p, T, fluid_data)
