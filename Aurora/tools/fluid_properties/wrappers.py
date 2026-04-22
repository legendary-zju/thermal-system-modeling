# -*- coding: utf-8

"""Module for fluid property wrappers.

It's copyrighted by the contributors recorded in the version control history of the file,
available from its original location
Aurora/tools/fluid_properties/wrappers.py

SPDX-License-Identifier: MIT
"""

import CoolProp as CP
import numpy as np
from .properties_reference_data import MOLTEN_SALT_REFERENCE_DATA as salt_data
from .properties_reference_data import molar_mass_dict
from .properties_reference_data import atomic_masses
from .properties_reference_data import compound_groups_masses as common_groups

from Aurora.tools.global_vars import ERR
from Aurora.tools import logger


def wrapper_registry(type):
    wrapper_registry.items[type.__name__] = type
    return type


wrapper_registry.items = {}


class SerializableAbstractState(CP.AbstractState):

    def __init__(self, back_end, fluid_name):
        self.back_end = back_end
        self.fluid_name = fluid_name

    def __reduce__(self):
        return (self.__class__, (self.back_end, self.fluid_name))


@wrapper_registry
class FluidPropertyWrapper:

    def __init__(self, fluid, back_end=None) -> None:
        """Base class for fluid property wrappers

        Parameters
        ----------
        fluid : str
            Name of the fluid.
        back_end : str, optional
            Name of the back end, by default None
        """
        self.back_end = back_end
        self.fluid = fluid
        self.mixture_type = None

        if "[" in self.fluid:
            if "|" not in self.fluid:
                msg = (
                    f"The fluid {self.fluid} requires the specification of "
                    "mass, volume or molar based composition information."
                    "You can do this by appending '|' and 'mass' at the end "
                    "of the fluid string. For example, "
                    "'NAMEOFFLUID[0.5]|mass' to indicate a mass based mixture."
                )
                raise ValueError(msg)

            self.fluid, self.mixture_type = self.fluid.split("|")
            allowed = ["mass", "molar", "volume"]
            if self.mixture_type not in allowed:
                msg = (
                    "For the specification of the composition type you have "
                    f"to select from {', '.join(allowed)}."
                )

        if "&" in self.fluid:
            _fluids_with_fractions = self.fluid.split("&")
        else:
            _fluids_with_fractions = [self.fluid]

        fluid_names = []
        fractions = []
        for fluid in _fluids_with_fractions:
            if "[" in fluid:
                _fluid_name, _fraction = fluid.split("[")
                _fraction = float(_fraction.replace("]", ""))
                fractions += [_fraction]
            else:
                _fluid_name = fluid
            fluid_names += [_fluid_name]

        self.fractions = fractions
        self.fluid = "&".join(fluid_names)

    def _not_implemented(self) -> None:
        raise NotImplementedError(
            f"Method is not implemented for {self.__class__.__name__}."
        )

    def isentropic(self, p_1, h_1, p_2):
        self._not_implemented()

    def _is_below_T_critical(self, T):
        self._not_implemented()

    def _make_p_subcritical(self, p):
        self._not_implemented()

    def T_ph(self, p, h):
        self._not_implemented()

    def T_ps(self, p, s):
        self._not_implemented()

    def h_pT(self, p, T):
        self._not_implemented()

    def h_QT(self, Q, T):
        self._not_implemented()

    def p_hT(self, h, T):
        self._not_implemented()

    def p_hQ(self, h, Q):
        self._not_implemented()

    def s_QT(self, Q, T):
        self._not_implemented()

    def T_sat(self, p):
        self._not_implemented()

    def p_sat(self, T):
        self._not_implemented()

    def Q_ph(self, p, h):
        self._not_implemented()

    def phase_ph(self, p, h):
        self._not_implemented()

    def d_ph(self, p, h):
        self._not_implemented()

    def d_pT(self, p, T):
        self._not_implemented()

    def d_QT(self, Q, T):
        self._not_implemented()

    def viscosity_ph(self, p, h):
        self._not_implemented()

    def viscosity_pT(self, p, T):
        self._not_implemented()

    def s_ph(self, p, h):
        self._not_implemented()

    def s_pT(self, p, T):
        self._not_implemented()


@wrapper_registry
class CoolPropWrapper(FluidPropertyWrapper):

    def __init__(self, fluid, back_end=None) -> None:
        """Wrapper for CoolProp.CoolProp.AbstractState instance calls

        Parameters
        ----------
        fluid : str
            Name of the fluid
        back_end : str, optional
            CoolProp back end for the AbstractState object, by default "HEOS"
        """
        if back_end is None:
            back_end = "HEOS"

        super().__init__(fluid, back_end)
        self.AS = SerializableAbstractState(self.back_end, self.fluid)
        self._set_constants()

    def _set_constants(self):
        if self.mixture_type == "mass":
            self.AS.set_mass_fractions(self.fractions)
        elif self.mixture_type == "molar":
            self.AS.set_molar_fractions(self.fractions)
        elif self.mixture_type == "volume":
            self.AS.set_volu_fractions(self.fractions)

        self._T_min = self.AS.trivial_keyed_output(CP.iT_min)
        self._T_max = self.AS.trivial_keyed_output(CP.iT_max)
        try:
            self._aliases = CP.CoolProp.get_aliases(self.fluid)
        except RuntimeError:
            self._aliases = [self.fluid]

        if self.back_end == "INCOMP":
            self._p_min = 1e2
            self._p_max = 1e8
            self._p_crit = 1e8
            self._T_crit = None
            self._molar_mass = 1
            try:
                # how to know that we have a binary mixture?
                self._T_min = self.AS.trivial_keyed_output(CP.iT_freeze)
            except ValueError:
                pass
        else:
            self._p_min = self.AS.trivial_keyed_output(CP.iP_min)
            self._p_max = self.AS.trivial_keyed_output(CP.iP_max)
            self._p_crit = self.AS.trivial_keyed_output(CP.iP_critical)
            self._T_crit = self.AS.trivial_keyed_output(CP.iT_critical)
            self._molar_mass = self.AS.trivial_keyed_output(CP.imolar_mass)

    def _is_below_T_critical(self, T):
        return T < self._T_crit

    def _make_p_subcritical(self, p):
        if p > self._p_crit:
            p = self._p_crit * 0.99
        return p

    def get_T_max(self, p):
        if self.back_end == "INCOMP":
            return self.T_sat(p)
        else:
            return self._T_max

    def isentropic(self, p_1, h_1, p_2):
        return self.h_ps(p_2, self.s_ph(p_1, h_1))

    def T_ph(self, p, h):
        self.AS.update(CP.HmassP_INPUTS, h, p)
        return self.AS.T()

    def T_ps(self, p, s):
        self.AS.update(CP.PSmass_INPUTS, p, s)
        return self.AS.T()

    def h_pQ(self, p, Q):
        self.AS.update(CP.PQ_INPUTS, p, Q)
        return self.AS.hmass()

    def h_ps(self, p, s):
        self.AS.update(CP.PSmass_INPUTS, p, s)
        return self.AS.hmass()

    def h_pT(self, p, T):
        self.AS.update(CP.PT_INPUTS, p, T)
        return self.AS.hmass()

    def h_QT(self, Q, T):
        self.AS.update(CP.QT_INPUTS, Q, T)
        return self.AS.hmass()

    def p_hT(self, h, T):
        self.AS.update(CP.HmassT_INPUTS, h, T)
        return self.AS.p()

    def p_hQ(self, h, Q):
        self.AS.update(CP.HmassQ_INPUTS, h, Q)
        return self.AS.p()

    def s_QT(self, Q, T):
        self.AS.update(CP.QT_INPUTS, Q, T)
        return self.AS.smass()

    def T_sat(self, p):
        p = self._make_p_subcritical(p)
        self.AS.update(CP.PQ_INPUTS, p, 0)
        return self.AS.T()

    def p_sat(self, T):
        if T > self._T_crit:
            T = self._T_crit * 0.99

        self.AS.update(CP.QT_INPUTS, 0, T)
        return self.AS.p()

    def Q_ph(self, p, h):
        p = self._make_p_subcritical(p)
        self.AS.update(CP.HmassP_INPUTS, h, p)

        if self.AS.phase() == CP.iphase_twophase:
            return self.AS.Q()
        elif self.AS.phase() == CP.iphase_liquid:
            return 0
        elif self.AS.phase() == CP.iphase_gas:
            return 1
        else:  # all other phases - though this should be unreachable as p is sub-critical
            return -1

    def phase_ph(self, p, h):
        p = self._make_p_subcritical(p)
        self.AS.update(CP.HmassP_INPUTS, h, p)

        if self.AS.phase() == CP.iphase_twophase:
            return "tp"
        elif self.AS.phase() == CP.iphase_liquid:
            return "l"
        elif self.AS.phase() == CP.iphase_gas:
            return "g"
        else:  # all other phases - though this should be unreachable as p is sub-critical
            return "state not recognised"

    def d_ph(self, p, h):
        self.AS.update(CP.HmassP_INPUTS, h, p)
        return self.AS.rhomass()

    def d_pT(self, p, T):
        self.AS.update(CP.PT_INPUTS, p, T)
        return self.AS.rhomass()

    def d_QT(self, Q, T):
        self.AS.update(CP.QT_INPUTS, Q, T)
        return self.AS.rhomass()

    def viscosity_ph(self, p, h):
        self.AS.update(CP.HmassP_INPUTS, h, p)
        return self.AS.viscosity()

    def viscosity_pT(self, p, T):
        self.AS.update(CP.PT_INPUTS, p, T)
        return self.AS.viscosity()

    def s_ph(self, p, h):
        self.AS.update(CP.HmassP_INPUTS, h, p)
        return self.AS.smass()

    def s_pT(self, p, T):
        self.AS.update(CP.PT_INPUTS, p, T)
        return self.AS.smass()


@wrapper_registry
class MoltenSaltWrapper(FluidPropertyWrapper):
    def __init__(self, fluid, back_end=None) -> None:
        """Wrapper for fuse salt, which is used for heat storage tank.

        Parameters
        ----------
        fluid : str
            Name of the fluid
        back_end : str, optional
            Salt back end for the AbstractState object, by default "default"
        """
        try:
            from salts import Salt
            self.Salt = Salt
        except ImportError:
            raise ModuleNotFoundError(
                "To use molten salt properties, you need to install the 'salts' package. "
                "Run: pip install salts"
            )
        if back_end is None:
            back_end = "default"
        super().__init__(fluid, back_end)
        # check the salt is contained
        try:
            self.salt_obj = self.Salt(fluid)
        except ValueError as e:
            available_salts = list(salt_data.keys())
            raise ValueError(
                f"Fluid '{fluid}' not found in Salts library. "
                f"Available salts: {', '.join(available_salts)}"
            ) from e
        # set reference enthalpy of salt
        self._setup_reference_enthalpy()
        # set limited properties
        self._set_constants()
        # enthalpy calculation preprocess
        self._setup_enthalpy_interpolation()

    def _setup_reference_enthalpy(self):
        """set reference enthalpy for salt"""
        salt_name = self.fluid
        # check whether salt has been set
        if salt_name in salt_data:
            ref_data = salt_data[salt_name]
            self._T_ref = ref_data["T_ref"]  # K
            self._h_ref = ref_data["h_ref"]  # J/kg
        else:
            try:
                self._T_ref = self.salt_obj.T_melt  # molten temperature supplied by Salts
                self._h_ref = 0.0
                logger.warning(f"No reference enthalpy found for {salt_name}. "
                               f"Using melting point {self._T_ref:.2f}K with h=0 J/kg")
            except AttributeError:
                self._T_ref = 500.0  # default reference temperature
                self._h_ref = 0.0
                logger.warning(f"Using default reference temperature {self._T_ref}K for {salt_name}, due to molten temperature could not be found.")

    def _set_constants(self):
        """set constant limited properties for salt"""
        # set temperature range
        self._T_min = self._T_ref  # set reference temperature as lower limit
        self._T_max = 1200.0  #
        # set pressure range
        self._p_min = 0.0  # 0 Pa
        self._p_max = 100e6  # 100 MPa
        # salt has no crit properties
        self._p_crit = 50e6  # 50 MPa
        self._T_crit = 2000.0  # 2000K
        # molar mass of salt
        self._molar_mass = self._calculate_molar_mass()

    def _calculate_molar_mass(self):
        """calculate molar mass of salt"""
        fluid_name = self.fluid
        # be contained in dict
        if fluid_name in molar_mass_dict:
            return molar_mass_dict[fluid_name]
        # try to analysis chemical construction
        return self._parse_molar_mass_from_formula(fluid_name)

    def _parse_molar_mass_from_formula(self, formula):
        """calculate molar mass of salt based on chemical formula"""
        try:
            # simple analysis of chemical constructure
            import re
            # match elements and quantities
            pattern = r'([A-Z][a-z]*)(\d*)'
            matches = re.findall(pattern, formula)
            molar_mass = 0.0
            for element, count_str in matches:
                if element in atomic_masses:
                    # atomic
                    count = 1 if count_str == '' else int(count_str)
                    molar_mass += atomic_masses[element] * count
                else:
                    # compound groups
                    if element in common_groups:
                        count = 1 if count_str == '' else int(count_str)
                        molar_mass += common_groups[element] * count
                    else:
                        # unknown element/group
                        msg = f"{element} has not been identified at formula {formula}"
                        logger.warning(msg)
                        return 0.085  # default value
            # convert unit
            return molar_mass / 1000.0
        # error
        except Exception as e:
            msg = f"{formula} has not been analysis, due to {e}"
            logger.error(msg)
            return 0.085  # default value

    def _setup_enthalpy_interpolation(self):
        """set enthalpy interpolation properties for salt"""
        # generate temperature range
        T_min_interp = max(self._T_min - 50, 273.15)  # above 0°C
        T_max_interp = self._T_max
        # generate temperature reference points
        n_points = 500
        # log distribution
        T_array = np.logspace(np.log10(T_min_interp), np.log10(T_max_interp), n_points)
        # set enthalpy list
        h_array = np.zeros_like(T_array)
        # calculate enthalpy by integration
        for i, T in enumerate(T_array):
            if i == 0:
                # first point is reference enthalpy
                h_array[i] = self._h_ref
                continue
            # integration：Δh = 0.5 * (cp(T_i) + cp(T_{i-1})) * (T_i - T_{i-1})
            cp_i = self.salt_obj.specific_heat(T)
            cp_prev = self.salt_obj.specific_heat(T_array[i - 1])
            delta_h = 0.5 * (cp_i + cp_prev) * (T - T_array[i - 1])
            h_array[i] = h_array[i - 1] + delta_h
        #
        # contain interpolation list
        self._T_array = T_array
        self._h_array = h_array
        # generate interpolation function
        from scipy import interpolate
        self._h_from_T = interpolate.interp1d(
            T_array, h_array, kind='cubic',
            bounds_error=False, fill_value='extrapolate'
        )
        self._T_from_h = interpolate.interp1d(
            h_array, T_array, kind='cubic',
            bounds_error=False, fill_value='extrapolate'
        )

    def _calculate_enthalpy(self, T: float) -> float:
        """calculate enthalpy for salt based on temperature"""
        return float(self._h_from_T(T))

    def _get_temperature_from_enthalpy(self, h: float) -> float:
        """calculate temperature for salt based on enthalpy"""
        return float(self._T_from_h(h))

    def _is_below_T_critical(self, T: float) -> bool:
        """check if temperature is below critical temperature"""
        return T < self._T_crit

    def _make_p_subcritical(self, p: float) -> float:
        """guarantee that pressure is below critical pressure"""
        if p > self._p_crit:
            p = self._p_crit * 0.99
        return p

    def isentropic(self, p_1: float, h_1: float, p_2: float) -> float:
        """The isentropic process is approximated as an isothermal process"""
        T1 = self._get_temperature_from_enthalpy(h_1)
        return self.h_pT(p_2, T1)

    def T_ph(self, p: float, h: float) -> float:
        """calculate temperature for salt based on enthalpy"""
        # ignore the impact of pressure
        return self._get_temperature_from_enthalpy(h)

    def T_ps(self, p: float, s: float) -> float:
        """calculate temperature for salt based on entropy"""
        # s = cp * ln(T/T_ref) approximate
        T0 = self._T_ref + 100
        dT = 1
        iter_ = 1
        fact = 0.1
        while True:
            delta_s = s - self.s_pT(p, T0)
            div = (self.s_pT(p, T0 + dT) - self.s_pT(p, T0)) / dT
            alpha = min((2 * fact * div / delta_s) ** 0.5, 1)
            T0 += alpha * delta_s / div
            iter_ += 1
            if abs(T0 - T0) < 0.01 or iter_ > 30:
                break
        return float(T0)

    def h_pT(self, p: float, T: float) -> float:
        """calculate enthalpy for salt based on temperature"""
        # ignore the impact of pressure
        return self._calculate_enthalpy(T)

    def h_ps(self, p: float, s: float) -> float:
        """calculate enthalpy for salt based on entropy"""
        T = self.T_ps(p, s)
        return self.h_pT(p, T)

    def h_pQ(self, p: float, Q: float) -> float:
        """calculate enthalpy for salt based on pressure and dryness fraction"""
        # return saturated liquid enthalpy
        T_sat = self.T_sat(p)
        return self.h_pT(p, T_sat)

    def h_QT(self, Q: float, T: float) -> float:
        """calculate enthalpy for salt based on temperature and dryness fraction"""
        # ignore dryness fraction
        return self._calculate_enthalpy(T)

    def p_hT(self, h: float, T: float) -> float:
        """calculate pressure for salt based on temperature and enthalpy"""
        msg = f"The pressure of salt is not determined by temperature and enthalpy"
        raise ValueError(msg)

    def p_hQ(self, h: float, Q: float) -> float:
        """calculate pressure for salt based on enthalpy and dryness fraction"""
        msg = f"The pressure of salt is not determined by enthalpy and dryness fraction"
        raise ValueError(msg)

    def s_QT(self, Q: float, T: float) -> float:
        """calculate entropy for salt based on temperature and dryness fraction"""
        # s = cp_avg * ln(T/T_ref)
        T_ref = self._T_ref
        cp = self.salt_obj.specific_heat(T)
        s = cp * np.log(T / T_ref)
        return s

    def T_sat(self, p: float) -> float:
        """calculate saturated temperature for salt based on pressure"""
        # salt has no boiling phenomenon
        return self._T_max

    def p_sat(self, T: float) -> float:
        """saturated pressure for salt based on temperature"""
        # approximate 0
        if T > self._T_max:
            T = self._T_max * 0.99
        return 100.0  # 100 Pa

    def Q_ph(self, p: float, h: float) -> float:
        """calculate dryness fraction for salt based on pressure and enthalpy"""
        # liquid constantly
        return 0

    def phase_ph(self, p: float, h: float) -> str:
        """judge phase for salt based on pressure and enthalpy"""
        T = self._get_temperature_from_enthalpy(h)
        if T < self._T_ref:
            return "s"  # solid
        else:
            return "l"  # liquid

    def d_ph(self, p: float, h: float) -> float:
        """calculate density for salt based on pressure and enthalpy"""
        # ignore the pressure impact
        T = self._get_temperature_from_enthalpy(h)
        return self.d_pT(p, T)

    def d_pT(self, p: float, T: float) -> float:
        """calculate density for salt based on pressure and temperature"""
        # ignore the impact of pressure
        return self.salt_obj.density(T)

    def d_QT(self, Q: float, T: float) -> float:
        """calculate density for salt based on temperature and dryness fraction"""
        # ignore dryness fraction
        return self.salt_obj.density(T)

    def viscosity_ph(self, p: float, h: float) -> float:
        """calculate viscosity for salt based on pressure and enthalpy"""
        # ignore pressure
        T = self._get_temperature_from_enthalpy(h)
        return self.viscosity_pT(p, T)

    def viscosity_pT(self, p: float, T: float) -> float:
        """calculate viscosity for salt based on pressure and temperature"""
        # ignore pressure
        return self.salt_obj.dynamic_viscosity(T)

    def s_ph(self, p: float, h: float) -> float:
        """通过压力和焓值计算熵（近似）"""
        T = self._get_temperature_from_enthalpy(h)
        return self.s_pT(p, T)

    def s_pT(self, p: float, T: float) -> float:
        """calculate entropy for salt based on pressure and temperature"""
        # s = ∫(cp/T)dT ≈ cp_avg * ln(T/T_ref)
        T_ref = self._T_ref
        cp = self.salt_obj.specific_heat(T)
        # log distribution
        if T > T_ref:
            # integration
            n_points = max(int(T - T_ref), 1)
            T_range = np.linspace(T_ref, T, n_points)
            cp_range = np.array([self.salt_obj.specific_heat(t) for t in T_range])
            s = np.trapz(cp_range / T_range, T_range, dx=T_range[1] - T_range[0])
        else:
            s = cp * np.log(T / T_ref)
        return s

    def thermal_conductivity_pT(self, p: float, T: float) -> float:
        """calculate thermal conductivity for salt based on temperature"""
        return self.salt_obj.thermal_conductivity(T)

    def thermal_conductivity_ph(self, p: float, h: float) -> float:
        """calculate thermal conductivity for salt based on enthalpy"""
        T = self._get_temperature_from_enthalpy(h)
        return self.thermal_conductivity_pT(p, T)

    def specific_heat_pT(self, p: float, T: float) -> float:
        """calculate specific heat for salt based on pressure and temperature"""
        return self.salt_obj.specific_heat(T)

    def specific_heat_ph(self, p: float, h: float) -> float:
        """calculate specific heat for salt based on enthalpy"""
        T = self._get_temperature_from_enthalpy(h)
        return self.specific_heat_pT(p, T)


@wrapper_registry
class IAPWSWrapper(FluidPropertyWrapper):

    def __init__(self, fluid, back_end=None) -> None:
        """Wrapper for iapws library calls

        Parameters
        ----------
        fluid : str
            Name of the fluid
        back_end : str, optional
            IAPWS back end for the AbstractState object, by default "IF97"
        """
        # avoid unncessary loading time if not used
        try:
            import iapws
        except ModuleNotFoundError:
            msg = (
                "To use the iapws fluid properties you need to install "
                "iapws."
            )
            raise ModuleNotFoundError(msg)

        if back_end is None:
            back_end = "IF97"
        super().__init__(fluid, back_end)
        self._aliases = CP.CoolProp.get_aliases("H2O")

        if self.fluid not in self._aliases:
            msg = "The iapws wrapper only supports water as fluid."
            raise ValueError(msg)

        if self.back_end == "IF97":
            self.AS = iapws.IAPWS97
        elif self.back_end == "IF95":
            self.AS = iapws.IAPWS95
        else:
            msg = f"The specified back_end {self.back_end} is not available."
            raise NotImplementedError(msg)
        self._set_constants(iapws)

    def _set_constants(self, iapws):
        self._T_min = iapws._iapws.Tt
        self._T_max = 2000
        self._p_min = iapws._iapws.Pt * 1e6
        self._p_max = 100e6
        self._p_crit = iapws._iapws.Pc * 1e6
        self._T_crit = iapws._iapws.Tc
        self._molar_mass = iapws._iapws.M

    def _is_below_T_critical(self, T):
        return T < self._T_crit

    def _make_p_subcritical(self, p):
        if p > self._p_crit:
            p = self._p_crit * 0.99
        return p

    def isentropic(self, p_1, h_1, p_2):
        return self.h_ps(p_2, self.s_ph(p_1, h_1))

    def T_ph(self, p, h):
        return self.AS(h=h / 1e3, P=p / 1e6).T

    def T_ps(self, p, s):
        return self.AS(s=s / 1e3, P=p / 1e6).T

    def h_pQ(self, p, Q):
        return self.AS(P=p / 1e6, x=Q).h * 1e3

    def h_ps(self, p, s):
        return self.AS(P=p / 1e6, s=s / 1e3).h * 1e3

    def h_pT(self, p, T):
        return self.AS(P=p / 1e6, T=T).h * 1e3

    def h_QT(self, Q, T):
        return self.AS(T=T, x=Q).h * 1e3

    def p_hT(self, h, T):
        return self.AS(h=h / 1e3, T=T).p * 1e6

    def p_hQ(self, h, Q):
        return self.AS(h=h / 1e3, x=Q).p * 1e6

    def s_QT(self, Q, T):
        return self.AS(T=T, x=Q).s * 1e3

    def T_sat(self, p):
        p = self._make_p_subcritical(p)
        return self.AS(P=p / 1e6, x=0).T

    def p_sat(self, T):
        if T > self._T_crit:
            T = self._T_crit * 0.99
        # !!!!!!
        return self.AS(T=T, x=0).P * 1e6

    def Q_ph(self, p, h):
        p = self._make_p_subcritical(p)
        return self.AS(h=h / 1e3, P=p / 1e6).x

    def phase_ph(self, p, h):
        p = self._make_p_subcritical(p)

        phase = self.AS(h=h / 1e3, P=p / 1e6).phase

        if phase in ["Liquid"]:
            return "l"
        elif phase in  ["Vapour"]:
            return "g"
        elif phase in ["Two phases", "Saturated vapor", "Saturated liquid"]:
            return "tp"
        else:  # to ensure consistent behaviour to CoolPropWrapper
            return "phase not recognised"

    def d_ph(self, p, h):
        return self.AS(h=h / 1e3, P=p / 1e6).rho

    def d_pT(self, p, T):
        return self.AS(T=T, P=p / 1e6).rho

    def d_QT(self, Q, T):
        return self.AS(T=T, x=Q).rho

    def viscosity_ph(self, p, h):
        return self.AS(P=p / 1e6, h=h / 1e3).mu

    def viscosity_pT(self, p, T):
        return self.AS(T=T, P=p / 1e6).mu

    def s_ph(self, p, h):
        return self.AS(P=p / 1e6, h=h / 1e3).s * 1e3

    def s_pT(self, p, T):
        return self.AS(P=p / 1e6, T=T).s * 1e3


@wrapper_registry
class PyromatWrapper(FluidPropertyWrapper):

    def __init__(self, fluid, back_end=None) -> None:
        """_summary_

        Parameters
        ----------
        fluid : str
            Name of the fluid
        back_end : str, optional
            CoolProp back end for the AbstractState object, by default None
        """
        # avoid unnecessary loading time if not used
        try:
            import pyromat as pm
            pm.config['unit_energy'] = "J"
            pm.config['unit_pressure'] = "Pa"
            pm.config['unit_molar'] = "mol"
        except ModuleNotFoundError:
            msg = (
                "To use the pyromat fluid properties you need to install "
                "pyromat."
            )
            raise ModuleNotFoundError(msg)

        super().__init__(fluid, back_end)
        self._create_AS(pm)
        self._set_constants()

    def _create_AS(self, pm):
        self.AS = pm.get(f"{self.back_end}.{self.fluid}")

    def _set_constants(self):
        self._p_min, self._p_max = 100, 1000e5
        self._T_min, self._T_max = self.AS.Tlim()
        self._molar_mass = self.AS.mw()

    def isentropic(self, p_1, h_1, p_2):
        return self.h_ps(p_2, self.s_ph(p_1, h_1))

    def T_ph(self, p, h):
        return self.AS.T(p=p, h=h)[0]

    def T_ps(self, p, s):
        return self.AS.T(p=p, s=s)[0]

    def h_pT(self, p, T):
        return self.AS.h(p=p, T=T)[0]

    def T_ph(self, p, h):
        return self.AS.T(p=p, h=h)[0]

    def T_ps(self, p, s):
        return self.AS.T(p=p, s=s)[0]

    def h_pT(self, p, T):
        return self.AS.h(p=p, T=T)[0]

    def h_ps(self, p, s):
        return self.AS.h(p=p, s=s)[0]

    def p_hT(self, h, T):
        return self.AS.p(h=h, T=T)[0]

    def p_hQ(self, h, Q):
        return self.AS.p(h=h, x=Q)[0]

    def d_ph(self, p, h):
        return self.AS.d(p=p, h=h)[0]

    def d_pT(self, p, T):
        return self.AS.d(p=p, T=T)[0]

    def s_ph(self, p, h):
        return self.AS.s(p=p, h=h)[0]

    def s_pT(self, p, T):
        if self.back_end == "ig":
            self._not_implemented()
        return self.AS.s(p=p, T=T)[0]

    def h_QT(self, Q, T):
        if self.back_end == "ig":
            self._not_implemented()
        return self.AS.h(x=Q, T=T)[0]

    def s_QT(self, Q, T):
        if self.back_end == "ig":
            self._not_implemented()
        return self.AS.s(x=Q, T=T)[0]

    def T_boiling(self, p):
        if self.back_end == "ig":
            self._not_implemented()
        return self.AS.T(x=1, p=p)[0]

    def p_boiling(self, T):
        if self.back_end == "ig":
            self._not_implemented()
        return self.AS.p(x=1, T=T)[0]

    def Q_ph(self, p, h):
        if self.back_end == "ig":
            self._not_implemented()
        return self.AS.x(p=p, h=h)[0]

    def d_QT(self, Q, T):
        if self.back_end == "ig":
            self._not_implemented()
        return self.AS.d(x=Q, T=T)[0]
