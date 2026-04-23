# -*- coding: utf-8

"""Module for free fluid property engine.
Aurora/tools/fluid_properties/free_fluid_engine.py

SPDX-License-Identifier: MIT
"""

from Aurora.tools.fluid_properties.properties_reference_data import MOLTEN_SALT_REFERENCE_DATA as salt_data
from Aurora.tools.fluid_properties.properties_reference_data import molar_mass_dict
from Aurora.tools.fluid_properties.properties_reference_data import atomic_masses
from Aurora.tools.fluid_properties.properties_reference_data import compound_groups_masses
from Aurora.tools.fluid_properties.properties_reference_data import Customized_Fluid
from Aurora.tools import logger


class FreeFluidEngine:
    """
    Supply properties for fluid composition defined by user.
    """
    def __init__(self, fluid):
        self.fluid = fluid
        if not self.fluid in Customized_Fluid.keys():
            msg = f'The fluid "{self.fluid}" is not defined in FreeFluidEngine.'
            raise ValueError(msg)
        self.__generate_fluid_properties_data__()

    def __generate_fluid_properties_data__(self):
        self.T_melt = Customized_Fluid[self.fluid]['T_melt']
        self.T_min = self.T_melt
        self.T_max = Customized_Fluid[self.fluid]['T_max']
        self.p_min = Customized_Fluid[self.fluid]['p_min']
        self.p_max = Customized_Fluid[self.fluid]['p_max']
        self.p_crit = Customized_Fluid[self.fluid]['p_crit']
        self.T_crit = Customized_Fluid[self.fluid]['T_crit']
        self.T_ref = Customized_Fluid[self.fluid]['reference_point']['T']
        self.h_ref = Customized_Fluid[self.fluid]['reference_point']['h']
        self.s_ref = Customized_Fluid[self.fluid]['reference_point']['s']
        self.molar_mass = self._calculate_molar_mass()

    def specific_heat(self, p, T):
        Cp = Customized_Fluid[self.fluid]['Cp'](p=p, T=T)
        return Cp

    def _calculate_molar_mass(self):
        """calculate molar mass of fluid"""
        fluid_name = self.fluid
        # be contained in dict
        if fluid_name in Customized_Fluid:
            return Customized_Fluid[fluid_name]["molar_mass"]
        # try to analysis chemical construction
        return self._parse_molar_mass_from_formula(fluid_name)

    @staticmethod
    def _parse_molar_mass_from_formula(formula):
        """calculate molar mass of fluid composition based on chemical formula"""
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
                    if element in compound_groups_masses:
                        count = 1 if count_str == '' else int(count_str)
                        molar_mass += compound_groups_masses[element] * count
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

    def enthalpy(self, p: float, T: float) -> float:
        """calculate enthalpy of fluid based on pressure and temperature"""
        h = Customized_Fluid[self.fluid]['enthalpy'](p=p, T=T)
        return h

    def entropy(self, p, T):
        """calculate entropy of fluid based on pressure and temperature"""
        s = Customized_Fluid[self.fluid]['entropy'](p=p, T=T)
        return s

    def density(self, p, T):
        """calculate density of fluid based on pressure and temperature"""
        d = Customized_Fluid[self.fluid]['density'](p=p, T=T)
        return d

    def phase(self, p, h):
        """calculate phase of fluid based on pressure and enthalpy"""
        pass

    def viscosity(self, p, T):
        """calculate viscosity of fluid based on pressure and temperature"""
        vis = Customized_Fluid[self.fluid]['viscosity'](p=p, T=T)
        return vis

    def T_ph(self, p, h):
        """calculate temperature of fluid based on pressure and enthalpy"""
        T = self.T_melt + 150
        dT = 1
        iter_ = 1
        fact = 0.1
        while True:
            delta_h = h - self.enthalpy(p, T)
            div = (self.enthalpy(p, T + dT) - self.enthalpy(p, T)) / dT
            alpha = min(abs(2 * fact * div / (delta_h + 1e-4)) ** 0.5, 1)
            T += alpha * delta_h / div
            iter_ += 1
            if abs(delta_h) < 1e1 or iter_ > 100:
                break
        return float(T)

    def T_ps(self, p, s):
        """calculate temperature of fluid based on pressure and entropy"""
        T = self.T_melt + 150
        dT = 1
        iter_ = 1
        fact = 0.1
        while True:
            delta_s = s - self.entropy(p, T)
            div = (self.entropy(p, T + dT) - self.entropy(p, T)) / dT
            alpha = min(abs(2 * fact * div / (delta_s + 1e-6)) ** 0.5, 1)
            T += alpha * delta_s / div
            iter_ += 1
            if abs(delta_s) < 0.01 or iter_ > 100:
                break
        return float(T)

    def h_pT(self, p, T):
        """calculate enthalpy of fluid based on pressure and temperature"""
        return self.enthalpy(p, T)

    def h_ps(self, p, s):
        """calculate enthalpy of fluid based on pressure and entropy"""
        T = self.T_ps(p, s)
        h = self.enthalpy(p, T)
        return h

    def h_pQ(self, p, Q):
        """calculate enthalpy of fluid based on pressure and dryness fraction"""
        T_sat = self.T_sat(p)
        return self.h_pT(p, T_sat)

    def h_QT(self, Q, T):
        """calculate enthalpy of fluid based on dryness fraction and temperature"""
        p = self.p_sat(T)
        return self.h_pT(p, T)

    def p_hT(self, h, T):
        """calculate pressure of fluid based on enthalpy and temperature"""
        p = 1e6
        dp = 1e1
        iter_ = 1
        fact = 1e4
        while True:
            delta_h = h - self.enthalpy(p, T)
            div = (self.enthalpy(p + dp, T) - self.enthalpy(p, T)) / dp
            alpha = min(abs(2 * fact * div / (delta_h + 1e-4)) ** 0.5, 1)
            p += alpha * delta_h / div
            iter_ += 1
            if abs(delta_h) < 1e1 or iter_ > 100:
                break
        return float(p)

    def p_hQ(self, h, Q):
        """calculate pressure of fluid based on enthalpy and dryness fraction"""
        pass

    def T_sat(self, p):
        """calculate saturated temperature of fluid based on pressure"""
        return self.T_crit

    def p_sat(self, T):
        """calculate saturated pressure of fluid based on temperature"""
        return self.p_crit

    def Q_ph(self, p, h):
        """calculate dryness fraction of fluid based on pressure and enthalpy"""
        pass





