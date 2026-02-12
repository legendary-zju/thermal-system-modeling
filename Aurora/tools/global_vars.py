# -*- coding: utf-8

"""Module for global variables used by other modules of the Aurora package.

This file is part of project TESPy (github.com/oemof/Aurora). It's copyrighted
by the contributors recorded in the version control history of the file,
available from its original location Aurora/tools/global_vars.py

SPDX-License-Identifier: MIT
"""
import math

ERR = 1e-6
min_derive = 1e-6
molar_masses = {}
gas_constants = {}
gas_constants['uni'] = 8.314462618

fluid_property_data = {
    'm': {
        'text': 'mass flow',
        'SI_unit': 'kg / s',
        'units': {
            'kg / s': 1, 'kg / min': 1 / 60, 'kg / h': 1 / 3.6e3,
            't / h': 1 / 3.6, 'g / s': 1 / 1e3
        },
        'type': 'ratio',
        'equal': 'm',
        'differ': 1e-2,
        'latex_eq': r'0 = \dot{m} - \dot{m}_\mathrm{spec}',
        'documentation': {'float_fmt': '{:,.3f}'}
    },
    'p': {
        'text': 'pressure',
        'SI_unit': 'Pa',
        'units': {
            'Pa': 1, 'kPa': 1e3, 'psi': 6.8948e3,
            'bar': 1e5, 'atm': 1.01325e5, 'MPa': 1e6
        },
        'type': 'ratio',
        'equal': 'p',
        'differ': 1e-1,
        'latex_eq': r'0 = p - p_\mathrm{spec}',
        'documentation': {'float_fmt': '{:,.3f}'}
    },
    'h': {
        'text': 'enthalpy',
        'SI_unit': 'J / kg',
        'units': {
            'J / kg': 1, 'kJ / kg': 1e3, 'MJ / kg': 1e6,
            'cal / kg': 4.184, 'kcal / kg': 4.184e3,
            'Wh / kg': 3.6e3, 'kWh / kg': 3.6e6
        },
        'type': 'ratio',
        'equal': 'h',
        'differ': 1e-1,
        'latex_eq': r'0 = h - h_\mathrm{spec}',
        'documentation': {'float_fmt': '{:,.3f}'}
    },
    'T': {
        'text': 'temperature',
        'SI_unit': 'K',
        'units': {
            'K': [0, 1], 'R': [0, 5 / 9],
            'C': [273.15, 1], 'F': [459.67, 5 / 9]
        },
        'type': 'linear',
        'equal': 'T',
        'differ': 1e-4,
        'latex_eq': r'0 = T \left(p, h \right) - T_\mathrm{spec}',
        'documentation': {'float_fmt': '{:,.1f}'}
    },
    'Td_bp': {
        'text': 'temperature difference above boiling point',
        'SI_unit': 'K',
        'units': {
            'K': 1, 'R': 5 / 9, 'C': 1, 'F': 5 / 9
        },
        'type': 'ratio',
        'equal': 'T',
        'differ': 1e-4,
        'latex_eq': r'0 = \Delta T_\mathrm{spec}- T_\mathrm{sat}\left(p\right)',
        'documentation': {'float_fmt': '{:,.1f}'}
    },
    'Td_dew': {
        'text': 'temperature difference under boiling point',
        'SI_unit': 'K',
        'units': {
            'K': 1, 'R': 5 / 9, 'C': 1, 'F': 5 / 9
        },
        'type': 'ratio',
        'equal': 'T',
        'differ': 1e-4,
        'latex_eq': r'0 = T_\mathrm{sat}\left(p\right) - \Delta T_\mathrm{spec}',
        'documentation': {'float_fmt': '{:,.1f}'}
    },
    'v': {
        'text': 'volumetric flow',
        'SI_unit': 'm^3 / s',
        'units': {
            'm^3 / s': 1, 'm^3 / min': 1 / 60, 'm^3 / h': 1 / 3.6e3,
            'l / s': 1 / 1e3, 'l / min': 1 / 60e3, 'l / h': 1 / 3.6e6
        },
        'type': 'ratio',
        'equal': '',
        'differ': 1e-4,
        'latex_eq': r'0 = \dot{m} \cdot v \left(p,h\right)- \dot{V}_\mathrm{spec}',
        'documentation': {'float_fmt': '{:,.3f}'}
    },
    'vol': {
        'text': 'specific volume',
        'SI_unit': 'm^3 / kg',
        'units': {
            'm^3 / kg': 1, 'l / kg': 1e-3
        },
        'type': 'ratio',
        'equal': '',
        'differ': 1e-4,
        'latex_eq': (
            r'0 = v\left(p,h\right) \cdot \dot{m} - \dot{V}_\mathrm{spec}'),
        'documentation': {'float_fmt': '{:,.3f}'}
    },
    'x': {
        'text': 'vapor mass fraction',
        'SI_unit': '-',
        'units': {
            '-': 1, '%': 1e-2, 'ppm': 1e-6
        },
        'type': 'ratio',
        'equal': '',
        'differ': 1e-4,
        'latex_eq': r'0 = h - h\left(p, x_\mathrm{spec}\right)',
        'documentation': {'float_fmt': '{:,.2f}'}
    },
    's': {
        'text': 'entropy',
        'SI_unit': 'J / kgK',
        'units': {
            'J / kgK': 1, 'kJ / kgK': 1e3, 'MJ / kgK': 1e6
        },
        'type': 'ratio',
        'equal': '',
        'differ': 1e-4,
        'latex_eq': r'0 = s_\mathrm{spec} - s\left(p, h \right)',
        'documentation': {'float_fmt': '{:,.2f}'}
    }
}

electromagnetic_property_data = {
    'U': {
        'text': 'voltage',
        'SI_unit': 'V',
        'units': {
            'V': 1, 'mV': 1e-3, 'uV': 1e-6, 'nV': 1e-9,
            'kV': 1e3, 'MV': 1e6, 'GV': 1e9
        },
        'type': 'ratio',
        'equal': '',
        'differ': 1e-4,
        'latex_eq': r'0 = U - U_\mathrm{spec}',
        'documentation': {'float_fmt': '{:,.3f}'}
    },
    'I': {
        'text': 'electricity',
        'SI_unit': 'A',
        'units': {
            'A': 1, 'mA': 1e-3, 'uA': 1e-6, 'nA': 1e-9,
            'kA': 1e3, 'MA': 1e6, 'GA': 1e9
        },
        'type': 'ratio',
        'equal': '',
        'differ': 1e-4,
        'latex_eq': r'0 = I - I_\mathrm{spec}',
        'documentation': {'float_fmt': '{:,.3f}'}
    },
    'Phi': {
        'text': 'electric phase',
        'SI_unit': 'rad',
        'units': {
            'rad': 1, 'deg': math.pi/180, 'grad': math.pi/200, 'turn': 2 * math.pi
        },
        'type': 'ratio',
        'equal': '',
        'differ': 1e-4,
        'latex_eq': r'0 = Phi - Phi_\mathrm{spec}',
        'documentation': {'float_fmt': '{:,.3f}'}
    },
    'f': {
        'text': 'frequency',
        'SI_unit': 'Hz',
        'units': {
            'Hz': 1, 'kHz': 1e3, 'MHz': 1e6, 'GHz': 1e9
        },
        'type': 'ratio',
        'equal': '',
        'differ': 1e-4,
        'latex_eq': r'0 = f - f_\mathrm{spec}',
        'documentation': {'float_fmt': '{:,.3f}'}
    },

}

component_property_data = {
    'Q': {
        'text': 'thermal power',
        'SI_unit': 'W',
        'units': {
            'W': 1, 'mW': 1e-3, 'kW': 1e3,
            'MW': 1e6, 'GW': 1e9
        },
        'type': 'ratio',
        'equal': 'P',
        'differ': 1e-1,
        'latex_eq': r'0 = \dot{Q} - \dot{Q}_\mathrm{spec}',
        'documentation': {'float_fmt': '{:,.3f}'}
    },
    'P': {
        'text': 'power',
        'SI_unit': 'W',
        'units': {
            'W': 1, 'mW': 1e-3, 'kW': 1e3,
            'MW': 1e6, 'GW': 1e9
        },
        'type': 'ratio',
        'equal': 'Q',
        'differ': 1e-1,
        'latex_eq': r'0 = P - P_\mathrm{spec}',
        'documentation': {'float_fmt': '{:,.3f}'}
    },
    'DT': {
        'text': 'temperature difference',
        'SI_unit': 'K',
        'units': {
            'K': 1, 'R': 5 / 9, 'C': 1, 'F': 5 / 9
        },
        'type': 'ratio',
        'equal': 'T',
        'differ': 1e-4,
        'latex_eq': r'0 = \Delta T_\mathrm{spec}',
        'documentation': {'float_fmt': '{:,.1f}'}
    },
    'dp': {
        'text': 'pressure difference',
        'SI_unit': 'Pa',
        'units': {
            'Pa': 1, 'kPa': 1e3, 'psi': 6.8948e3,
            'bar': 1e5, 'atm': 1.01325e5, 'MPa': 1e6
        },
        'type': 'ratio',
        'equal': 'p',
        'differ': 1e-1,
        'latex_eq': r'0 = dp - dp_\mathrm{spec}',
        'documentation': {'float_fmt': '{:,.3f}'}
    },
    'kA': {
        'text': 'heat exchange rate',
        'SI_unit': 'W/K',
        'units': {
            'W/K': 1, 'W/C': 1, 'mW/K': 1e-3, 'W/mK': 1e3,
            'kW/K': 1e3, 'MW/K': 1e6, 'GW/K': 1e9
        },
        'type': 'ratio',
        'equal': '',
        'differ': 1e-1,
        'latex_eq': r'0 = kA - kA_\mathrm{spec}',
        'documentation': {'float_fmt': '{:,.3f}'}
    },
    'hf': {
        'text': 'convective heat transfer coefficient',
        'SI_unit': 'W/m^2*K',
        'units': {
            'W/m^2*K': 1, 'W/m^2*C': 1, 'mW/m^2*K': 1e-3, 'W/m^2*mK': 1e3,
            'KW/m^2*K': 1e3, 'MW/m^2*K': 1e6, 'GW/m^2*K': 1e9
        },
        'type': 'ratio',
        'equal': '',
        'differ': 1e-1,
        'latex_eq': r'0 = hf - hf_\mathrm{spec}',
        'documentation': {'float_fmt': '{:,.3f}'}
    },
    'fA': {
        'text': 'equivalent area',
        'SI_unit': 'm^2',
        'units': {
            'm^2': 1, 'dm^2': 1e-2, 'cm^2': 1e-4, 'mm^2': 1e-6,
        },
        'type': 'ratio',
        'equal': '',
        'differ': 1e-1,
        'latex_eq': r'0 = fA - fA_\mathrm{spec}',
        'documentation': {'float_fmt': '{:,.3f}'}
    },
    'ratio': {
        'text': 'ratio',
        'SI_unit': '-',
        'units': {
            '-': 1, '%': 1e-2, 'ppm': 1e-6
        },
        'type': 'ratio',
        'equal': '',
        'differ': 1e-4,
        'latex_eq': r'0 = ratio - ratio_\mathrm{spec}',
        'documentation': {'float_fmt': '{:,.2f}'}
    },
    'eta': {
        'text': 'efficiency',
        'SI_unit': '-',
        'units': {
            '-': 1, '%': 1e-2, 'ppm': 1e-6
        },
        'type': 'ratio',
        'equal': '',
        'differ': 1e-4,
        'latex_eq': r'0 = eta - eta_\mathrm{spec}',
        'documentation': {'float_fmt': '{:,.2f}'}
    },
    'zeta': {
        'text': 'pressure drop rate',
        'SI_unit': '(Pa*s)/(m^3*s)',
        'units': {
            '(Pa*s)/(m^3*s)': 1,
        },
        'type': 'ratio',
        'equal': '',
        'differ': 1e-4,
        'latex_eq': r'0 = zeta - zeta_\mathrm{spec}',
        'documentation': {'float_fmt': '{:,.3f}'}
    },
    'ks': {
        'text': 'roughness',
        'SI_unit': 'm',
        'units': {
            'm': 1, 'dm': 1e-1, 'cm': 1e-2, 'mm': 1e-3, 'um': 1e-6, 'nm': 1e-9,
            'km': 1e3,
        },
        'type': 'ratio',
        'equal': '',
        'differ': 1e-4,
        'latex_eq': r'0 = ks - ks_\mathrm{spec}',
        'documentation': {'float_fmt': '{:,.3f}'}
    },
    'Angle': {
        'text': 'blade angle',
        'SI_unit': 'deg',
        'units': {
            'deg': 1, 'rad': 180/math.pi, 'grad': 180/200, 'turn': 360
        },
        'type': 'ratio',
        'equal': '',
        'differ': 1e-4,
        'latex_eq': r'0 = Angle - Angle_\mathrm{spec}',
        'documentation': {'float_fmt': '{:,.3f}'}
    },
    'Omega': {
        'text': 'rotate speed',
        'SI_unit': 'rad/s',
        'units': {
            'rad/s': 1, 'rpm': math.pi/30, 'r/s': 2 * math.pi, 'r/min': math.pi/30, 'r/h': math.pi/1800
        },
        'type': 'ratio',
        'equal': '',
        'differ': 1e-4,
        'latex_eq': r'0 = Omega - Omega_\mathrm{spec}',
        'documentation': {'float_fmt': '{:,.3f}'}
    },
    'J': {
        'text': 'rotational inertia',
        'SI_unit': 'kg*m^2',
        'units': {
            'kg*m^2': 1, 'g*m^2': 1e-3, 'g*cm^2': 1e-5,
        },
        'type': 'ratio',
        'equal': '',
        'differ': 1e-4,
        'latex_eq': r'0 = J - J_\mathrm{spec}',
        'documentation': {'float_fmt': '{:,.3f}'}
    },
    'M': {
        'text': 'moment',
        'SI_unit': 'N*m',
        'units': {
            'N*m': 1, 'kgf*m': 9.80665, 'lb*ft': 1.3558,
        },
        'type': 'ratio',
        'equal': '',
        'differ': 1e-4,
        'latex_eq': r'0 = M - M_\mathrm{spec}',
        'documentation': {'float_fmt': '{:,.3f}'}
    },
    'q': {
        'text': 'electric charge',
        'SI_unit': 'C',
        'units': {
            'C': 1, 'mC': 1e-3, 'kC': 1e3,
            'MC': 1e6, 'GC': 1e9
        },
        'type': 'ratio',
        'equal': '',
        'differ': 1e-4,
        'latex_eq': r'0 = q - q_\mathrm{spec}',
        'documentation': {'float_fmt': '{:,.3f}'}
    },
    'C': {
        'text': 'electric capacitance',
        'SI_unit': 'F',
        'units': {
            'F': 1, 'mF': 1e-3, 'kF': 1e3,
            'uF': 1e-6, 'nF': 1e-9, 'pF': 1e-12
        },
        'type': 'ratio',
        'equal': '',
        'differ': 1e-4,
        'latex_eq': r'0 = C - C_\mathrm{spec}',
        'documentation': {'float_fmt': '{:,.3f}'}
    },
    'L': {
        'text': 'electric inductance',
        'SI_unit': 'H',
        'units': {
            'H': 1, 'mH': 1e-3, 'kH': 1e3, 'MH': 1e6,
            'uH': 1e-6, 'nH': 1e-9, 'pH': 1e-12
        },
        'type': 'ratio',
        'equal': '',
        'differ': 1e-4,
        'latex_eq': r'0 = H - H_\mathrm{spec}',
        'documentation': {'float_fmt': '{:,.3f}'}
    },
    'R': {
        'text': 'electric resistance',
        'SI_unit': 'Ohm',
        'units': {
            'Ohm': 1,  'mOhm': 1e-3, 'uOhm': 1e-6,
            'KOhm': 1e3, 'MOhm': 1e6,
        },
        'type': 'ratio',
        'equal': '',
        'differ': 1e-4,
        'latex_eq': r'0 = R - R_\mathrm{spec}',
        'documentation': {'float_fmt': '{:,.3f}'}
    },
    'E': {
        'text': 'electric field intensity',
        'SI_unit': 'V/m',
        'units': {
            'V/m': 1, 'N/C': 1,
        },
        'type': 'ratio',
        'equal': '',
        'differ': 1e-4,
        'latex_eq': r'0 = E - E_\mathrm{spec}',
        'documentation': {'float_fmt': '{:,.3f}'}
    },
    'B': {
        'text': 'magnetic flux density',
        'SI_unit': 'T',
        'units': {
            'T': 1, 'G': 1e-4,
        },
        'type': 'ratio',
        'equal': '',
        'differ': 1e-4,
        'latex_eq': r'0 = B - B_\mathrm{spec}',
        'documentation': {'float_fmt': '{:,.3f}'}
    },
    'H': {
        'text': 'magnetic field intensity',
        'SI_unit': 'A/mf',
        'units': {
            'A/mf': 1, 'Oe': 103/(4*math.pi), 'Gb/cm': 103/(4*math.pi),
        },
        'type': 'ratio',
        'equal': '',
        'differ': 1e-4,
        'latex_eq': r'0 = H - H_\mathrm{spec}',
        'documentation': {'float_fmt': '{:,.3f}'}
    },
    'Z': {
        'text': 'impedance',
        'SI_unit': 'Ohm',
        'units': {
            'Ohm': 1,  'mOhm': 1e-3, 'uOhm': 1e-6,
            'KOhm': 1e3, 'MOhm': 1e6,
        },
        'type': 'ratio',
        'equal': 'R',
        'differ': 1e-4,
        'latex_eq': r'0 = Z - Z_\mathrm{spec}',
        'documentation': {'float_fmt': '{:,.3f}'}
    },
    'XL': {
        'text': 'inductive impedance',
        'SI_unit': 'Ohm',
        'units': {
            'Ohm': 1,  'mOhm': 1e-3, 'uOhm': 1e-6,
            'KOhm': 1e3, 'MOhm': 1e6,
        },
        'type': 'ratio',
        'equal': 'R',
        'differ': 1e-4,
        'latex_eq': r'0 = XL - XL_\mathrm{spec}',
        'documentation': {'float_fmt': '{:,.3f}'}
    },
    'CL': {
        'text': 'capacitive reactance',
        'SI_unit': 'Ohm',
        'units': {
            'Ohm': 1,  'mOhm': 1e-3, 'uOhm': 1e-6,
            'KOhm': 1e3, 'MOhm': 1e6,
        },
        'type': 'ratio',
        'equal': 'R',
        'differ': 1e-4,
        'latex_eq': r'0 = CL - CL_\mathrm{spec}',
        'documentation': {'float_fmt': '{:,.3f}'}
    },
}

space_time_property_data = {
    'l': {
        'text': 'spatial size',
        'SI_unit': 'm',
        'units': {
            'm': 1, 'dm': 1e-1, 'cm': 1e-2, 'mm': 1e-3, 'um': 1e-6, 'nm': 1e-9,
            'km': 1e3,
        },
        'type': 'ratio',
        'equal': '',
        'differ': 1e-4,
        'latex_eq': r'0 = l - l_\mathrm{spec}',
        'documentation': {'float_fmt': '{:,.3f}'}
    },
    't': {
        'text': 'time size',
        'SI_unit': 's',
        'units': {
            's': 1, 'min': 60, 'h': 3600,'Day': 86400,
            'month': 2592000, 'year': 31536000,
        },
        'type': 'ratio',
        'equal': '',
        'differ': 1e-4,
        'latex_eq': r'0 = t - t_\mathrm{spec}',
        'documentation': {'float_fmt': '{:,.3f}'}
    },
}

mathematical_property_data = {
    'n': {
        'text': 'differ number',
        'SI_unit': '',
        'units': {

        },
        'type': 'ratio',
        'equal': '',
        'differ': 1e-4,
        'latex_eq': r'0 = n - n_\mathrm{spec}',
        'documentation': {'float_fmt': '{:,.3f}'}
    },
}

combustion_gases = ['methane', 'ethane', 'propane', 'butane', 'hydrogen', 'nDodecane']  # the main component of gas/oil

property_scale = {
    'm': {
        'text': 'mass flow',
        'scale': 1e0,
    },
    'v': {
        'text': 'volumetric flow',
        'scale': 1e1,
    },
    'p': {
        'text': 'pressure',
        'scale': 1e5,
    },
    'h': {
        'text': 'enthalpy',
        'scale': 1e5,
    },
    'T': {
        'text': 'temperature',
        'scale': 1e1,
    },
    'DT': {
        'text': 'delta temperature',
        'scale': 1e1,
    },
    'U': {
        'text': 'voltage',
        'scale': 1e1,
    },
    'I': {
        'text': 'electricity',
        'scale': 1e1,
    },
    'f': {
        'text': 'electronic frequency',
        'scale': 1e1,
    },
    'vol': {
        'text': 'specific volume',
        'scale': 1e0,
    },
    'x': {
        'text': 'vapor mass fraction',
        'scale': 1e0,
    },
    's': {
        'text': 'entropy',
        'scale': 1e1,
    },
    'fluid': {
        'text': 'fluid composition mass fraction',
        'scale': 1e0,
    },
    'num_inter': {
        'text': 'number of interfaces for subsystem',
        'scale': 1e0,
    },
    'pr': {
        'text': 'pressure ratio',
        'scale': 1e0,
    },
    'eff': {
        'text': 'efficiency',
        'scale': 1e0,
    }
}