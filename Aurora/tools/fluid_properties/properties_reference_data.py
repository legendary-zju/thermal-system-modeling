# -*- coding: utf-8

"""Module for free fluid property data.
Aurora/tools/fluid_properties/properties_reference_data.py

SPDX-License-Identifier: MIT
"""

from .property_container import FluidPropertyContainer

# reference data for molten salt
MOLTEN_SALT_REFERENCE_DATA = {
    # 常见熔盐在熔点时的参考焓（示例值，实际需要从文献获取）
    "Solar Salt": {"T_ref": 494.15, "h_ref": 0.0},  # 太阳盐熔点~221°C，参考焓设为0
    "Hitec": {"T_ref": 415.15, "h_ref": 0.0},  # Hitec盐熔点~142°C
    "HitecXL": {"T_ref": 393.15, "h_ref": 0.0},  # Hitec XL盐熔点~120°C
    # 纯盐
    "NaNO3": {"T_ref": 579.15, "h_ref": 0.0},  # 硝酸钠熔点~306°C
    "KNO3": {"T_ref": 607.15, "h_ref": 0.0},  # 硝酸钾熔点~334°C
    "LiNO3": {"T_ref": 527.15, "h_ref": 0.0},  # 硝酸锂熔点~254°C
    "NaNO2": {"T_ref": 544.15, "h_ref": 0.0},  # 亚硝酸钠熔点~271°C
    "KNO2": {"T_ref": 710.15, "h_ref": 0.0},  # 亚硝酸钾熔点~437°C
    "Na2CO3": {"T_ref": 1124.15, "h_ref": 0.0},  # 碳酸钠熔点~851°C
    "K2CO3": {"T_ref": 1164.15, "h_ref": 0.0},  # 碳酸钾熔点~891°C
    "Li2CO3": {"T_ref": 996.15, "h_ref": 0.0},  # 碳酸锂熔点~723°C
}

molar_mass_dict = {
            # mixture
            "Solar Salt": 0.0892,  # 60%NaNO3-40%KNO3
            "Hitec": 0.0866,  # 53%KNO3-40%NaNO2-7%NaNO3
            "HitecXL": 0.0950,  # Hitec XL
            # pure
            "NaNO3": 0.084994,  # 硝酸钠
            "KNO3": 0.101103,  # 硝酸钾
            "LiNO3": 0.068946,  # 硝酸锂
            "NaNO2": 0.068995,  # 亚硝酸钠
            "KNO2": 0.085104,  # 亚硝酸钾
            "Na2CO3": 0.105989,  # 碳酸钠
            "K2CO3": 0.138206,  # 碳酸钾
            "Li2CO3": 0.073891,  # 碳酸锂
            # other salt
            "NaCl": 0.058443,  # 氯化钠
            "KCl": 0.074551,  # 氯化钾
            "LiCl": 0.042394,  # 氯化锂
            "CaCl2": 0.110984,  # 氯化钙
            "MgCl2": 0.095211,  # 氯化镁
        }

# atomic molar mass（unit：g/mol）
atomic_masses = {
            "H": 1.00794, "He": 4.002602,
            "Li": 6.941, "Be": 9.012182, "B": 10.811,
            "C": 12.0107, "N": 14.0067, "O": 15.9994, "F": 18.998403,
            "Na": 22.98977, "Mg": 24.3050, "Al": 26.98154,
            "Si": 28.0855, "P": 30.97376, "S": 32.065, "Cl": 35.453,
            "K": 39.0983, "Ca": 40.078, "Sc": 44.95591,
            "Ti": 47.867, "V": 50.9415, "Cr": 51.9961,
            "Mn": 54.938045, "Fe": 55.845, "Co": 58.93320,
            "Ni": 58.6934, "Cu": 63.546, "Zn": 65.38,
            "Ga": 69.723, "Ge": 72.64, "As": 74.92160,
            "Se": 78.96, "Br": 79.904, "Kr": 83.798,
            "Rb": 85.4678, "Sr": 87.62, "Y": 88.90585,
            "Zr": 91.224, "Nb": 92.90638, "Mo": 95.96,
            "Tc": 98, "Ru": 101.07, "Rh": 102.90550,
            "Pd": 106.42, "Ag": 107.8682, "Cd": 112.411,
            "In": 114.818, "Sn": 118.710, "Sb": 121.760,
            "Te": 127.60, "I": 126.90447, "Xe": 131.293,
            "Cs": 132.90545, "Ba": 137.327, "La": 138.90547,
            "Ce": 140.116, "Pr": 140.90765, "Nd": 144.242,
            "Pm": 145, "Sm": 150.36, "Eu": 151.964,
            "Gd": 157.25, "Tb": 158.92535, "Dy": 162.500,
            "Ho": 164.93032, "Er": 167.259, "Tm": 168.93421,
            "Yb": 173.054, "Lu": 174.9668,
            "Hf": 178.49, "Ta": 180.94788, "W": 183.84,
            "Re": 186.207, "Os": 190.23, "Ir": 192.217,
            "Pt": 195.084, "Au": 196.966569, "Hg": 200.59,
            "Tl": 204.3833, "Pb": 207.2, "Bi": 208.98040,
            "Po": 209, "At": 210, "Rn": 222,
            "Fr": 223, "Ra": 226, "Ac": 227,
            "Th": 232.03806, "Pa": 231.03588, "U": 238.02891,
            "Np": 237, "Pu": 244, "Am": 243,
            "Cm": 247, "Bk": 247, "Cf": 251,
            "Es": 252, "Fm": 257, "Md": 258,
            "No": 259, "Lr": 262, "Rf": 267,
            "Db": 268, "Sg": 271, "Bh": 272,
            "Hs": 270, "Mt": 276, "Ds": 281,
            "Rg": 280, "Cn": 285, "Nh": 284,
            "Fl": 289, "Mc": 288, "Lv": 293,
            "Ts": 294, "Og": 294
        }

compound_groups_masses = {
                "NO3": 62.0049,  # 硝酸根
                "NO2": 46.0055,  # 亚硝酸根
                "CO3": 60.0089,  # 碳酸根
                "OH": 17.0073,  # 氢氧根
                "SO4": 96.0626,  # 硫酸根
                "PO4": 94.9714,  # 磷酸根
                    }

Customized_Fluid = {
    "NaNO3": {
        # properties of reference point
        "reference_point": {
            "T": 579.15,  # 熔点 (K)
            "p": 101325.0,  # 标准大气压 (Pa)
            "h": 0.0,  # 熔点时液相参考焓值 (J/kg)
            "s": 0.0,  # 熔点时液相参考熵值 (J/kg-K)
            "phase": "liquid"  # 参考点相态
        },
        "T_melt": 579.15,  # 熔点 (K)
        "T_boil": 1153.15,  # 沸点估算 (K)
        "T_triple": 579.15,  # 三相点 (假设与熔点相同)
        "p_triple": 101325.0,  # 三相点压力 (Pa)
        # latent heat of phase change (J/kg)
        "delta_h_fusion": 189000.0,  # 熔化潜热
        "delta_h_vaporization": 1550000.0,  # 汽化潜热估算
        # critical properties
        "T_crit": 2100.0,  # 临界温度估算 (K)
        "p_crit": 45.0e6,  # 临界压力估算 (Pa) - 实际可能更高
        # range of application
        "T_min": 500.0,  # 最低工作温度 (K)
        "T_max": 1200.0,  # 最高工作温度 (K)
        "p_min": 1.0,  # 最低工作压力 (Pa)
        "p_max": 100.0e6,  # 最高工作压力 (Pa)
        # basic properties
        "molar_mass": 84.9947/1e3,  # 摩尔质量 (kg/mol)
        "density_ref": 1900.0,  # 参考密度 @ 600K (kg/m³)
        # specific heat
        'Cp': FluidPropertyContainer(
            value=1650.0,  # 参考值 J/kg-K @ 600K
            type='polynomial',
            intro={
                'coefficients': [1490.0, 1.456, -0.00045],  # a0 + a1*T + a2*T^2
                'valid_range': [500.0, 1200.0],  # 有效温度范围 (K)
                'parameters': ['T'],
                'units': {
                    'T': 'K',
                    'Cp': 'J/kgK'
                },
            }
        ),
        'Cv': FluidPropertyContainer(
            value=1400.0,  # 估算值
            type='constant'
        ),
        'density': FluidPropertyContainer(
            value=1900.0,
            type='linear_expansion',
            intro={
                'value0': 1920.0,  # 参考密度 @ 600K (kg/m³)
                'alpha': 3.5e-4,  # 体积膨胀系数 (1/K)
                'belta': 600,
                'valid_range': [500.0, 1200.0],
                'parameters': ['T'],
                'units': {
                    'T': 'K',
                    'density': 'kg/m³'
                },
            }
        ),
        'viscosity': FluidPropertyContainer(
            value=0.003,  # Pa-s @ 600K
            type='exponential',
            intro={
                'A': 1.2e-4,
                'B': 2800.0,  # 活化能/R
                'valid_range': [500.0, 1200.0],
                'parameters': ['T'],
                'units': {
                    'T': 'K',
                    'viscosity': 'Pa·s'
                },
            }
        ),
        'thermal_conductivity': FluidPropertyContainer(
            value=0.52,  # W/m-K @ 600K
            type='polynomial',
            intro={
                'coefficients': [0.48, 1.2e-4, -2.5e-8],
                'valid_range': [500.0, 1200.0],
                'parameters': ['T'],
            }
        ),
        'saturated_pressure': FluidPropertyContainer(
            type='exponential',
            intro={
                'A': 10.23,
                'B': -12045.0,
                'C': -2.15,
                'valid_range': [600.0, 1153.0],
                'parameters': ['T'],
            }
        ),
        'phase_change': {
            'melting': {
                'T': 579.15,
                'delta_h': 189000.0,
                'delta_s': 326.4
            },
            'vaporization': {
                'T': 1153.15,
                'delta_h': 1550000.0,
                'delta_s': 1344.0
            }
        },
    },
    "KNO3": {
        "reference_point": {
            "T": 607.15,
            "p": 101325.0,
            "h": 0.0,
            "s": 0.0,
            "phase": "liquid"
        },
        "T_melt": 607.15,
        "T_boil": 1273.15,
        "T_triple": 607.15,
        "p_triple": 101325.0,
        "delta_h_fusion": 117000.0,
        "delta_h_vaporization": 1450000.0,
        "T_crit": 2200.0,
        "p_crit": 40.0e6,
        "T_min": 550.0,
        "T_max": 1300.0,
        "p_min": 1.0,
        "p_max": 100.0e6,
        "molar_mass": 101.1032,
        "density_ref": 1950.0,
        'Cp': FluidPropertyContainer(
            value=1550.0,
            type='polynomial',
            intro={
                'coefficients': [1420.0, 1.215, -0.00038],
                'valid_range': [550.0, 1300.0],
                'parameters': ['T'],
                'units': {
                    'T': 'K',
                    'Cp': 'J/kgK'
                },
            }
        ),
        'Cv': FluidPropertyContainer(
            value=1350.0,
            type='constant',
        ),
        'density': FluidPropertyContainer(
            value=1950.0,
            type='linear_expansion',
            intro={
                'value0': 1960.0,
                'alpha': 3.2e-4,
                'belta': 600,
                'valid_range': [550.0, 1300.0],
                'parameters': ['T'],
                'units': {
                    'T': 'K',
                    'density': 'kg/m³'
                },
            }
        ),
        'viscosity': FluidPropertyContainer(
            value=0.0035,
            type='exponential',
            intro={
                'A': 1.5e-4,
                'B': 2700.0,
                'valid_range': [550.0, 1300.0],
                'parameters': ['T'],
                'units': {
                    'T': 'K',
                    'viscosity': 'Pa·s'
                },
            }
        ),
        'phase_change': {
            'melting': {
                'T': 607.15,
                'delta_h': 117000.0,
                'delta_s': 326.4
            },
            'vaporization': {
                'T': 1273.15,
                'delta_h': 1450000.0,
                'delta_s': 1344.0
            }
        },
    },
    "Solar Salt": {
        "reference_point": {
            "T": 503.15,
            "p": 101325.0,
            "h": 0.0,
            "s": 0.0,
            "phase": "liquid"
        },
        "T_melt": 503.15,  # 60% NaNO3 + 40% KNO3 共晶混合物
        "T_boil": 1400.0,  # 估算
        "T_triple": 503.15,
        "p_triple": 101325.0,
        "delta_h_fusion": 161000.0,  # 混合盐的熔化潜热
        "delta_h_vaporization": 1500000.0,
        "T_crit": 2300.0,
        "p_crit": 42.0e6,
        "T_min": 450.0,
        "T_max": 850.0,  # 推荐运行上限
        "p_min": 1.0,
        "p_max": 100.0e6,
        "molar_mass": 91.0,  # 平均摩尔质量
        "density_ref": 1850.0,
        # composition
        "composition": {
            "NaNO3": 0.6,  # 质量分数
            "KNO3": 0.4
        },
        'Cp': FluidPropertyContainer(
            value=1600.0,
            type='polynomial',
            intro={
                'coefficients': [1495.0, 1.2, -0.00035],
                'valid_range': [450.0, 850.0],
                'parameters': ['T'],
                'units': {
                    'T': 'K',
                    'Cp': 'J/kgK'
                },
            }
        ),
        'Cv': FluidPropertyContainer(
            value=1450.0,
            type='constant'
        ),
        'density': FluidPropertyContainer(
            value=1850.0,
            type='linear_expansion',
            intro={
                'value0': 1850.0,
                'alpha': 3.0e-4,
                'belta': 573.15,
                'valid_range': [450.0, 850.0],
                'parameters': ['T'],
                'units': {
                    'T': 'K',
                    'density': 'kg/m³'
                },
            }
        ),
        'viscosity': FluidPropertyContainer(
            value=0.0032,
            type='exponential',
            intro={
                'A': 1.3e-4,
                'B': 2600.0,
                'valid_range': [450.0, 850.0],
                'parameters': ['T'],
                'units': {
                    'T': 'K',
                    'viscosity': 'Pa·s'
                },
            }
        ),
        'thermal_conductivity': FluidPropertyContainer(
            value=0.49,
            type='constant'
        ),

        'entropy': FluidPropertyContainer(
            value=0.0,
            type='table_lookup',
            intro={
                'parameters': ['T', 'p'],
                'method': 'linear',
                'table': {
                    'axes': [
                        [400.0, 500.0, 600.0, 700.0, 800.0, 900.0],  # T (K)
                        [1e5, 1e6, 1e7]  # p (Pa)
                    ],
                    'data': [  # s (J/kg-K)
                        # 熵值通过∫Cp/T dT计算
                        [-580.0, -580.5, -581.5],  # T=400K
                        [-290.0, -290.5, -291.5],  # T=500K
                        [0.0, -0.5, -1.5],  # T=600K (参考点)
                        [258.0, 257.5, 256.5],  # T=700K
                        [490.0, 489.5, 488.5],  # T=800K
                        [705.0, 704.5, 703.5]  # T=900K
                    ]
                },
                'reference_point': {
                    'T': 503.15,
                    'p': 101325.0,
                    's': 0.0
                },
                'units': {
                    'T': 'K',
                    'p': 'Pa',
                    's': 'J/kg-K'
                }
            }
        ),
        'enthalpy': FluidPropertyContainer(
            value=0.0,
            type='table_lookup',
            intro={
                 'parameters': ['T', 'p'],
                 'method': 'linear',
                 'table': {
                     'axes': [
                             # 温度轴 (K) - 更密集的点
                                [400.0, 425.0, 450.0, 475.0, 500.0, 525.0, 550.0, 575.0,
                                600.0, 625.0, 650.0, 675.0, 700.0, 725.0, 750.0, 775.0, 800.0],
                             # 压力轴 (Pa) - 更宽的范围
                                [1e3, 1e4, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7, 1e8]
                             ],
                    # 焓值矩阵 (J/kg) - 基于状态方程计算
                    # 使用公式: h(T,p) = h_ref + ∫Cp(T)dT + ∫v(1 - αT)dp
                     'data': [
                             # T=400K
                                [-218.2e3, -218.3e3, -218.5e3, -219.0e3, -219.5e3, -221.0e3, -222.5e3, -228.0e3, -233.0e3],
                             # T=425K
                                [-152.8e3, -152.9e3, -153.1e3, -153.6e3, -154.1e3, -155.6e3, -157.1e3, -162.6e3, -167.6e3],
                             # T=450K
                                [-87.4e3, -87.5e3, -87.7e3, -88.2e3, -88.7e3, -90.2e3, -91.7e3, -97.2e3, -102.2e3],
                             # T=475K
                                [-21.8e3, -21.9e3, -22.1e3, -22.6e3, -23.1e3, -24.6e3, -26.1e3, -31.6e3, -36.6e3],
                             # T=500K
                                [43.6e3, 43.5e3, 43.3e3, 42.8e3, 42.3e3, 40.8e3, 39.3e3, 33.8e3, 28.8e3],
                             # T=525K
                                [109.2e3, 109.1e3, 108.9e3, 108.4e3, 107.9e3, 106.4e3, 104.9e3, 99.4e3, 94.4e3],
                             # T=550K
                                [175.0e3, 174.9e3, 174.7e3, 174.2e3, 173.7e3, 172.2e3, 170.7e3, 165.2e3, 160.2e3],
                             # T=575K
                                [241.0e3, 240.9e3, 240.7e3, 240.2e3, 239.7e3, 238.2e3, 236.7e3, 231.2e3, 226.2e3],
                             # T=600K
                                [307.2e3, 307.1e3, 306.9e3, 306.4e3, 305.9e3, 304.4e3, 302.9e3, 297.4e3, 292.4e3],
                             # T=625K
                                [373.6e3, 373.5e3, 373.3e3, 372.8e3, 372.3e3, 370.8e3, 369.3e3, 363.8e3, 358.8e3],
                             # T=650K
                                [440.2e3, 440.1e3, 439.9e3, 439.4e3, 438.9e3, 437.4e3, 435.9e3, 430.4e3, 425.4e3],
                             # T=675K
                                [507.0e3, 506.9e3, 506.7e3, 506.2e3, 505.7e3, 504.2e3, 502.7e3, 497.2e3, 492.2e3],
                             # T=700K
                                [574.0e3, 573.9e3, 573.7e3, 573.2e3, 572.7e3, 571.2e3, 569.7e3, 564.2e3, 559.2e3],
                             # T=725K
                                [641.2e3, 641.1e3, 640.9e3, 640.4e3, 639.9e3, 638.4e3, 636.9e3, 631.4e3, 626.4e3],
                             # T=750K
                                [708.6e3, 708.5e3, 708.3e3, 707.8e3, 707.3e3, 705.8e3, 704.3e3, 698.8e3, 693.8e3],
                             # T=775K
                                [776.2e3, 776.1e3, 775.9e3, 775.4e3, 774.9e3, 773.4e3, 771.9e3, 766.4e3, 761.4e3],
                             # T=800K
                                [844.0e3, 843.9e3, 843.7e3, 843.2e3, 842.7e3, 841.2e3, 839.7e3, 834.2e3, 829.2e3]
                     ]
                },
                'units': {
                          'T': 'K',
                          'p': 'Pa',
                          'h': 'J/kg'
                }
            }
        )
    },
}