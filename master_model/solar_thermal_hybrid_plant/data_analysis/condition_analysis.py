from path_dict import path_dict
import numpy as np
import pandas as pd
import json

pd.set_option('future.no_silent_downcasting', True)
E_list = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0', '1.1']
gas_list = ['0.6', '0.7', '0.8', '0.9', '1.0', '1.1', '1.2']

def condition_reading(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    # reversion dataframe
    dfs = {}
    for key, value in data["Connection"].items():  # key: connection type
        dfs[key] = pd.DataFrame.from_dict(value, orient="index").fillna(np.nan)
        dfs[key].index = dfs[key].index.astype(str)
    for key, value in data["Component"].items():  # key: component type
        dfs[key] = pd.DataFrame.from_dict(value, orient="index").fillna(np.nan)
        dfs[key].index = dfs[key].index.astype(str)
    for key, value in data["Bus"].items():  # key: bus.label
        dfs[key] = pd.DataFrame.from_dict(value, orient="index").fillna(np.nan)
        dfs[key].index = dfs[key].index.astype(str)
    return dfs


def condition_analysis():
    gas_mass_flow = pd.DataFrame(columns=gas_list, index=E_list, dtype=float)
    gas_turbine_power = pd.DataFrame(columns=gas_list, index=E_list, dtype=float)
    steam_turbine_power = pd.DataFrame(columns=gas_list, index=E_list, dtype=float)
    gas_cycle_efficiency = pd.DataFrame(columns=gas_list, index=E_list, dtype=float)
    steam_cycle_efficiency = pd.DataFrame(columns=gas_list, index=E_list, dtype=float)
    combined_cycle_power = pd.DataFrame(columns=gas_list, index=E_list, dtype=float)
    combined_cycle_efficiency = pd.DataFrame(columns=gas_list, index=E_list, dtype=float)
    for E_index in E_list:
        for gas_index in gas_list:
            data = condition_reading(path_dict[E_index][gas_index]['path'])
            P_gas_turbine = data['Turbine'].loc['gas_turbine', 'P']
            P_hp_turbine1 = data['Turbine'].loc['hp_turbine1', 'P']
            P_hp_turbine2 = data['Turbine'].loc['hp_turbine2', 'P']
            P_lp_turbine1 = data['Turbine'].loc['lp_turbine1', 'P']
            P_lp_turbine2 = data['Turbine'].loc['lp_turbine2', 'P']
            P_lp_turbine3 = data['Turbine'].loc['lp_turbine3', 'P']
            P_lp_turbine4 = data['Turbine'].loc['lp_turbine4', 'P']
            P_lp_turbine5 = data['Turbine'].loc['lp_turbine5', 'P']
            P_compressor = data['Compressor'].loc['air_compressor', 'P']
            P_pump1 = data['Pump'].loc['oil_recycle_pump1', 'P']
            P_pump2 = data['Pump'].loc['oil_recycle_pump2', 'P']
            P_pump3 = data['Pump'].loc['steam_recycle_pump', 'P']
            P_pump4 = data['Pump'].loc['water_recycle_pump', 'P']
            P_pump5 = data['Pump'].loc['condense_pump', 'P']
            P_combustion = data['DiabaticCombustionChamber'].loc['combustion', 'ti']
            P_hot_salt_tank = data['HeatStorageTank'].loc['hot_salt_tank', 'Q']
            P_cold_salt_tank = data['HeatStorageTank'].loc['cold_salt_tank', 'Q']
            P_solar_collector1 = data['SolarCollector'].loc['solar_collector1', 'Q']
            P_solar_collector2 = data['SolarCollector'].loc['solar_collector2', 'Q']
            P_solar_collector3 = data['SolarCollector'].loc['solar_collector3', 'Q']
            P_solar_collector4 = data['SolarCollector'].loc['solar_collector4', 'Q']
            P_heatexchanger_d = data['HeatExchanger'].loc['heatexchanger_d', 'Q']
            P_evaporator = data['Evaporator'].loc['evaporator', 'Q']
            P_heatexchanger_e = data['HeatExchanger'].loc['heatexchanger_e', 'Q']
            P_heatexchanger_f = data['HeatExchanger'].loc['heatexchanger_f', 'Q']
            mass_gas = data['FluidConnection'].loc['c2', 'm']
            # value calculation
            gas_power = abs(P_gas_turbine) - abs(P_compressor)
            gas_heat = abs(P_combustion)
            gas_efficiency = gas_power / gas_heat
            steam_power = (abs(P_hp_turbine1 + P_hp_turbine2 + P_lp_turbine1 + P_lp_turbine2 + P_lp_turbine3 + P_lp_turbine4 + P_lp_turbine5) -
                           abs(2 * P_pump3 + P_pump4 + P_pump5))
            steam_heat = 2 * abs(P_heatexchanger_d + P_evaporator + P_heatexchanger_e + P_heatexchanger_f)
            steam_efficiency = steam_power / steam_heat
            combine_power = (abs(P_gas_turbine + P_hp_turbine1 + P_hp_turbine2 +
                                P_lp_turbine1 + P_lp_turbine2 + P_lp_turbine3 + P_lp_turbine4 + P_lp_turbine5) -
                             abs(P_compressor) - abs(P_pump1 + P_pump2 + 2 * P_pump3 + P_pump4 + P_pump5))
            combine_heat = (156 * abs(P_solar_collector1 + P_solar_collector2 + P_solar_collector3 + P_solar_collector4) +
                            abs(P_combustion) - abs(P_hot_salt_tank) + abs(P_cold_salt_tank))
            combine_efficiency = combine_power / combine_heat
            # document
            gas_turbine_power.loc[E_index, gas_index] = gas_power
            steam_turbine_power.loc[E_index, gas_index] = steam_power
            combined_cycle_power.loc[E_index, gas_index] = combine_power
            gas_cycle_efficiency.loc[E_index, gas_index] = gas_efficiency
            steam_cycle_efficiency.loc[E_index, gas_index] = steam_efficiency
            combined_cycle_efficiency.loc[E_index, gas_index] = combine_efficiency
            gas_mass_flow.loc[E_index, gas_index] = mass_gas
    # save results
    gas_turbine_power.to_csv("data/gas_turbine_power.csv")
    steam_turbine_power.to_csv("data/steam_turbine_power.csv")
    combined_cycle_power.to_csv("data/combined_cycle_power.csv")
    gas_cycle_efficiency.to_csv("data/gas_cycle_efficiency.csv")
    steam_cycle_efficiency.to_csv("data/steam_cycle_efficiency.csv")
    combined_cycle_efficiency.to_csv("data/combined_cycle_efficiency.csv")
    gas_mass_flow.to_csv("data/gas_mass_flow.csv")

if __name__ == '__main__':
    condition_analysis()