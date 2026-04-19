from Aurora.components import CycleCloser, CombustionChamber
from Aurora.components import Sink
from Aurora.components import Source
from Aurora.components import FlowAmplifier
from Aurora.components import Condenser
from Aurora.components import Deaerator, DeaeratorSimple
from Aurora.components import Desuperheater
from Aurora.components import SimpleHeatExchanger, HeatExchanger, Evaporator, ExtractHeatExchanger, BoilerSimple, OverHeater
from Aurora.components import DiabaticCombustionChamber
from Aurora.components import Merge
from Aurora.components import Splitter
from Aurora.components import DropletSeparator, Drum, EvaporateTank
from Aurora.components import HeatStorageTank
from Aurora.components import Valve
from Aurora.components import Pump
from Aurora.components import Compressor
from Aurora.components import Turbine
from Aurora.components import SolarCollector
from Aurora.connections import Bus
from Aurora.connections import Connection
from Aurora.networks import Network
from Aurora.connections import Bus

from Aurora.tools.characteristics import load_default_char
from Aurora.tools.characteristics import CharLine
from Aurora.tools import logger
import logging


class SolarThermalCombinePlant1:
    def __init__(self, name):
        self.name = name
        self.nw = Network(p_unit="bar", T_unit='C', h_unit="kJ / kg", m_unit='kg / s', iterinfo=True)

        #########################
        # set components
        # combustion
        self.combustion = DiabaticCombustionChamber('combustion')
        # solar collector
        self.solar_collector1 = SolarCollector('solar_collector1')
        self.solar_collector2 = SolarCollector('solar_collector2')
        self.solar_collector3 = SolarCollector('solar_collector3')
        self.solar_collector4 = SolarCollector('solar_collector4')
        # heat exchanger
        self.heatexchanger_a = HeatExchanger('heatexchanger_a', nodes_num=40)
        self.heatexchanger_b = ExtractHeatExchanger('heatexchanger_b', nodes_num=40)
        self.heatexchanger_c = ExtractHeatExchanger('heatexchanger_c', nodes_num=40)
        self.heatexchanger_d = OverHeater('heatexchanger_d', nodes_num=40)
        self.heatexchanger_e = ExtractHeatExchanger('heatexchanger_e', nodes_num=40)
        self.heatexchanger_f = HeatExchanger('heatexchanger_f', nodes_num=40)
        self.heatexchanger_g = ExtractHeatExchanger('heatexchanger_g', nodes_num=40)
        self.heatexchanger_h = HeatExchanger('heatexchanger_h', nodes_num=40)
        self.heatexchanger_i = ExtractHeatExchanger('heatexchanger_i', nodes_num=40)
        self.heatexchanger_j = HeatExchanger('heatexchanger_j', nodes_num=40)
        self.heatexchanger_k = ExtractHeatExchanger('heatexchanger_k', nodes_num=40)
        # evaporator
        self.evaporator = Evaporator('evaporator', nodes_num=40)
        # condenser
        self.condenser = Condenser('condenser', nodes_num=40)
        # evaporator drum
        self.evaporator_drum = EvaporateTank('evaporator_drum')
        # deaerator
        self.deaerator = Deaerator('deaerator')
        # turbine
        self.hp_turbine1 = Turbine('hp_turbine1')
        self.hp_turbine2 = Turbine('hp_turbine2')
        self.lp_turbine1 = Turbine('lp_turbine1')
        self.lp_turbine2 = Turbine('lp_turbine2')
        self.lp_turbine3 = Turbine('lp_turbine3')
        self.lp_turbine4 = Turbine('lp_turbine4')
        self.lp_turbine5 = Turbine('lp_turbine5')
        self.gas_turbine = Turbine('gas_turbine')
        # compressor
        self.air_compressor = Compressor('air_compressor')
        # pump
        self.condense_pump = Pump('condense_pump')
        self.steam_recycle_pump = Pump('steam_recycle_pump')  # used for evaporate module
        self.water_recycle_pump = Pump('water_recycle_pump')  # used for deaerator
        self.oil_recycle_pump = Pump('oil_recycle_pump')  # used for solar thermal cycle
        # heat storage tank
        self.hot_salt_tank = HeatStorageTank('hot_salt_tank')
        self.cold_salt_tank = HeatStorageTank('cold_salt_tank')
        # mass flow amplifier
        self.af_solar_in = FlowAmplifier('af_solar_in')
        self.af_solar_out = FlowAmplifier('af_solar_out')
        self.af_boiler_hot_in = FlowAmplifier('af_boiler_hot_in')
        self.af_boiler_hot_out = FlowAmplifier('af_boiler_hot_out')
        self.af_boiler_cold_in = FlowAmplifier('af_boiler_cold_in')
        self.af_boiler_cold_out = FlowAmplifier('af_boiler_cold_out')
        self.af_reheater_hot_in = FlowAmplifier('af_reheater_hot_in')
        self.af_reheater_hot_out = FlowAmplifier('af_reheater_hot_out')
        self.af_reheater_cold_in = FlowAmplifier('af_reheater_cold_in')
        self.af_reheater_cold_out = FlowAmplifier('af_reheater_cold_out')
        # valve
        self.valve_50_51 = Valve('valve_50_51')
        self.valve_52_53 = Valve('valve_52_53')
        self.valve_76_77 = Valve('valve_76_77')
        self.valve_73_74 = Valve('valve_73_74')
        self.valve_69_71 = Valve('valve_69_71')
        # cycle closer
        self.oil_cycle_closer = CycleCloser('oil_cycle_closer')
        self.steam_cycle_closer = CycleCloser('steam_cycle_closer')
        # distributor
        # merge
        self.merge_8_10_11 = Merge('merge_8_10_11', num_in=2)
        self.merge_13_14_21 = Merge('merge_13_14_21', num_in=2)
        self.merge_36_80_81 = Merge('merge_36_80_81', num_in=2)
        self.merge_19_82_83 = Merge('merge_19_82_83', num_in=2)
        self.merge_67_68_69 = Merge('merge_67_68_69', num_in=2)
        self.merge_heater_k = Merge('merge_heater_k', num_in=2)
        self.merge_heater_j = Merge('merge_heater_j', num_in=2)
        self.merge_deaerator = Merge('merge_deaerator', num_in=2)
        self.merge_heater_h = Merge('merge_heater_h', num_in=2)
        # splitter
        self.splitter_11_12_13 = Splitter('splitter_11_12_13', num_out=2)
        self.splitter_21_22_23 = Splitter('splitter_21_22_23', num_out=2)
        self.splitter_20_81_82 = Splitter('splitter_20_81_82', num_out=2)
        self.splitter_7_84_85 = Splitter('splitter_7_84_85', num_out=2)
        self.splitter_h1 = Splitter('splitter_h1', num_out=2)
        self.splitter_42_47_48 = Splitter('splitter_42_47_48', num_out=2)
        self.splitter_l1 = Splitter('splitter_l1', num_out=2)
        self.splitter_l2 = Splitter('splitter_l2', num_out=2)
        self.splitter_l3 = Splitter('splitter_l3', num_out=2)
        self.splitter_l4 = Splitter('splitter_l4', num_out=2)
        # source
        self.air_source = Source('air_source')
        self.fuel_source = Source('fuel_source')
        self.cooling_water_source = Source('cooling_water_source')
        # sink
        self.gas_sink = Sink('gas_sink')
        self.evaporator_drain = Sink('evaporator_drain')
        self.deaerator_drain = Sink('deaerator_drain')
        self.cooling_water_sink = Sink('cooling_water_sink')

        ########################
        # connection
        self.c1 = Connection(self.air_source,'out1', self.air_compressor, 'in1', label='c1')
        self.c2 = Connection(self.fuel_source, 'out1', self.combustion, 'in2', label='c2')
        self.c3 = Connection(self.air_compressor, 'out1', self.combustion, 'in1', label='c3')
        self.c5 = Connection(self.combustion, 'out1', self.gas_turbine, 'in1', label='c5')
        self.c6 = Connection(self.gas_turbine, 'out1', self.heatexchanger_a, 'in1', label='c6')
        self.c7 = Connection(self.splitter_7_84_85, 'out1', self.heatexchanger_a, 'in2', label='c7')
        self.c8 = Connection(self.heatexchanger_a, 'out2', self.merge_8_10_11, 'in1', label='c8')
        self.c9 = Connection(self.heatexchanger_a, 'out1', self.gas_sink, 'in1', label='c9')
        self.c10 = Connection(self.af_solar_out, 'out1', self.merge_8_10_11, 'in2', label='c10')
        self.c11 = Connection(self.merge_8_10_11, 'out1', self.splitter_11_12_13, 'in1', label='c11')
        self.c12 = Connection(self.splitter_11_12_13, 'out2', self.heatexchanger_b, 'in1', label='c12')
        self.c13 = Connection(self.splitter_11_12_13, 'out1', self.merge_13_14_21, 'in1', label='c13')
        self.c14 = Connection(self.heatexchanger_c, 'out2', self.merge_13_14_21, 'in2', label='c14')
        self.c15 = Connection(self.heatexchanger_b, 'out2', self.hot_salt_tank, 'in1', label='c15')
        self.c16 = Connection(self.hot_salt_tank, 'out1', self.heatexchanger_c, 'in1', label='c16')
        self.c17 = Connection(self.cold_salt_tank, 'out1', self.heatexchanger_b, 'in2', label='c17')
        self.c18 = Connection(self.heatexchanger_c, 'out1', self.cold_salt_tank, 'in1', label='c18')
        self.c19 = Connection(self.heatexchanger_b, 'out1', self.merge_19_82_83, 'in2', label='c19')
        self.c20 = Connection(self.splitter_20_81_82, 'out2', self.heatexchanger_c, 'in2', label='c20')
        self.c21 = Connection(self.merge_13_14_21, 'out1', self.splitter_21_22_23, 'in1', label='c21')
        self.c22 = Connection(self.splitter_21_22_23, 'out2', self.af_reheater_hot_in, 'in1', label='c22')
        self.c23 = Connection(self.splitter_21_22_23, 'out1', self.af_boiler_hot_in, 'in1', label='c23')
        self.c24 = Connection(self.af_boiler_hot_in, 'out1', self.heatexchanger_d, 'in1', label='c24')
        self.c25 = Connection(self.heatexchanger_d, 'out2', self.af_boiler_cold_out, 'in1', label='c25')
        self.c26 = Connection(self.heatexchanger_d, 'out1', self.evaporator, 'in1', label='c26')
        self.c27 = Connection(self.evaporator_drum, 'out2', self.heatexchanger_d, 'in2', label='c27')
        self.c28 = Connection(self.evaporator, 'out2', self.evaporator_drum, 'in2', label='c28')
        self.c29 = Connection(self.steam_recycle_pump, 'out1', self.evaporator, 'in2', label='c29')
        self.c30 = Connection(self.evaporator_drum, 'out1', self.steam_recycle_pump, 'in1', label='c30')
        self.c31 = Connection(self.evaporator_drum, 'out3', self.evaporator_drain, 'in1', label='c31')
        self.c32 = Connection(self.heatexchanger_e, 'out2', self.evaporator_drum, 'in1', label='c32')
        self.c33 = Connection(self.evaporator, 'out1', self.heatexchanger_e, 'in1', label='c33')
        self.c34 = Connection(self.heatexchanger_e, 'out1', self.af_boiler_hot_out, 'in1', label='c34')
        self.c35 = Connection(self.af_boiler_cold_in, 'out1', self.heatexchanger_e, 'in2', label='c35')
        self.c36 = Connection(self.af_boiler_hot_out, 'out1', self.merge_36_80_81, 'in1', label='c36')
        self.c37 = Connection(self.af_reheater_hot_in, 'out1', self.heatexchanger_f, 'in1', label='c37')
        self.c38 = Connection(self.af_reheater_cold_in, 'out1', self.heatexchanger_f, 'in2', label='c38')
        self.c39 = Connection(self.heatexchanger_f, 'out1', self.af_reheater_hot_out, 'in1', label='c39')
        self.c40 = Connection(self.heatexchanger_f, 'out2', self.af_reheater_cold_out, 'in1', label='c40')
        self.c41 = Connection(self.af_reheater_cold_out, 'out1', self.lp_turbine1, 'in1', label='c41')
        self.c42 = Connection(self.splitter_42_47_48, 'out1', self.af_reheater_cold_in, 'in1', label='c42')
        self.c43 = Connection(self.heatexchanger_g, 'out2', self.af_boiler_cold_in, 'in1', label='c43')
        self.c44 = Connection(self.af_boiler_cold_out, 'out1', self.hp_turbine1, 'in1', label='c44')
        self.h1_out = Connection(self.hp_turbine1, 'out1', self.splitter_h1, 'in1', label='h1_out')
        self.c45 = Connection(self.splitter_h1, 'out1', self.hp_turbine2, 'in1', label='c45')
        self.c46 = Connection(self.splitter_h1, 'out2', self.heatexchanger_g, 'in1', label='c46')
        self.c47 = Connection(self.hp_turbine2, 'out1', self.splitter_42_47_48, 'in1', label='c47')
        self.c48 = Connection(self.splitter_42_47_48, 'out2', self.merge_heater_h, 'in2', label='c48')
        self.c49 = Connection(self.heatexchanger_h, 'out2', self.heatexchanger_g, 'in2', label='c49')
        self.c50 = Connection(self.heatexchanger_g, 'out1', self.valve_50_51, 'in1', label='c50')
        self.c51 = Connection(self.valve_50_51, 'out1', self.merge_heater_h, 'in1', label='c51')
        self.heater_h_hot_in = Connection(self.merge_heater_h, 'out1', self.heatexchanger_h, 'in1', label='heater_h_hot_in')
        self.c52 = Connection(self.heatexchanger_h, 'out1', self.valve_52_53, 'in1', label='c52')
        self.c53 = Connection(self.valve_52_53, 'out1', self.merge_deaerator, 'in2', label='c53')
        self.c54 = Connection(self.deaerator, 'out1', self.steam_cycle_closer, 'in1', label='c54')
        self.d54 = Connection(self.steam_cycle_closer, 'out1', self.water_recycle_pump, 'in1', label='d54')
        self.c55 = Connection(self.water_recycle_pump, 'out1', self.heatexchanger_h, 'in2', label='c55')
        self.l1_out = Connection(self.lp_turbine1, 'out1', self.splitter_l1, 'in1', label='l1_out')
        self.c56 = Connection(self.splitter_l1, 'out2', self.deaerator, 'in2', label='c56')
        self.deaerator_cold_in = Connection(self.merge_deaerator, 'out1', self.deaerator, 'in1', label='deaerator_cold_in')
        self.c57 = Connection(self.splitter_l1, 'out1', self.lp_turbine2, 'in1', label='c57')
        self.l2_out = Connection(self.lp_turbine2, 'out1', self.splitter_l2, 'in1', label='l2_out')
        self.c58 = Connection(self.splitter_l2, 'out2', self.heatexchanger_i, 'in1', label='c58')
        self.c59 = Connection(self.splitter_l2, 'out1', self.lp_turbine3, 'in1', label='c59')
        self.l3_out = Connection(self.lp_turbine3, 'out1', self.splitter_l3, 'in1', label='l3_out')
        self.c60 = Connection(self.splitter_l3, 'out2', self.merge_heater_j, 'in1', label='c60')
        self.c61 = Connection(self.splitter_l3, 'out1', self.lp_turbine4, 'in1', label='c61')
        self.l4_out = Connection(self.lp_turbine4, 'out1', self.splitter_l4, 'in1', label='l4_out')
        self.c62 = Connection(self.splitter_l4, 'out2', self.merge_heater_k, 'in1', label='c62')
        self.c63 = Connection(self.splitter_l4, 'out1', self.lp_turbine5, 'in1', label='c63')
        self.c64 = Connection(self.lp_turbine5, 'out1', self.condenser, 'in1', label='c64')
        self.c65 = Connection(self.condenser, 'out2', self.cooling_water_sink, 'in1', label='c65')
        self.c66 = Connection(self.cooling_water_source, 'out1', self.condenser, 'in2', label='c66')
        self.c67 = Connection(self.condenser, 'out1', self.merge_67_68_69, 'in1', label='c67')
        self.c68 = Connection(self.merge_67_68_69, 'out1', self.condense_pump, 'in1', label='c68')
        self.c69 = Connection(self.valve_69_71, 'out1', self.merge_67_68_69, 'in2', label='c69')
        self.c70 = Connection(self.condense_pump, 'out1', self.heatexchanger_k, 'in2', label='c70')
        self.c71 = Connection(self.heatexchanger_k, 'out1', self.valve_69_71, 'in1', label='c71')
        self.c72 = Connection(self.heatexchanger_k, 'out2', self.heatexchanger_j, 'in2', label='c72')
        self.c73 = Connection(self.valve_73_74, 'out1', self.merge_heater_k, 'in2', label='c73')
        self.heater_k_hot_in = Connection(self.merge_heater_k, 'out1', self.heatexchanger_k, 'in1', label='heater_k_hot_in')
        self.c74 = Connection(self.heatexchanger_j, 'out1', self.valve_73_74, 'in1', label='c74')
        self.c75 = Connection(self.heatexchanger_j, 'out2', self.heatexchanger_i, 'in2', label='c75')
        self.c76 = Connection(self.valve_76_77, 'out1', self.merge_heater_j, 'in2', label='c76')
        self.heater_j_hot_in = Connection(self.merge_heater_j, 'out1', self.heatexchanger_j, 'in1', label='heater_j_hot_in')
        self.c77 = Connection(self.heatexchanger_i, 'out1', self.valve_76_77, 'in1', label='c77')
        self.c78 = Connection(self.heatexchanger_i, 'out2', self.merge_deaerator, 'in1', label='c78')
        self.c79 = Connection(self.deaerator, 'out2', self.deaerator_drain, 'in1', label='c79')
        self.c80 = Connection(self.af_reheater_hot_out, 'out1', self.merge_36_80_81, 'in2', label='c80')
        self.c81 = Connection(self.merge_36_80_81, 'out1', self.splitter_20_81_82, 'in1', label='c81')
        self.c82 = Connection(self.splitter_20_81_82, 'out1', self.merge_19_82_83, 'in1', label='c82')
        self.c83 = Connection(self.merge_19_82_83, 'out1', self.oil_recycle_pump, 'in1', label='c83')
        self.c84 = Connection(self.oil_recycle_pump, 'out1', self.oil_cycle_closer, 'in1', label='c84')
        self.d84 = Connection(self.oil_cycle_closer, 'out1', self.splitter_7_84_85, 'in1', label='d84')
        self.c85 = Connection(self.splitter_7_84_85, 'out2', self.af_solar_in, 'in1', label='c85')
        self.c86 = Connection(self.af_solar_in, 'out1', self.solar_collector4, 'in1', label='c86')
        self.c87 = Connection(self.solar_collector4, 'out1', self.solar_collector3, 'in1', label='c87')
        self.c88 = Connection(self.solar_collector3, 'out1', self.solar_collector2, 'in1', label='c88')
        self.c89 = Connection(self.solar_collector2, 'out1', self.solar_collector1, 'in1', label='c89')
        self.c90 = Connection(self.solar_collector1, 'out1', self.af_solar_out, 'in1', label='c90')
        # add connection to network
        self.nw.add_conns(self.c1, self.c2, self.c3, self.c5, self.c6, self.c7, self.c8, self.c9, self.c10, self.c11,
                          self.c12, self.c13, self.c14, self.c15, self.c16, self.c17, self.c18, self.c19, self.c20, self.c21,
                          self.c22, self.c23, self.c24, self.c25, self.c26, self.c27, self.c28, self.c29, self.c30, self.c31,
                          self.c32, self.c33, self.c34, self.c35, self.c36, self.c37, self.c38, self.c39, self.c40, self.c41,
                          self.c42, self.c43, self.c44, self.c45, self.c46, self.c47, self.c48, self.c49, self.c50, self.c51,
                          self.c52, self.c53, self.c54, self.c55, self.c56, self.c57, self.c58, self.c59, self.c60, self.c61,
                          self.c62, self.c63, self.c64, self.c65, self.c66, self.c67, self.c68, self.c69, self.c70, self.c71,
                          self.c72, self.c73, self.c74, self.c75, self.c76, self.c77, self.c78, self.c79, self.c80, self.c81,
                          self.c82, self.c83, self.c84, self.c85, self.c86, self.c87, self.c88, self.c89, self.c90, self.d54,
                          self.d84, self.h1_out, self.heater_h_hot_in, self.deaerator_cold_in,
                          self.l1_out, self.l2_out, self.l3_out, self.l4_out, self.heater_k_hot_in, self.heater_j_hot_in)






    def info_module(self):
        logger.define_logging(
            logpath=f"{self.name}_loggings", log_the_path=True, log_the_version=True,
            screen_level=logging.INFO, file_level=logging.DEBUG)