from Aurora.components import CycleCloser
from Aurora.components import Sink
from Aurora.components import Source
from Aurora.components import Condenser
from Aurora.components import Deaerator, DeaeratorSimple
from Aurora.components import Desuperheater
from Aurora.components import SimpleHeatExchanger, HeatExchanger, Evaporator, ExtractHeatExchanger, BoilerSimple, OverHeater
from Aurora.components import DiabaticCombustionChamber
from Aurora.components import Merge
from Aurora.components import Splitter
from Aurora.components import DropletSeparator, Drum, EvaporateTank
from Aurora.components import Valve
from Aurora.components import Pump
from Aurora.components import Compressor
from Aurora.components import Turbine
from Aurora.connections import Bus
from Aurora.connections import Connection
from Aurora.networks import Network
from Aurora.connections import Bus

from Aurora.tools.characteristics import load_default_char
from Aurora.tools.characteristics import CharLine
from Aurora.tools import logger
import logging

from Aurora.tools.helpers import AURORANetworkError

import numpy as np

class PureSteamCyclePlant1:
    def __init__(self, name):
        self.name = name

        logger.define_logging(
            logpath=f"{self.name}_loggings", log_the_path=True, log_the_version=True,
            screen_level=logging.INFO, file_level=logging.DEBUG)

        self.nw = Network(p_unit="bar", T_unit='C', h_unit="kJ / kg", m_unit='kg / s', iterinfo=True)

        ##############################################################################
        ##############################################################################
        # # # # component
        # # # steam section
        # heat source
        self.boiler = BoilerSimple('boiler', num_side=2)
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
        # condenser
        self.condenser = Condenser('condenser', nodes_num=40)
        # deaerator
        self.deaerator = Deaerator('deaerator')
        # turbine
        self.turbine1 = Turbine('turbine1')
        self.turbine2 = Turbine('turbine2')
        self.turbine3 = Turbine('turbine3')
        self.turbine4 = Turbine('turbine4')
        self.turbine5 = Turbine('turbine5')
        self.turbine6 = Turbine('turbine6')
        self.turbine7 = Turbine('turbine7')
        # pump
        self.recycle_pump = Pump('recycle_pump')
        self.condense_pump = Pump('condense_pump')
        # valve
        self.valve_3_4 = Valve('valve_3_4')
        self.valve_5_6 = Valve('valve_5_6')
        self.valve_26_27 = Valve('valve_26_27')
        self.valve_25_30 = Valve('valve_25_30')
        self.valve_34_35 = Valve('valve_34_35')
        self.valve_42_43 = Valve('valve_42_43')
        self.valve_47_48 = Valve('valve_47_48')
        self.valve_49_50 = Valve('valve_49_50')
        self.valve_51_53 = Valve('valve_51_53')
        self.valve_58_60 = Valve('valve_58_60')
        # splitter
        self.boiler_splitter = Splitter('boiler_splitter', num_out=2)
        self.turbine1_splitter = Splitter('turbine1_splitter', num_out=2)
        self.turbine2_splitter = Splitter('turbine2_splitter', num_out=2)
        self.turbine3_splitter = Splitter('turbine3_splitter', num_out=2)
        self.turbine4_splitter = Splitter('turbine4_splitter', num_out=2)
        self.turbine5_splitter = Splitter('turbine5_splitter', num_out=2)
        self.turbine6_splitter = Splitter('turbine6_splitter', num_out=2)
        self.split_8_61_64 = Splitter('split_8_61_64', num_out=2)
        self.split_12_48_49 = Splitter('split_12_48_49', num_out=2)
        self.split_16_38_39 = Splitter('split_16_38_39', num_out=2)
        # merger
        self.merge_4_5_65 = Merge('merge_4_5_65', num_in=2)
        self.merge_47_65_40 = Merge('merge_47_65_40', num_in=2)
        self.merge_39_40_28 = Merge('merge_39_40_28', num_in=2)
        self.merge_25_26_24 = Merge('merge_25_26_24', num_in=2)
        self.condense_merge = Merge('condense_merge', num_in=2)
        self.heater_i_merge = Merge('heater_i_merge', num_in=2)
        self.heater_g_merge = Merge('heater_g_merge', num_in=2)
        self.deaerator_merge = Merge('deaerator_merge', num_in=2)
        self.heater_b_merge = Merge('heater_b_merge', num_in=2)
        # distributor
        self.cycle_closer = CycleCloser('cycle_closer')
        self.cooling_water_source = Source('cooling_water_source')
        self.cooling_water_sink = Sink('cooling_water_sink')
        self.deaerator_sink = Sink('deaerator_sink')
        ##############################################################################
        ##############################################################################
        # # # # connection
        self.c1 = Connection(self.boiler, 'out1', self.turbine2, 'in1', label='c1')
        self.c2 = Connection(self.boiler, 'out2', self.boiler_splitter, 'in1', label='c2')
        self.c3 = Connection(self.boiler_splitter, 'out1', self.valve_3_4, 'in1', label='c3')
        self.c4 = Connection(self.valve_3_4, 'out1', self.merge_4_5_65, 'in1', label='c4')
        self.c5 = Connection(self.valve_5_6, 'out1', self.merge_4_5_65, 'in2', label='c5')
        self.c6 = Connection(self.turbine1_splitter, 'out1', self.valve_5_6, 'in1', label='c6')
        self.c7 = Connection(self.boiler_splitter, 'out2', self.turbine1, 'in1', label='c7')
        self.turbine1_out = Connection(self.turbine1, 'out1', self.turbine1_splitter, 'in1', label='turbine1_out')
        self.c8 = Connection(self.turbine1_splitter, 'out2', self.split_8_61_64, 'in1', label='c8')
        self.turbine2_out = Connection(self.turbine2, 'out1', self.turbine2_splitter, 'in1', label='turbine2_out')
        self.c9 = Connection(self.turbine2_splitter, 'out1', self.turbine3, 'in1', label='c9')
        self.c10 = Connection(self.turbine2_splitter, 'out2', self.heatexchanger_d, 'in1', label='c10')
        self.turbine3_out = Connection(self.turbine3, 'out1', self.turbine3_splitter, 'in1', label='turbine3_out')
        self.c11 = Connection(self.turbine3_splitter, 'out1', self.turbine4, 'in1', label='c11')
        self.c12 = Connection(self.turbine3_splitter, 'out2', self.split_12_48_49, 'in1', label='c12')
        self.turbine4_out = Connection(self.turbine4, 'out1', self.turbine4_splitter, 'in1', label='turbine4_out')
        self.c13 = Connection(self.turbine4_splitter, 'out1', self.turbine5, 'in1', label='c13')
        self.c14 = Connection(self.turbine4_splitter, 'out2', self.heatexchanger_e, 'in1', label='c14')
        self.turbine5_out = Connection(self.turbine5, 'out1', self.turbine5_splitter, 'in1', label='turbine5_out')
        self.c15 = Connection(self.turbine5_splitter, 'out1', self.turbine6, 'in1', label='c15')
        self.c16 = Connection(self.turbine5_splitter, 'out2', self.split_16_38_39, 'in1', label='c16')
        self.turbine6_out = Connection(self.turbine6, 'out1', self.turbine6_splitter, 'in1', label='turbine6_out')
        self.c17 = Connection(self.turbine6_splitter, 'out1', self.turbine7, 'in1', label='c17')
        self.c18 = Connection(self.turbine6_splitter, 'out2', self.heater_i_merge, 'in1', label='c18')
        self.c19 = Connection(self.turbine7, 'out1', self.condense_merge, 'in1', label='c19')
        self.condenser_hot_in = Connection(self.condense_merge, 'out1', self.condenser, 'in1', label='condenser_hot_in')
        self.c20 = Connection(self.condenser, 'out2', self.cooling_water_sink, 'in1', label='c20')
        self.c21 = Connection(self.cooling_water_source, 'out1', self.condenser, 'in2', label='c21')
        self.c22 = Connection(self.condenser, 'out1', self.condense_pump, 'in1', label='c22')
        self.c23 = Connection(self.condense_pump, 'out1', self.heatexchanger_k, 'in2', label='c23')
        self.c24 = Connection(self.merge_25_26_24, 'out1', self.condense_merge, 'in2', label='c24')
        self.c25 = Connection(self.valve_25_30, 'out1', self.merge_25_26_24, 'in1', label='c25')
        self.c26 = Connection(self.valve_26_27, 'out1', self.merge_25_26_24, 'in2', label='c26')
        self.c27 = Connection(self.heatexchanger_k, 'out1', self.valve_26_27, 'in1', label='c27')
        self.c28 = Connection(self.merge_39_40_28, 'out1', self.heatexchanger_k, 'in1', label='c28')
        self.c29 = Connection(self.heatexchanger_k, 'out2', self.heatexchanger_j, 'in2', label='c29')
        self.c30 = Connection(self.heatexchanger_j, 'out1', self.valve_25_30, 'in1', label='c30')
        self.c31 = Connection(self.heatexchanger_i, 'out1', self.heatexchanger_j, 'in1', label='c31')
        self.c32 = Connection(self.heatexchanger_j, 'out2', self.heatexchanger_i, 'in2', label='c32')
        self.c33 = Connection(self.heatexchanger_i, 'out2', self.heatexchanger_h, 'in2', label='c33')
        self.c34 = Connection(self.valve_34_35, 'out1', self.heater_i_merge, 'in2', label='c34')
        self.heater_i_hot_in = Connection(self.heater_i_merge, 'out1', self.heatexchanger_i, 'in1', label='heater_i_hot_in')
        self.c35 = Connection(self.heatexchanger_h, 'out1', self.valve_34_35, 'in1', label='c35')
        self.c36 = Connection(self.heatexchanger_g, 'out1', self.heatexchanger_h, 'in1', label='c36')
        self.c37 = Connection(self.heatexchanger_h, 'out2', self.heatexchanger_g, 'in2', label='c37')
        self.c38 = Connection(self.split_16_38_39, 'out1', self.heater_g_merge, 'in1', label='c38')
        self.heater_g_hot_in = Connection(self.heater_g_merge, 'out1', self.heatexchanger_g, 'in1', label='heater_g_hot_in')
        self.c39 = Connection(self.split_16_38_39, 'out2', self.merge_39_40_28, 'in2', label='c39')
        self.c40 = Connection(self.merge_47_65_40, 'out1', self.merge_39_40_28, 'in1', label='c40')
        self.c41 = Connection(self.heatexchanger_g, 'out2', self.heatexchanger_f, 'in2', label='c41')
        self.c42 = Connection(self.valve_42_43, 'out1', self.heater_g_merge, 'in2', label='c42')
        self.c43 = Connection(self.heatexchanger_f, 'out1', self.valve_42_43, 'in1', label='c43')
        self.c44 = Connection(self.heatexchanger_e, 'out1', self.heatexchanger_f, 'in1', label='c44')
        self.c45 = Connection(self.heatexchanger_f, 'out2', self.heatexchanger_e, 'in2', label='c45')
        self.c46 = Connection(self.heatexchanger_e, 'out2', self.deaerator_merge, 'in2', label='c46')
        self.c47 = Connection(self.valve_47_48, 'out1', self.merge_47_65_40, 'in2', label='c47')
        self.c48 = Connection(self.split_12_48_49, 'out2', self.valve_47_48, 'in1', label='c48')
        self.c49 = Connection(self.split_12_48_49, 'out1', self.valve_49_50, 'in1', label='c49')
        self.c50 = Connection(self.valve_49_50, 'out1', self.deaerator, 'in2', label='c50')
        self.c51 = Connection(self.valve_51_53, 'out1', self.deaerator_merge, 'in1', label='c51')
        self.deaerator_cold_in = Connection(self.deaerator_merge, 'out1', self.deaerator, 'in1', label='deaerator_cold_in')
        self.c52 = Connection(self.deaerator, 'out1', self.recycle_pump, 'in1', label='c52')
        self.deaerator_drain = Connection(self.deaerator, 'out2', self.deaerator_sink, 'in1', label='deaerator_drain')
        self.c53 = Connection(self.heatexchanger_a, 'out1', self.valve_51_53, 'in1', label='c53')
        self.c54 = Connection(self.recycle_pump, 'out1', self.heatexchanger_a, 'in2', label='c54')
        self.c55 = Connection(self.heatexchanger_b, 'out1', self.heatexchanger_a, 'in1', label='c55')
        self.c56 = Connection(self.heatexchanger_a, 'out2', self.heatexchanger_b, 'in2', label='c56')
        self.heater_b_hot_in = Connection(self.heater_b_merge, 'out1', self.heatexchanger_b, 'in1', label='heater_b_hot_in')
        self.c57 = Connection(self.heatexchanger_d, 'out1', self.heater_b_merge, 'in1', label='c57')
        self.c58 = Connection(self.valve_58_60, 'out1', self.heater_b_merge, 'in2', label='c58')
        self.c59 = Connection(self.heatexchanger_b, 'out2', self.heatexchanger_c, 'in2', label='c59')
        self.c60 = Connection(self.heatexchanger_c, 'out1', self.valve_58_60, 'in1', label='c60')
        self.c61 = Connection(self.split_8_61_64, 'out2', self.heatexchanger_c, 'in1', label='c61')
        self.c62 = Connection(self.heatexchanger_c, 'out2', self.heatexchanger_d, 'in2', label='c62')
        self.c63 = Connection(self.heatexchanger_d, 'out2', self.boiler, 'in2', label='c63')
        self.c64 = Connection(self.split_8_61_64, 'out1', self.boiler, 'in1', label='c64')
        self.c65 = Connection(self.merge_4_5_65, 'out1', self.cycle_closer, 'in1', label='c65')
        self.d65 = Connection(self.cycle_closer, 'out1', self.merge_47_65_40, 'in1', label='d65')
        # add connection to network
        self.nw.add_conns(self.c1, self.c2, self.c3, self.c4, self.c5, self.c6, self.c7, self.c8, self.c9, self.c10, self.c11, self.c12,
                          self.c13, self.c14, self.c15, self.c16, self.c17, self.c18, self.c19, self.c20, self.c21, self.c22, self.c23, self.c24,
                          self.c25, self.c26, self.c27, self.c28, self.c29, self.c30, self.c31, self.c32, self.c33, self.c34, self.c35, self.c36,
                          self.c37, self.c38, self.c39, self.c40, self.c41, self.c42, self.c43, self.c44, self.c45, self.c46, self.c47, self.c48,
                          self.c49, self.c50, self.c51, self.c52, self.c53, self.c54, self.c55, self.c56, self.c57, self.c58, self.c59, self.c60,
                          self.c61, self.c62, self.c63, self.c64, self.c65, self.deaerator_drain, self.d65,
                          self.turbine1_out, self.turbine2_out, self.turbine3_out, self.turbine4_out, self.turbine5_out, self.turbine6_out,
                          self.condenser_hot_in, self.heater_i_hot_in, self.heater_g_hot_in, self.heater_b_hot_in,
                          self.deaerator_cold_in)
        ##############################################################################
        ##############################################################################


    def set_properties_(self):
        # set properties
        # components
        self.boiler.set_attr(T_out1=530, T_out2=530, dp1=3.2, dp2=2)
        self.heatexchanger_a.set_attr(dp1=0, dp2=0.05, DTL=5)
        self.heatexchanger_b.set_attr(dp1=0, dp2=0.05, DTU_sh=5)
        self.heatexchanger_c.set_attr(dp1=0, dp2=0.05, DTU_sh=5)
        self.heatexchanger_d.set_attr(dp1=0, dp2=0.05, DTNS=100)
        self.heatexchanger_e.set_attr(dp1=0, dp2=0.05, DTU_sh=5)
        self.heatexchanger_f.set_attr(dp1=0, dp2=0.05, DTL=5)
        self.heatexchanger_g.set_attr(dp1=0, dp2=0.05, DTU_sh=5)
        self.heatexchanger_h.set_attr(dp1=0, dp2=0.05, DTL=5)
        self.heatexchanger_i.set_attr(dp1=0, dp2=0.05, DTU_sh=5)
        self.heatexchanger_j.set_attr(dp1=0, dp2=0.05, DTL=5)
        self.heatexchanger_k.set_attr(dp1=0, dp2=0.05, DTU_sh=70)
        self.condenser.set_attr(dp1=0, dp2=0.05, DTU_sh=5)
        self.deaerator.set_attr(dp1=0, dp2=0)
        self.turbine1.set_attr(eta_s=0.88)
        self.turbine2.set_attr(eta_s=0.88)
        self.turbine3.set_attr(eta_s=0.88)
        self.turbine4.set_attr(eta_s=0.88)
        self.turbine5.set_attr(eta_s=0.88)
        self.turbine6.set_attr(eta_s=0.88)
        self.turbine7.set_attr(eta_s=0.88)
        self.condense_pump.set_attr(eta_s=0.8)
        self.recycle_pump.set_attr(eta_s=0.8)
        # connections
        self.c2.set_attr(m=580.486, p=191, fluid={"Water": 1})
        self.c1.set_attr(p=38)
        self.c9.set_attr(p=21.5)
        self.c11.set_attr(p=10.24)
        self.c13.set_attr(p=3.9)
        self.c15.set_attr(p=1.27)
        self.c17.set_attr(p=0.251)
        self.c3.set_attr(m=0.008)
        self.c6.set_attr(m=0.01)
        self.c48.set_attr(m=0.016)
        self.c19.set_attr(p=0.058)
        self.c23.set_attr(p=4)
        self.c21.set_attr(p=3, T=15, fluid={"Water": 1})
        self.deaerator_drain.set_attr(m=0)

    def set_off_design_properties_(self):
        # heat exchanger
        kA_charline1 = CharLine(x=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
                               y=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1, 1.04, 1.07])
        kA_charline2 = CharLine(x=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
                               y=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        kA_charline1_con = CharLine(x=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
                                y=[0.56, 0.6, 0.65, 0.7, 0.72, 0.74, 0.78, 1, 0.8, 0.79])
        self.condenser.set_attr(design=['DTU_sh'], offdesign=['KDTA'],
                                KDTA_fit='charline', KDTA_char2=kA_charline1_con) #
        self.heatexchanger_a.set_attr(design=['DTL'], offdesign=['KDTA'],
                                      KDTA_fit='charline', dp2_fit='default',
                                      KDTA_char1=kA_charline2, KDTA_char2=kA_charline1)
        kA_charline1_b = CharLine(x=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
                                y=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1, 1.04, 1.07])
        self.heatexchanger_b.set_attr(design=['DTU_sh'], offdesign=['KDTA'],
                                      KDTA_fit='charline', dp2_fit='default',
                                      KDTA_char1=kA_charline2, KDTA_char2=kA_charline1)
        kA_charline1_c = CharLine(x=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
                                y=[0.56, 0.65, 0.74, 0.84, 0.88, 0.93, 0.97, 1, 1.08, 1.07])
        self.heatexchanger_c.set_attr(design=['DTU_sh'], offdesign=['KDTA'],
                                      KDTA_fit='charline', dp2_fit='default',
                                      KDTA_char1=kA_charline2, KDTA_char2=kA_charline1)
        kA_charline1_d = CharLine(x=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
                                y=[0.52, 0.62, 0.72, 0.82, 0.87, 0.2, 0.97, 1, 1.06, 1.05])
        self.heatexchanger_d.set_attr(design=['DTNS'], offdesign=['KDTA'],
                                      KDTA_fit='charline', dp2_fit='default',
                                      KDTA_char1=kA_charline2, KDTA_char2=kA_charline1)
        kA_charline1_e = CharLine(x=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
                                y=[0.1, 0.15, 0.18, 0.2, 0.26, 0.32, 0.4, 1, 0.4, 0.34])
        kA_charline2_e = CharLine(x=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
                                  y=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.84, 1, 0.84, 0.8])  #
        self.heatexchanger_e.set_attr(design=['DTU_sh'], offdesign=['KDTA'],
                                      KDTA_fit='charline', dp2_fit='default',
                                      KDTA_char1=kA_charline2_e, KDTA_char2=kA_charline1_e)  #
        kA_charline1_f = CharLine(x=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
                                  y=[0.15, 0.2, 0.26, 0.32, 0.4, 0.85, 0.96, 1, 0.96, 0.95])
        kA_charline2_f = CharLine(x=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
                                  y=[0.15, 0.2, 0.26, 0.32, 0.4, 0.85, 0.96, 1, 0.96, 0.95])
        self.heatexchanger_f.set_attr(design=['DTL'], offdesign=['KDTA'],
                                      KDTA_fit='charline', dp2_fit='default',
                                      KDTA_char1=kA_charline2_f, KDTA_char2=kA_charline1_f)
        kA_charline1_g = CharLine(x=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
                                y=[0.1, 0.15, 0.18, 0.2, 0.26, 0.32, 0.4, 1, 0.4, 0.34])
        kA_charline2_g = CharLine(x=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
                                  y=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.86, 1, 0.86, 0.8])  #
        self.heatexchanger_g.set_attr(design=['DTU_sh'], offdesign=['KDTA'],
                                      KDTA_fit='charline', dp2_fit='default',
                                      KDTA_char1=kA_charline2_g, KDTA_char2=kA_charline1_g)
        kA_charline1_h = CharLine(x=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
                                  y=[0.52, 0.62, 0.72, 0.82, 0.87, 0.2, 0.97, 1, 1.16, 1.15])
        kA_charline2_h = CharLine(x=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
                                  y=[0.52, 0.62, 0.72, 0.82, 0.87, 0.2, 0.97, 1, 1.4, 1.5])
        self.heatexchanger_h.set_attr(design=['DTL'], offdesign=['KDTA'],
                                      KDTA_fit='charline', dp2_fit='default',
                                      KDTA_char1=kA_charline2_h, KDTA_char2=kA_charline1_h)
        kA_charline1_i = CharLine(x=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
                                y=[0.15, 0.2, 0.26, 0.32, 0.4, 0.45, 0.6, 1, 0.6, 0.55])
        kA_charline2_i = CharLine(x=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
                                  y=[0.15, 0.2, 0.26, 0.32, 0.4, 0.45, 0.6, 1, 0.6, 0.5])  #
        self.heatexchanger_i.set_attr(design=['DTU_sh'], offdesign=['KDTA'],
                                      KDTA_fit='charline', dp2_fit='default',
                                      KDTA_char1=kA_charline2_i, KDTA_char2=kA_charline1)
        kA_charline1_j = CharLine(x=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
                                  y=[0.52, 0.62, 0.72, 0.82, 0.87, 0.2, 0.97, 1, 1.16, 1.15])
        kA_charline2_j = CharLine(x=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
                                  y=[0.52, 0.62, 0.72, 0.82, 0.87, 0.92, 0.97, 1, 1.16, 1.15])
        self.heatexchanger_j.set_attr(design=['DTL'], offdesign=['KDTA'],
                                      KDTA_fit='charline', dp2_fit='default',
                                      KDTA_char1=kA_charline2_j, KDTA_char2=kA_charline1_j)
        kA_charline1_k = CharLine(x=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
                                  y=[0.1, 0.15, 0.18, 0.2, 0.22, 0.24, 0.3, 0.98, 0.3, 0.24])
        kA_charline2_k = CharLine(x=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
                                  y=[0.1, 0.15, 0.18, 0.2, 0.22, 0.24, 0.3, 1, 0.3, 0.24])  #
        self.heatexchanger_k.set_attr(design=['DTU_sh'], offdesign=['KDTA'],
                                      KDTA_fit='charline', dp2_fit='default',
                                      KDTA_char1=kA_charline2_k, KDTA_char2=kA_charline1_k)
        # turbine
        tur_charline = CharLine(x=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1, 1.05, 1.1],
                               y=[y+0.135 for y in [0.9, 0.92, 0.94, 0.96, 0.98, 0.99, 0.995, 1, 0.99, 0.98]])
        self.turbine1.set_attr(eta_s_char=tur_charline, eta_s_fit='charline')
        self.turbine2.set_attr(eta_s_char=tur_charline, eta_s_fit='charline')
        self.turbine3.set_attr(eta_s_char=tur_charline, eta_s_fit='charline')
        self.turbine4.set_attr(eta_s_char=tur_charline, eta_s_fit='charline')
        self.turbine5.set_attr(eta_s_char=tur_charline, eta_s_fit='charline')
        self.turbine6.set_attr(eta_s_char=tur_charline, eta_s_fit='charline')
        self.turbine7.set_attr(eta_s_char=tur_charline, eta_s_fit='charline')
        # pump
        pump_charline = CharLine(x=[0.5, 0.6, 0.7, 0.8, 1, 1.1],
                               y=[0.92, 0.94, 0.96, 0.98, 1, 0.98])
        self.condense_pump.set_attr(eta_s_char=pump_charline, eta_s_fit='charline')
        self.recycle_pump.set_attr(eta_s_char=pump_charline, eta_s_fit='charline')
        # set other properties due to condition
        self.set_off_design_1_()

    def set_off_design_1_(self):
        self.c1.set_attr(p=35.14)
        self.c9.set_attr(p=19.43)
        self.c11.set_attr(p=9.0)
        self.c13.set_attr(p=3.44)
        self.c15.set_attr(p=1.15)
        self.c17.set_attr(p=0.244)
        self.c19.set_attr(p=0.046)
        self.boiler.set_attr(T_out1=600, T_out2=600)
        self.c21.set_attr(T=10)
        self.c23.set_attr(p=4.3)
        pass

    def solve(self, mode='design', max_iter=50, algo_factor=0.1):
        self.set_properties_()
        if mode == 'offdesign':
            self.set_off_design_properties_()
        self.nw.solve(mode=mode, max_iter=max_iter, algo_factor=algo_factor,
                      plot_iteration=False, print_results=True,
                      design_path= f"{self.name}_design_",
                      init_path=f"{self.name}_design_"
                      )
        # self.nw.save(f"{self.name}_design_")
        # self.nw.save_csv(f"{self.name}_design_csv_")
        # self.nw.save_csv(f"{self.name}_offdesign_csv_")


if __name__ == '__main__':
    pure_steam_plant_model1 = PureSteamCyclePlant1('pure_steam_plant_model1')
    pure_steam_plant_model1.solve(max_iter=150, mode='offdesign', algo_factor=0.01)



