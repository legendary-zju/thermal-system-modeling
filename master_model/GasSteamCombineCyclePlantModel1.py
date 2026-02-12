from Aurora.components import CycleCloser, HeatExchanger, Evaporator, ExtractHeatExchanger
from Aurora.components import Sink
from Aurora.components import Source
from Aurora.components import Condenser
from Aurora.components import Deaerator, DeaeratorSimple
from Aurora.components import Desuperheater
from Aurora.components import SimpleHeatExchanger
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


class GasSteamCombineCyclePlant1:
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
        # # main steam heat exchanger
        # main steam section1
        self.main_heat_exchanger1 = HeatExchanger('main_heat_exchanger1', nodes_num=40)

        self.evaporator1 = Evaporator('evaporator1', nodes_num=40)
        self.evaporate_pump1 = Pump('evaporate_pump1')
        self.evaporate_tank1 = EvaporateTank('evaporate_tank1')
        self.sa_liquid1 = Sink('sa_liquid1')

        self.main_heat_exchanger2 = HeatExchanger('main_heat_exchanger2', nodes_num=40)
        # main steam section2
        self.evaporator2 = Evaporator('evaporator2', nodes_num=40)
        self.evaporate_pump2 = Pump('evaporate_pump2')
        self.evaporate_tank2 = EvaporateTank('evaporate_tank2')
        self.sa_liquid2 = Sink('sa_liquid2')

        self.main_heat_exchanger3 = HeatExchanger('main_heat_exchanger3', nodes_num=40)

        # # valve of main steam section1
        self.main_steam1_valve = Valve('main_steam1_valve')

        # # turbine
        self.turbine1 = Turbine('turbine1')
        self.turbine2 = Turbine('turbine2')
        self.turbine3 = Turbine('turbine3')
        self.turbine4 = Turbine('turbine4')
        self.turbine5 = Turbine('turbine5')

        # # steam merge of turbine1/2
        self.main_steam_merge = Merge('main_steam_merge', num_in=2)

        # # condenser
        self.condenser = Condenser('condenser', nodes_num=40)
        # recycle pump
        self.recycle_pump1 = Pump('recycle_pump1')
        self.feed_water_merge = Merge('feed_water_merge', num_in=2)
        self.feed_water = Source('feed_water')

        # # steam extract of turbine2
        self.extract_splitter_turbine2 = Splitter('extract_splitter_turbine2', num_out=2)
        self.extract_valve_turbine2 = Valve('extract_valve_turbine2')
        # deaerator
        #
        self.deaerator = Deaerator('deaerator')
        self.sa_liquid = Sink('sa_liquid')

        #######################################
        # # steam extract of turbine3
        self.extract_splitter_turbine3 = Splitter('extract_splitter_turbine3', num_out=2)
        self.extract_heat_exchanger1_turbine3 = ExtractHeatExchanger('extract_heat_exchanger1_turbine3', nodes_num=40)
        self.extract_heat_exchanger2_turbine3 = HeatExchanger('extract_heat_exchanger2_turbine3', nodes_num=40)
        self.extract_pump_turbine3 = Pump('extract_pump_turbine3')
        self.extract_merge_turbine3 = Merge('extract_merge_turbine3', num_in=2)

        # # steam extract of turbine4
        self.extract_splitter_turbine4 = Splitter('extract_splitter_turbine4', num_out=2)
        self.extract_heat_exchanger1_turbine4 = ExtractHeatExchanger('extract_heat_exchanger1_turbine4', nodes_num=40)
        self.extract_heat_exchanger2_turbine4 = HeatExchanger('extract_heat_exchanger2_turbine4', nodes_num=40)
        self.extract_pump_turbine4 = Pump('extract_pump_turbine4')
        self.extract_merge_turbine4 = Merge('extract_merge_turbine4', num_in=2)

        # # cycle closer
        self.cycle_closer = CycleCloser('cycle_closer')

        # # main water splitter
        self.main_steam_splitter = Splitter('main_steam_splitter', num_out=2)

        # # recycle to main steam pump
        self.recycle_pump21 = Pump('recycle_pump21')
        self.recycle_pump22 = Pump('recycle_pump22')

        # # # gas section
        self.fuel_source = Source('fuel_source')
        self.fuel_splitter = Splitter('fuel_splitter', num_out=2)

        # # fuel section1
        self.fuel_compressor = Compressor('fuel_compressor')
        # air section of fuel section1
        self.air_source = Source('air_source')
        self.air_compressor = Compressor('air_compressor')
        # combus section of fuel section1
        self.fuel_combustion = DiabaticCombustionChamber('fuel_combustion')
        self.gas_turbine = Turbine('gas_turbine')

        # # fuel section2
        self.fuel_add_valve = Valve('fuel_add_valve')
        # combustion section of fuel section2
        self.gas_drop_pressure_valve = Valve('gas_drop_pressure_valve')
        self.gas_re_combustion = DiabaticCombustionChamber('gas_re_combustion')
        self.gas_sink = Sink('gas_sink')

        # # # cooling water section
        self.cooling_water_source = Source('cooling_water_source')
        self.cooling_water_sink = Sink('cooling_water_sink')

        ##############################################################################
        ##############################################################################
        # # # # connection
        # # # gas cycle section
        self.fuel_source_splitter = Connection(self.fuel_source, 'out1', self.fuel_splitter, 'in1', label='fuel_source_splitter')

        # # fuel section1
        self.fuel1_splitter_compressor = Connection(self.fuel_splitter, 'out1', self.fuel_compressor, 'in1', label='fuel1_splitter_compressor')
        self.fuel1_compressor_combustion = Connection(self.fuel_compressor, 'out1', self.fuel_combustion, 'in2', label='fuel1_compressor_combustion')
        # fuel1 air section
        # air filter
        self.fuel1_air_source_compressor = Connection(self.air_source, 'out1', self.air_compressor, 'in1', label='fuel1_air_source_compressor')
        self.fuel1_air_compressor_combustion = Connection(self.air_compressor, 'out1', self.fuel_combustion, 'in1', label='fuel1_air_compressor_combustion')
        # gas turbine
        self.gas_turbine_in = Connection(self.fuel_combustion, 'out1', self.gas_turbine, 'in1', label='gas_turbine_in')

        # # fuel section2
        self.gas_turbine_out = Connection(self.gas_turbine, 'out1', self.gas_drop_pressure_valve, 'in1', label='gas_turbine_out')
        self.gas_re_combustion_in = Connection(self.gas_drop_pressure_valve, 'out1', self.gas_re_combustion, 'in1', label='gas_re_combustion_in')
        self.fuel2_splitter_valve = Connection(self.fuel_splitter, 'out2', self.fuel_add_valve, 'in1', label='fuel2_splitter_valve')
        self.fuel2_valve_combustion = Connection(self.fuel_add_valve, 'out1', self.gas_re_combustion, 'in2', label='fuel2_valve_combustion')

        self.nw.add_conns(self.fuel_source_splitter, self.fuel1_splitter_compressor, self.fuel1_compressor_combustion,
                          self.fuel1_air_source_compressor, self.fuel1_air_compressor_combustion, self.gas_turbine_in,
                          self.gas_turbine_out, self.gas_re_combustion_in, self.fuel2_splitter_valve, self.fuel2_valve_combustion)

        #########################################
        # # # main steam cycle section
        # # main steam heat exchanger section
        # main steam section1
        # gas section
        self.gas_deliver = Connection(self.gas_re_combustion, 'out1', self.main_heat_exchanger1, 'in1', label='gas_deliver')
        self.evaporator1_gas_in = Connection(self.main_heat_exchanger1, 'out1', self.evaporator1, 'in1', label='evaporator1_gas_in')
        self.evaporator1_gas_out = Connection(self.evaporator1, 'out1', self.main_heat_exchanger2, 'in1', label='evaporator1_gas_out')
        # steam section of main steam section1
        self.main_steam1_recycle = Connection(self.recycle_pump21, 'out1', self.main_heat_exchanger2, 'in2',
                                              label='main_steam1_recycle')
        self.evaporate_tank1_wator_in = Connection(self.main_heat_exchanger2, 'out2', self.evaporate_tank1, 'in1',
                                                   label='evaporate_tank1_wator_in')
        self.evaporate_tank1_wator_out = Connection(self.evaporate_tank1, 'out1', self.evaporate_pump1, 'in1',
                                                    label='evaporate_tank1_wator_out')
        self.evaporator1_steam_in = Connection(self.evaporate_pump1, 'out1', self.evaporator1, 'in2',
                                               label='evaporator1_steam_in')
        self.evaporator1_steam_out = Connection(self.evaporator1, 'out2', self.evaporate_tank1, 'in2',
                                                label='evaporator1_steam_out')
        self.evaporate_tank1_vapour_out = Connection(self.evaporate_tank1, 'out2', self.main_heat_exchanger1, 'in2',
                                                     label='evaporate_tank1_vapour_out')
        self.drain_liquid_tank1 = Connection(self.evaporate_tank1, 'out3', self.sa_liquid1, 'in1',
                                                     label='drain_liquid_tank1')

        # main steam section2
        # gas section
        self.evaporator2_gas_in = Connection(self.main_heat_exchanger2, 'out1', self.evaporator2, 'in1', label='evaporator2_gas_in')
        self.evaporator2_gas_out = Connection(self.evaporator2, 'out1', self.main_heat_exchanger3, 'in1', label='evaporator2_gas_out')
        self.gas_out = Connection(self.main_heat_exchanger3, 'out1', self.gas_sink, 'in1', label='gas_out')
        # steam section
        self.main_steam2_recycle = Connection(self.recycle_pump22, 'out1', self.main_heat_exchanger3, 'in2',
                                              label='main_steam2_recycle')
        self.evaporate_tank2_wator_in = Connection(self.main_heat_exchanger3, 'out2', self.evaporate_tank2, 'in1',
                                                   label='evaporate_tank2_wator_in')
        self.evaporate_tank2_wator_out = Connection(self.evaporate_tank2, 'out1', self.evaporate_pump2, 'in1',
                                                    label='evaporate_tank2_wator_out')
        self.evaporator2_steam_in = Connection(self.evaporate_pump2, 'out1', self.evaporator2, 'in2',
                                               label='evaporator2_steam_in')
        self.evaporator2_steam_out = Connection(self.evaporator2, 'out2', self.evaporate_tank2, 'in2',
                                                label='evaporator2_steam_out')
        self.evaporate_tank2_vapour_out = Connection(self.evaporate_tank2, 'out2', self.main_steam_merge, 'in2',
                                                     label='evaporate_tank2_vapour_out')
        self.drain_liquid_tank2 = Connection(self.evaporate_tank2, 'out3', self.sa_liquid2, 'in1',
                                             label='drain_liquid_tank2')

        self.nw.add_conns(self.gas_deliver, self.evaporator1_gas_in, self.evaporator1_gas_out, self.main_steam1_recycle,
                          self.evaporate_tank1_wator_in, self.evaporate_tank1_wator_out, self.evaporator1_steam_in,
                          self.evaporator1_steam_out, self.evaporate_tank1_vapour_out, self.drain_liquid_tank1,
                          self.evaporator2_gas_in, self.evaporator2_gas_out, self.gas_out, self.main_steam2_recycle,
                          self.evaporate_tank2_wator_in, self.evaporate_tank2_wator_out, self.drain_liquid_tank2,
                          self.evaporator2_steam_in, self.evaporator2_steam_out, self.evaporate_tank2_vapour_out)

        # # main steam to turbine section
        self.main_steam1_deliver = Connection(self.main_heat_exchanger1, 'out2', self.main_steam1_valve, 'in1',
                                              label='main_steam1_deliver')
        self.main_steam_turbine1 = Connection(self.main_steam1_valve, 'out1', self.turbine1, 'in1',
                                              label='main_steam_turbine1')
        self.turbine1_steam_out = Connection(self.turbine1, 'out1', self.main_steam_merge, 'in1',
                                             label='turbine1_steam_out')

        self.turbine2_steam_in = Connection(self.main_steam_merge, 'out1', self.turbine2, 'in1',
                                            label='turbine2_steam_in')
        self.turbine2_steam_out = Connection(self.turbine2, 'out1', self.extract_splitter_turbine2, 'in1',
                                             label='turbine2_steam_out')

        self.turbine3_steam_in = Connection(self.extract_splitter_turbine2, 'out1', self.turbine3, 'in1',
                                            label='turbine3_steam_in')
        self.turbine3_steam_out = Connection(self.turbine3, 'out1', self.extract_splitter_turbine3, 'in1',
                                             label='turbine3_steam_out')

        self.turbine4_steam_in = Connection(self.extract_splitter_turbine3, 'out1', self.turbine4, 'in1',
                                            label='turbine4_steam_in')
        self.turbine4_steam_out = Connection(self.turbine4, 'out1', self.extract_splitter_turbine4, 'in1',
                                             label='turbine4_steam_out')

        self.turbine5_steam_in = Connection(self.extract_splitter_turbine4, 'out1', self.turbine5, 'in1',
                                            label='turbine5_steam_in')
        self.turbine5_steam_out = Connection(self.turbine5, 'out1', self.condenser, 'in1',
                                             label='turbine5_steam_out')

        self.nw.add_conns(self.main_steam1_deliver, self.main_steam_turbine1, self.turbine1_steam_out, self.turbine2_steam_in,
                          self.turbine2_steam_out, self.turbine3_steam_in, self.turbine3_steam_out, self.turbine4_steam_in,
                          self.turbine4_steam_out, self.turbine5_steam_in, self.turbine5_steam_out)

        # # # cooling water section
        self.cooling_water_in = Connection(self.cooling_water_source, 'out1', self.condenser, 'in2', label='cooling_water_in')
        self.cooling_water_out = Connection(self.condenser, 'out2', self.cooling_water_sink, 'in1', label='cooling_water_out')
        # self.condenser_recycle = Connection(self.condenser, 'out1', self.recycle_pump1, 'in1',
        #                                     label='condenser_recycle')
        self.condenser_recycle = Connection(self.condenser, 'out1', self.cycle_closer, 'in1',
                                            label='condenser_recycle')
        self.recycle_pump = Connection(self.cycle_closer, 'out1', self.recycle_pump1, 'in1',
                                            label='recycle_pump')
        self.feed_water_pre = Connection(self.recycle_pump1, 'out1', self.feed_water_merge, 'in1',
                                         label='feed_water_pre')
        self.feed_water_post = Connection(self.feed_water, 'out1', self.feed_water_merge, 'in2',
                                          label='feed_water_post')

        self.nw.add_conns(self.cooling_water_in, self.cooling_water_out, self.condenser_recycle, self.recycle_pump,
                          self.feed_water_pre, self.feed_water_post)

        ########################################
        # # # extract of turbine4
        # # main steam section
        self.recycle_merge_extract_tb4 = Connection(self.feed_water_merge, 'out1', self.extract_merge_turbine4, 'in1',
                                                    label='recycle_merge_extract_tb4')
        self.exchanger2_ex_tb4_main_steam_in = Connection(self.extract_merge_turbine4, 'out1', self.extract_heat_exchanger2_turbine4, 'in2',
                                                          label='exchanger2_ex_tb4_main_steam_in')
        self.exchanger2_1_ex_tb4_main_steam = Connection(self.extract_heat_exchanger2_turbine4, 'out2', self.extract_heat_exchanger1_turbine4, 'in2',
                                                         label='exchanger2_1_ex_tb4_main_steam')
        self.exchanger1_ex_tb4_main_steam_out = Connection(self.extract_heat_exchanger1_turbine4, 'out2', self.extract_merge_turbine3, 'in1',
                                                           label='exchanger1_ex_tb4_main_steam_out')
        # # extract steam section
        self.exchanger1_ex_tb4_extract_steam_in = Connection(self.extract_splitter_turbine4, 'out2', self.extract_heat_exchanger1_turbine4, 'in1',
                                                             label='exchanger1_ex_tb4_extract_steam_in')
        self.exchanger1_2_ex_tb4_extract_steam = Connection(self.extract_heat_exchanger1_turbine4, 'out1', self.extract_heat_exchanger2_turbine4, 'in1',
                                                            label='exchanger1_2_ex_tb4_extract_steam')
        self.exchanger2_ex_tb4_extract_steam_out = Connection(self.extract_heat_exchanger2_turbine4, 'out1', self.extract_pump_turbine4, 'in1',
                                                              label='exchanger2_ex_tb4_extract_steam_out')
        self.extract_main_steam_tb4 = Connection(self.extract_pump_turbine4, 'out1', self.extract_merge_turbine4, 'in2',
                                                 label='extract_main_steam_tb4')

        self.nw.add_conns(self.recycle_merge_extract_tb4, self.exchanger2_ex_tb4_main_steam_in, self.exchanger2_1_ex_tb4_main_steam, self.exchanger1_ex_tb4_main_steam_out,
                          self.exchanger1_ex_tb4_extract_steam_in, self.exchanger1_2_ex_tb4_extract_steam, self.exchanger2_ex_tb4_extract_steam_out, self.extract_main_steam_tb4)

        # # # extract of turbine3
        # # main steam section
        self.exchanger2_ex_tb3_main_steam_in = Connection(self.extract_merge_turbine3, 'out1', self.extract_heat_exchanger2_turbine3, 'in2',
                                                          label='exchanger2_ex_tb3_main_steam_in')
        self.exchanger2_1_ex_tb3_main_steam = Connection(self.extract_heat_exchanger2_turbine3, 'out2', self.extract_heat_exchanger1_turbine3, 'in2',
                                                         label='exchanger2_1_ex_tb3_main_steam')
        self.exchanger1_ex_tb3_main_steam_out = Connection(self.extract_heat_exchanger1_turbine3, 'out2', self.deaerator, 'in1',
                                                           label='exchanger1_ex_tb3_main_steam_out')
        # # extract steam section
        self.exchanger1_ex_tb3_extract_steam_in = Connection(self.extract_splitter_turbine3, 'out2', self.extract_heat_exchanger1_turbine3, 'in1',
                                                             label='exchanger1_ex_tb3_extract_steam_in')
        self.exchanger1_2_ex_tb3_extract_steam = Connection(self.extract_heat_exchanger1_turbine3, 'out1', self.extract_heat_exchanger2_turbine3, 'in1',
                                                            label='exchanger1_2_ex_tb3_extract_steam')
        self.exchanger2_ex_tb3_extract_steam_out = Connection(self.extract_heat_exchanger2_turbine3, 'out1', self.extract_pump_turbine3, 'in1',
                                                              label='exchanger2_ex_tb3_extract_steam_out')
        self.extract_main_steam_tb3 = Connection(self.extract_pump_turbine3, 'out1', self.extract_merge_turbine3, 'in2',label='extract_main_steam_tb3')

        self.nw.add_conns(self.exchanger2_ex_tb3_main_steam_in, self.exchanger2_1_ex_tb3_main_steam, self.exchanger1_ex_tb3_main_steam_out,
                          self.exchanger1_ex_tb3_extract_steam_in, self.exchanger1_2_ex_tb3_extract_steam, self.exchanger2_ex_tb3_extract_steam_out, self.extract_main_steam_tb3)

        # # # extract of turbine2
        self.extract_valve = Connection(self.extract_splitter_turbine2, 'out2', self.extract_valve_turbine2, 'in1',
                                        label='extract_valve')
        self.extract_deaerator = Connection(self.extract_valve_turbine2, 'out1', self.deaerator, 'in2',
                                            label='extract_deaerator')
        self.extract_deaerator_splitter = Connection(self.deaerator, 'out1', self.main_steam_splitter, 'in1',
                                                     label='extract_deaerator_splitter')
        self.drain_liquid_deaerator = Connection(self.deaerator, 'out2', self.sa_liquid, 'in1',
                                                 label='drain_liquid_deaerator')

        # # main steam1
        self.main_steam1_splitter = Connection(self.main_steam_splitter, 'out1', self.recycle_pump21, 'in1',
                                               label='main_steam1_splitter')
        self.main_steam2_splitter = Connection(self.main_steam_splitter, 'out2', self.recycle_pump22, 'in1',
                                               label='main_steam2_splitter')

        self.nw.add_conns(self.extract_valve, self.extract_deaerator, self.extract_deaerator_splitter,
                          self.drain_liquid_deaerator, self.main_steam1_splitter, self.main_steam2_splitter)

    def set_properties(self):
        # components
        # mechanical
        self.turbine1.set_attr(eta_s=0.88)
        self.turbine2.set_attr(eta_s=0.88)
        self.turbine3.set_attr(eta_s=0.88)
        self.turbine4.set_attr(eta_s=0.88)
        self.turbine5.set_attr(eta_s=0.9)
        self.gas_turbine.set_attr(eta_s=0.858)  # 0.865
        self.air_compressor.set_attr(eta_s=0.845)  # 0.85
        self.fuel_compressor.set_attr(eta_s=0.85)  # 0.85
        self.evaporate_pump1.set_attr(eta_s=0.8)
        self.evaporate_pump2.set_attr(eta_s=0.8)
        self.recycle_pump1.set_attr(eta_s=0.8)
        self.extract_pump_turbine3.set_attr(eta_s=0.8)
        self.extract_pump_turbine4.set_attr(eta_s=0.8)
        self.recycle_pump21.set_attr(eta_s=0.8)
        self.recycle_pump22.set_attr(eta_s=0.8)
        # thermal
        self.main_heat_exchanger1.set_attr(pr1=1, dp2=0.2, DTU=5)
        self.evaporator1.set_attr(pr1=1, pr2=1 , DTM=5)  #
        self.main_heat_exchanger2.set_attr(pr1=1, dp2=0.2, DTU=30)
        self.evaporator2.set_attr(pr1=1, pr2=1 , DTM=5)  #
        self.main_heat_exchanger3.set_attr(pr1=1, dp2=0.2, DTU=7)
        self.condenser.set_attr(pr1=1, dp2=0.05, DTU=5)
        self.extract_heat_exchanger1_turbine3.set_attr(pr1=1, dp2=0.2, DTU_sh=20)
        self.extract_heat_exchanger2_turbine3.set_attr(pr1=1, dp2=0.05, DTL=7)
        self.extract_heat_exchanger1_turbine4.set_attr(pr1=1, dp2=0.2, DTU_sh=20)
        self.extract_heat_exchanger2_turbine4.set_attr(pr1=1, dp2=0.05, DTL=7)
        self.fuel_combustion.set_attr(eta=0.999, dp=0.005)  # 1
        self.gas_drop_pressure_valve.set_attr(dp=0.025)
        self.gas_re_combustion.set_attr(eta=0.999, pr=1, T_out=600)
        self.evaporate_tank1.set_attr(Ki=10)
        self.evaporate_tank2.set_attr(Ki=10)
        self.deaerator.set_attr(dp1=0, dp2=0)
        # connections
        # steam
        self.main_steam1_deliver.set_attr(p=48)
        self.main_steam_turbine1.set_attr(p=47.5)
        self.evaporator1_steam_out.set_attr(h0=2800)  #
        self.drain_liquid_tank1.set_attr(m=0.225)
        self.evaporator2_steam_out.set_attr(p=9, h0=2773)  #
        self.drain_liquid_tank2.set_attr(m=0.001)
        self.turbine2_steam_out.set_attr(p=4)
        self.turbine3_steam_out.set_attr(p=1.5)
        self.turbine4_steam_out.set_attr(p=0.5)
        self.turbine5_steam_out.set_attr(p=0.04)
        self.cooling_water_in.set_attr(T=15, p=3, fluid={"Water": 1})
        self.condenser_recycle.set_attr(h0=121)
        self.recycle_merge_extract_tb4.set_attr(p=2)  # ,
        self.feed_water_post.set_attr(m=0.282, T=30)
        self.extract_deaerator_splitter.set_attr(fluid={"water": 1}, h0=467)
        self.drain_liquid_deaerator.set_attr(m=0.056)
        # gas
        self.fuel_source_splitter.set_attr(T=15, p=4.5, fluid={"CH4": 1})  # "CO2": 0.04, "CH4": 0.92, "H2": 0.04
        self.fuel1_compressor_combustion.set_attr(m=1.401, p=15)
        self.fuel1_air_source_compressor.set_attr(T=15, p=0.978, m=79.86, fluid={"Ar": 0.0129, "N2": 0.7553, "CO2": 0.0004, "O2": 0.2314})
        self.fuel1_air_compressor_combustion.set_attr(p=14)  # 15
        self.gas_turbine_out.set_attr(p=1)  # 1.003
        self.fuel2_valve_combustion.set_attr(p=0.978)  # 0.98

    def set_off_design_properties(self):
        # heat exchanger
        kA_charline = CharLine(x=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2], y=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1, 1.04, 1.07])
        self.condenser.set_attr(design=['kA', 'DTU'], offdesign=['KDTA'], KDTA_fit='charline', KDTA_char2=kA_charline)
        self.evaporator1.set_attr(design=['kA', 'DTM'], offdesign=['KDTA'], KDTA_fit='default')
        self.evaporator2.set_attr(design=['kA', 'DTM'], offdesign=['KDTA'], KDTA_fit='default')
        self.extract_heat_exchanger1_turbine3.set_attr(design=['kA', 'DTU_sh'], offdesign=['KDTA'],
                                                       KDTA_fit='charline', dp2_fit='default',
                                                       KDTA_char1=kA_charline, KDTA_char2=kA_charline)
        self.extract_heat_exchanger1_turbine4.set_attr(design=['kA', 'DTU_sh'], offdesign=['KDTA'],
                                                       KDTA_fit='charline', dp2_fit='default',
                                                       KDTA_char1=kA_charline, KDTA_char2=kA_charline)
        self.extract_heat_exchanger2_turbine3.set_attr(design=['kA', 'DTL'], offdesign=['KDTA'],
                                                       KDTA_fit='charline', dp2_fit='default',
                                                       KDTA_char1=kA_charline, KDTA_char2=kA_charline)
        self.extract_heat_exchanger2_turbine4.set_attr(design=['kA', 'DTL'], offdesign=['KDTA'],
                                                       KDTA_fit='charline', dp2_fit='default',
                                                       KDTA_char1=kA_charline, KDTA_char2=kA_charline)
        self.main_heat_exchanger1.set_attr(design=['kA', 'DTU'], offdesign=['KDTA'], KDTA_fit='default')
        self.main_heat_exchanger2.set_attr(design=['kA', 'DTU'], offdesign=['KDTA'], KDTA_fit='default')
        self.main_heat_exchanger3.set_attr(design=['kA', 'DTU'], offdesign=['KDTA'], KDTA_fit='default')
        # turbine
        eta_s_charline_tur = CharLine(x=[0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1, 1.05, 1.1], y=[0.98, 0.991, 0.993, 0.995, 0.9975, 0.999, 0.9998, 1, 0.9995, 0.998])
        self.turbine1.set_attr(eta_s_char=eta_s_charline_tur, eta_s_fit='charline')
        self.turbine2.set_attr(eta_s_char=eta_s_charline_tur, eta_s_fit='charline')
        self.turbine3.set_attr(eta_s_char=eta_s_charline_tur, eta_s_fit='charline')
        self.turbine4.set_attr(eta_s_char=eta_s_charline_tur, eta_s_fit='charline')
        self.turbine5.set_attr(eta_s_char=eta_s_charline_tur, eta_s_fit='charline')
        eta_s_charline_gas_tur = CharLine(x=[0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1, 1.05, 1.1], y=[0.98, 0.991, 0.993, 0.995, 0.9975, 1.05, 1.03, 1, 0.9995, 0.998])
        self.gas_turbine.set_attr(eta_s_char=eta_s_charline_tur, eta_s_fit='charline')
        # pump
        eta_s_charline_pu = CharLine(x=[0, 0.6, 0.7, 0.8, 1, 1.1], y=[0.92, 0.94, 0.96, 0.98, 1, 0.98])
        self.evaporate_pump1.set_attr(eta_s_char=eta_s_charline_pu, eta_s_fit='charline')
        self.evaporate_pump2.set_attr(eta_s_char=eta_s_charline_pu, eta_s_fit='charline')
        self.recycle_pump1.set_attr(eta_s_char=eta_s_charline_pu, eta_s_fit='charline')
        self.extract_pump_turbine3.set_attr(eta_s_char=eta_s_charline_pu, eta_s_fit='charline')
        self.extract_pump_turbine4.set_attr(eta_s_char=eta_s_charline_pu, eta_s_fit='charline')
        self.recycle_pump21.set_attr(eta_s_char=eta_s_charline_pu, eta_s_fit='charline')
        self.recycle_pump22.set_attr(eta_s_char=eta_s_charline_pu, eta_s_fit='charline')
        # compressor
        eta_s_charline_cos = CharLine(x=[0, 0.4, 1, 1.2], y=[0.5, 0.9, 1, 1.1])
        self.fuel_compressor.set_attr(eta_s_char=eta_s_charline_cos, eta_s_fit='charline')
        self.air_compressor.set_attr(eta_s_char=eta_s_charline_cos, eta_s_fit='charline')
        pr_charline_cos = CharLine(x=[0, 0.4, 0.6, 1, 1.2], y=[0.961, 0.974, 0.985, 1, 0.998])
        self.fuel1_air_compressor_combustion.set_attr(design=['p'])  # outlet of air compressor
        self.air_compressor.set_attr(offdesign=['pr'], pr_char=pr_charline_cos, pr_fit='charline')  # pr fit
        # self.set_off_design_80_()  #
        # self.set_off_design_60_()
        # self.set_off_design_sat_()
        self.set_off_design_extract_()

    def set_off_design_80_(self):
        # 80% power of gas turbine
        self.fuel1_compressor_combustion.set_attr(m=1.163)  # fuel mass flow 1.163
        self.fuel1_air_source_compressor.set_attr(m=71.76)  # air mass flow 71.76
        self.turbine2_steam_out.set_attr(p=3.63)
        self.turbine3_steam_out.set_attr(p=1.36)
        self.turbine4_steam_out.set_attr(p=0.455)
        self.turbine5_steam_out.set_attr(p=0.0373)  # 0.1
        self.drain_liquid_tank1.set_attr(m=0.21)  # drain of tank1
        self.drain_liquid_tank2.set_attr(m=0.001)  # drain of tank2
        self.drain_liquid_deaerator.set_attr(m=0.056)  # drain of deaerator
        self.feed_water_post.set_attr(m=0.267)  # feed water
        self.gas_re_combustion.set_attr(T_out=609)

    def set_off_design_60_(self):
        self.fuel1_compressor_combustion.set_attr(m=0.937)  # fuel mass flow 0.937
        self.fuel1_air_source_compressor.set_attr(m=64.1)  # air mass flow 64.1
        self.turbine2_steam_out.set_attr(p=3.23)
        self.turbine3_steam_out.set_attr(p=1.21)
        self.turbine4_steam_out.set_attr(p=0.407)
        self.turbine5_steam_out.set_attr(p=0.0346)  #
        self.drain_liquid_tank1.set_attr(m=0.19)  # drain of tank1
        self.drain_liquid_tank2.set_attr(m=0.001)  # drain of tank2
        self.drain_liquid_deaerator.set_attr(m=0.056)  # drain of deaerator
        self.feed_water_post.set_attr(m=0.247)  # feed water
        self.gas_re_combustion.set_attr(T_out=600)

    def set_off_design_40_(self):
        self.fuel1_compressor_combustion.set_attr(m=0.937)  # fuel mass flow
        self.fuel1_air_source_compressor.set_attr(m=64.1)  # air mass flow 70.76
        self.turbine2_steam_out.set_attr(p=3.23)
        self.turbine3_steam_out.set_attr(p=1.21)
        self.turbine4_steam_out.set_attr(p=0.407)
        self.turbine5_steam_out.set_attr(p=0.0346)  #
        self.drain_liquid_tank1.set_attr(m=0.19)  # drain of tank1
        self.drain_liquid_tank2.set_attr(m=0.001)  # drain of tank2
        self.drain_liquid_deaerator.set_attr(m=0.056)  # drain of deaerator
        self.feed_water_post.set_attr(m=0.247)  # feed water

    def set_off_design_sat_(self):
        self.turbine5_steam_out.set_attr(p=0.12)  #

    def set_off_design_extract_(self):
        self.turbine5_steam_out.set_attr(p=0.12)  #
        self.turbine3_steam_out.set_attr(p=0.7)

    def solve(self, mode='design', max_iter=50, algo_factor=0.1):
        self.set_properties()
        if mode == 'offdesign':
            self.set_off_design_properties()
        self.nw.solve(mode=mode, max_iter=max_iter, algo_factor=algo_factor,
                      plot_iteration=False, print_results=True,
                      design_path= f"{self.name}_design_",
                      init_path=f"{self.name}_design_"
                      )  # , init_path=f"{self.name}_design_"
        # self.nw.save(f"{self.name}_design_")
        # self.nw.save_csv(f"{self.name}_design_csv_")
        self.nw.save_csv(f"{self.name}_offdesign_csv_")


if __name__ == '__main__':
    combine_plant_model1 = GasSteamCombineCyclePlant1('combine_plant_model1')
    combine_plant_model1.solve(max_iter=150, mode='offdesign', algo_factor=0.01)
