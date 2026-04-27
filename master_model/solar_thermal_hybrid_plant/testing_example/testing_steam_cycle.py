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
from Aurora.connections import Connection, Ref
from Aurora.networks import Network
from Aurora.connections import Bus

from Aurora.tools.characteristics import load_default_char
from Aurora.tools.characteristics import CharLine
from Aurora.tools import logger
import logging

logger.define_logging(
            logpath=f"steam_testing_loggings", log_the_path=True, log_the_version=True,
            screen_level=logging.INFO, file_level=logging.DEBUG)

nw = Network(p_unit="bar", T_unit='C', h_unit="kJ / kg", m_unit='kg / s', iterinfo=True)
# components
oil_source_21 = Source('oil_source_21')
oil_sink_81 = Sink('oil_sink_81')
# heat exchanger
heatexchanger_d = HeatExchanger('heatexchanger_d', nodes_num=40)
heatexchanger_e = HeatExchanger('heatexchanger_e', nodes_num=40)
heatexchanger_f = HeatExchanger('heatexchanger_f', nodes_num=40)
heatexchanger_g = ExtractHeatExchanger('heatexchanger_g', nodes_num=40)
heatexchanger_h = ExtractHeatExchanger('heatexchanger_h', nodes_num=40)
heatexchanger_i = ExtractHeatExchanger('heatexchanger_i', nodes_num=40)
heatexchanger_j = ExtractHeatExchanger('heatexchanger_j', nodes_num=40)
heatexchanger_k = ExtractHeatExchanger('heatexchanger_k', nodes_num=40)
# evaporator
evaporator = Evaporator('evaporator', nodes_num=40)
# condenser
condenser = Condenser('condenser', nodes_num=40)
# evaporator drum
evaporator_drum = EvaporateTank('evaporator_drum')
# deaerator
deaerator = Deaerator('deaerator')
# turbine
hp_turbine1 = Turbine('hp_turbine1')
hp_turbine2 = Turbine('hp_turbine2')
lp_turbine1 = Turbine('lp_turbine1')
lp_turbine2 = Turbine('lp_turbine2')
lp_turbine3 = Turbine('lp_turbine3')
lp_turbine4 = Turbine('lp_turbine4')
lp_turbine5 = Turbine('lp_turbine5')
gas_turbine = Turbine('gas_turbine')
# pump
condense_pump = Pump('condense_pump')
steam_recycle_pump = Pump('steam_recycle_pump')  # used for evaporate module
water_recycle_pump = Pump('water_recycle_pump')  # used for deaerator
# mass flow amplifier
af_boiler_hot_in = FlowAmplifier('af_boiler_hot_in')
af_boiler_hot_out = FlowAmplifier('af_boiler_hot_out')
af_boiler_cold_in = FlowAmplifier('af_boiler_cold_in')
af_boiler_cold_out = FlowAmplifier('af_boiler_cold_out')
af_reheater_hot_in = FlowAmplifier('af_reheater_hot_in')
af_reheater_hot_out = FlowAmplifier('af_reheater_hot_out')
af_reheater_cold_in = FlowAmplifier('af_reheater_cold_in')
af_reheater_cold_out = FlowAmplifier('af_reheater_cold_out')
# valve
valve_50_51 = Valve('valve_50_51')
valve_52_53 = Valve('valve_52_53')
valve_76_77 = Valve('valve_76_77')
valve_73_74 = Valve('valve_73_74')
valve_69_71 = Valve('valve_69_71')
valve_7 = Valve('valve_7')
valve_9 = Valve('valve_9')
valve_36 = Valve('valve_36')
valve_80 = Valve('valve_80')
# cycle closer
steam_cycle_closer = CycleCloser('steam_cycle_closer')
# distributor
# merge
merge_7_9_10 = Merge('merge_7_9_10', num_in=2)
merge_12_13_21 = Merge('merge_12_13_21', num_in=2)
merge_36_80_81 = Merge('merge_36_80_81', num_in=2)
merge_18_82_83 = Merge('merge_18_82_83', num_in=2)
merge_67_68_69 = Merge('merge_67_68_69', num_in=2)
merge_heater_k = Merge('merge_heater_k', num_in=2)
merge_heater_j = Merge('merge_heater_j', num_in=2)
merge_deaerator = Merge('merge_deaerator', num_in=2)
merge_heater_h = Merge('merge_heater_h', num_in=2)
# splitter
splitter_10_11_12 = Splitter('splitter_10_11_12', num_out=2)
splitter_21_22_23 = Splitter('splitter_21_22_23', num_out=2)
splitter_20_81_82 = Splitter('splitter_20_81_82', num_out=2)
splitter_6_84_85 = Splitter('splitter_6_84_85', num_out=2)
splitter_h1 = Splitter('splitter_h1', num_out=2)
splitter_42_47_48 = Splitter('splitter_42_47_48', num_out=2)
splitter_l1 = Splitter('splitter_l1', num_out=2)
splitter_l2 = Splitter('splitter_l2', num_out=2)
splitter_l3 = Splitter('splitter_l3', num_out=2)
splitter_l4 = Splitter('splitter_l4', num_out=2)
# source
cooling_water_source = Source('cooling_water_source')
# sink
evaporator_drain = Sink('evaporator_drain')
deaerator_drain = Sink('deaerator_drain')
cooling_water_sink = Sink('cooling_water_sink')

## connections
c21 = Connection(oil_source_21, 'out1', splitter_21_22_23, 'in1', label='c21')
c22 = Connection(splitter_21_22_23, 'out2', af_reheater_hot_in, 'in1', label='c22')
c23 = Connection(splitter_21_22_23, 'out1', af_boiler_hot_in, 'in1', label='c23')
c24 = Connection(af_boiler_hot_in, 'out1', heatexchanger_d, 'in1', label='c24')
c25 = Connection(heatexchanger_d, 'out2', af_boiler_cold_out, 'in1', label='c25')
c26 = Connection(heatexchanger_d, 'out1', evaporator, 'in1', label='c26')
c27 = Connection(evaporator_drum, 'out2', heatexchanger_d, 'in2', label='c27')
c28 = Connection(evaporator, 'out2', evaporator_drum, 'in2', label='c28')
c29 = Connection(steam_recycle_pump, 'out1', evaporator, 'in2', label='c29')
c30 = Connection(evaporator_drum, 'out1', steam_recycle_pump, 'in1', label='c30')
c31 = Connection(evaporator_drum, 'out3', evaporator_drain, 'in1', label='c31')
c32 = Connection(heatexchanger_e, 'out2', evaporator_drum, 'in1', label='c32')
c33 = Connection(evaporator, 'out1', heatexchanger_e, 'in1', label='c33')
c34 = Connection(heatexchanger_e, 'out1', af_boiler_hot_out, 'in1', label='c34')
c35 = Connection(af_boiler_cold_in, 'out1', heatexchanger_e, 'in2', label='c35')
c36 = Connection(af_boiler_hot_out, 'out1', valve_36, 'in1', label='c36')
d36 = Connection(valve_36, 'out1',  merge_36_80_81, 'in1', label='d36')
c37 = Connection(af_reheater_hot_in, 'out1', heatexchanger_f, 'in1', label='c37')
c38 = Connection(af_reheater_cold_in, 'out1', heatexchanger_f, 'in2', label='c38')
c39 = Connection(heatexchanger_f, 'out1', af_reheater_hot_out, 'in1', label='c39')
c40 = Connection(heatexchanger_f, 'out2', af_reheater_cold_out, 'in1', label='c40')
c41 = Connection(af_reheater_cold_out, 'out1', lp_turbine1, 'in1', label='c41')
c42 = Connection(splitter_42_47_48, 'out1', af_reheater_cold_in, 'in1', label='c42')
c43 = Connection(heatexchanger_g, 'out2', af_boiler_cold_in, 'in1', label='c43')
c44 = Connection(af_boiler_cold_out, 'out1', hp_turbine1, 'in1', label='c44')
h1_out = Connection(hp_turbine1, 'out1', splitter_h1, 'in1', label='h1_out')
c45 = Connection(splitter_h1, 'out1', hp_turbine2, 'in1', label='c45')
c46 = Connection(splitter_h1, 'out2', heatexchanger_g, 'in1', label='c46')
c47 = Connection(hp_turbine2, 'out1', splitter_42_47_48, 'in1', label='c47')
c48 = Connection(splitter_42_47_48, 'out2', merge_heater_h, 'in2', label='c48')
c49 = Connection(heatexchanger_h, 'out2', heatexchanger_g, 'in2', label='c49')
c50 = Connection(heatexchanger_g, 'out1', valve_50_51, 'in1', label='c50')
c51 = Connection(valve_50_51, 'out1', merge_heater_h, 'in1', label='c51')
heater_h_hot_in = Connection(merge_heater_h, 'out1', heatexchanger_h, 'in1', label='heater_h_hot_in')
c52 = Connection(heatexchanger_h, 'out1', valve_52_53, 'in1', label='c52')
c53 = Connection(valve_52_53, 'out1', merge_deaerator, 'in2', label='c53')
c54 = Connection(deaerator, 'out1', steam_cycle_closer, 'in1', label='c54')
d54 = Connection(steam_cycle_closer, 'out1', water_recycle_pump, 'in1', label='d54')
c55 = Connection(water_recycle_pump, 'out1', heatexchanger_h, 'in2', label='c55')
l1_out = Connection(lp_turbine1, 'out1', splitter_l1, 'in1', label='l1_out')
c56 = Connection(splitter_l1, 'out2', deaerator, 'in2', label='c56')
deaerator_cold_in = Connection(merge_deaerator, 'out1', deaerator, 'in1', label='deaerator_cold_in')
c57 = Connection(splitter_l1, 'out1', lp_turbine2, 'in1', label='c57')
l2_out = Connection(lp_turbine2, 'out1', splitter_l2, 'in1', label='l2_out')
c58 = Connection(splitter_l2, 'out2', heatexchanger_i, 'in1', label='c58')
c59 = Connection(splitter_l2, 'out1', lp_turbine3, 'in1', label='c59')
l3_out = Connection(lp_turbine3, 'out1', splitter_l3, 'in1', label='l3_out')
c60 = Connection(splitter_l3, 'out2', merge_heater_j, 'in1', label='c60')
c61 = Connection(splitter_l3, 'out1', lp_turbine4, 'in1', label='c61')
l4_out = Connection(lp_turbine4, 'out1', splitter_l4, 'in1', label='l4_out')
c62 = Connection(splitter_l4, 'out2', merge_heater_k, 'in1', label='c62')
c63 = Connection(splitter_l4, 'out1', lp_turbine5, 'in1', label='c63')
c64 = Connection(lp_turbine5, 'out1', condenser, 'in1', label='c64')
c65 = Connection(condenser, 'out2', cooling_water_sink, 'in1', label='c65')
c66 = Connection(cooling_water_source, 'out1', condenser, 'in2', label='c66')
c67 = Connection(condenser, 'out1', merge_67_68_69, 'in1', label='c67')
c68 = Connection(merge_67_68_69, 'out1', condense_pump, 'in1', label='c68')
c69 = Connection(valve_69_71, 'out1', merge_67_68_69, 'in2', label='c69')
c70 = Connection(condense_pump, 'out1', heatexchanger_k, 'in2', label='c70')
c71 = Connection(heatexchanger_k, 'out1', valve_69_71, 'in1', label='c71')
c72 = Connection(heatexchanger_k, 'out2', heatexchanger_j, 'in2', label='c72')
c73 = Connection(valve_73_74, 'out1', merge_heater_k, 'in2', label='c73')
heater_k_hot_in = Connection(merge_heater_k, 'out1', heatexchanger_k, 'in1', label='heater_k_hot_in')
c74 = Connection(heatexchanger_j, 'out1', valve_73_74, 'in1', label='c74')
c75 = Connection(heatexchanger_j, 'out2', heatexchanger_i, 'in2', label='c75')
c76 = Connection(valve_76_77, 'out1', merge_heater_j, 'in2', label='c76')
heater_j_hot_in = Connection(merge_heater_j, 'out1', heatexchanger_j, 'in1', label='heater_j_hot_in')
c77 = Connection(heatexchanger_i, 'out1', valve_76_77, 'in1', label='c77')
c78 = Connection(heatexchanger_i, 'out2', merge_deaerator, 'in1', label='c78')
c79 = Connection(deaerator, 'out2', deaerator_drain, 'in1', label='c79')
c80 = Connection(af_reheater_hot_out, 'out1', valve_80, 'in1', label='c80')
d80 = Connection(valve_80, 'out1', merge_36_80_81, 'in2', label='d80')
c81 = Connection(merge_36_80_81, 'out1', oil_sink_81, 'in1', label='c81')

nw.add_conns(c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39, c40, c41,
             c42, c43, c44, c45, c46, c47, c48, c49, c50, c51, c52, c53, c54, c55, c56, c57, c58, c59, c60, c61, c62, c63,
             c64, c65, c66, c67, c68, c69, c70, c71, c72, c73, c74, c75, c76, c77, c78, c79, c80, c81, d36, d54, d80,
             h1_out, heater_h_hot_in, l1_out, l2_out, l3_out, l4_out, heater_j_hot_in, heater_k_hot_in, deaerator_cold_in)
# set properties
# components
heatexchanger_d.set_attr(dp1=1.6, dp2=0.5, DTU=22)
heatexchanger_e.set_attr(dp1=1.6, dp2=0.5, DTU=10.5)  # !!!
heatexchanger_f.set_attr(dp1=1.6, dp2=0.5, DTU=23)
heatexchanger_g.set_attr(dp1=0, dp2=0.05, DTU_sh=5)
heatexchanger_h.set_attr(dp1=0, dp2=0.05, DTU_sh=5)
heatexchanger_i.set_attr(dp1=0, dp2=0.05, DTU_sh=5)
heatexchanger_j.set_attr(dp1=0, dp2=0.05, DTU_sh=5)
heatexchanger_k.set_attr(dp1=0, dp2=0.05, DTU_sh=5)
condenser.set_attr(dp1=0, dp2=0.05, DTU_sh=5)
evaporator.set_attr(dp1=0, dp2=0.002, DTM=7.5)
# vapour tank
evaporator_drum.set_attr(Ki=10)
# deaerator
deaerator.set_attr(dp1=0, dp2=0)
# mass amplifier
af_boiler_hot_in.set_attr(Ki=6)
af_boiler_hot_out.set_attr(Ki=1/6)
af_boiler_cold_in.set_attr(Ki=2)
af_boiler_cold_out.set_attr(Ki=1/2)
af_reheater_hot_in.set_attr(Ki=6)
af_reheater_hot_out.set_attr(Ki=1/6)
af_reheater_cold_in.set_attr(Ki=2)
af_reheater_cold_out.set_attr(Ki=1/2)
# pump
steam_recycle_pump.set_attr(eta_s=0.8)
water_recycle_pump.set_attr(eta_s=0.8)
condense_pump.set_attr(eta_s=0.8)
# turbine
hp_turbine1.set_attr(eta_s=0.88)
hp_turbine2.set_attr(eta_s=0.88)
lp_turbine1.set_attr(eta_s=0.88)
lp_turbine2.set_attr(eta_s=0.88)
lp_turbine3.set_attr(eta_s=0.88)
lp_turbine4.set_attr(eta_s=0.88)
lp_turbine5.set_attr(eta_s=0.88)

# connections
# cooling water
c66.set_attr(p=2, T=30, fluid={'water': 1})
# steam module
c44.set_attr(p=100, fluid={'water': 1})
c45.set_attr(p=40)
c41.set_attr(p=16.5)
c57.set_attr(p=6)
c59.set_attr(p=2.5)
c61.set_attr(p=1.2)
c63.set_attr(p=0.6)
c64.set_attr(p=0.08)
c79.set_attr(m=0)  # deaerator drain
c31.set_attr(m=0)  # evaporate tank drain
# oil cycle
c21.set_attr(p=21, m=550, T=393, fluid={'DowthermA': 1})
c22.set_attr(m=Ref(c21, 0.11, 0))
# c22.set_attr(m=59.2 * 3)
# c23.set_attr(m=479.1 * 3)
c81.set_attr(p=15)  # , T=295
#

nw.solve(mode='design', max_iter=100, algo_factor=0.01,
        plot_iteration=False, print_results=True,
        init_path="steam_testing_design_",
        # design_path="steam_testing_design_"
        )
# nw.save(f"steam_testing_design_2")