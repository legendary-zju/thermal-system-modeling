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

logger.define_logging(
            logpath=f"salt_testing_loggings", log_the_path=True, log_the_version=True,
            screen_level=logging.INFO, file_level=logging.DEBUG)

# components
# heat storage tank
cold_salt_tank = HeatStorageTank('cold_salt_tank')
hot_salt_tank = HeatStorageTank('hot_salt_tank')
# cycle closer
salt_cycle_closer = CycleCloser('salt_cycle_closer')
# distributor
oil_source_left = Source('oil_source_left')
oil_sink_left = Sink('oil_sink_left')
oil_source_right = Source('oil_source_right')
oil_sink_right = Sink('oil_sink_right')
oil_source_6 = Source('oil_source_6')
oil_sink_7 = Sink('oil_sink_7')
oil_source_81 = Source('oil_source_81')
oil_sink_21 = Sink('oil_sink_21')
#
air_source = Source('air_source')
fuel_source = Source('fuel_source')
gas_sink = Sink('gas_sink')
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
# heat exchanger
heat_exchanger_a = HeatExchanger('heat_exchanger_a')
heat_exchanger_b = HeatExchanger('heat_exchanger_b')
heat_exchanger_c = HeatExchanger('heat_exchanger_c')
# solar collector
solar_collector1 = SolarCollector('solar_collector1')
solar_collector2 = SolarCollector('solar_collector2')
solar_collector3 = SolarCollector('solar_collector3')
solar_collector4 = SolarCollector('solar_collector4')
# combustion
combustion = DiabaticCombustionChamber('combustion')
# turbine
gas_turbine = Turbine('gas_turbine')
# compressor
air_compressor = Compressor('air_compressor')
# pump
oil_recycle_pump1 = Pump('oil_recycle_pump1')
oil_recycle_pump2 = Pump('oil_recycle_pump2')
# mass amplifier
af_solar_in = FlowAmplifier('af_solar_in')
af_solar_out = FlowAmplifier('af_solar_out')
# pressure adjustment
valve_7 = Valve('valve_7')
valve_9 = Valve('valve_9')
# connections
# gas cycle
c1 = Connection(air_source, 'out1', air_compressor, 'in1', label='c1')
c2 = Connection(fuel_source, 'out1', combustion, 'in2', label='c2')
c3 = Connection(air_compressor, 'out1', combustion, 'in1', label='c3')
c4 = Connection(combustion, 'out1', gas_turbine, 'in1', label='c4')
c5 = Connection(gas_turbine, 'out1', heat_exchanger_a, 'in1', label='c5')
c6 = Connection(splitter_6_84_85, 'out2', heat_exchanger_a, 'in2', label='c6')
c7 = Connection(heat_exchanger_a, 'out2', valve_7, 'in1', label='c7')
d7 = Connection(valve_7, 'out1', merge_7_9_10, 'in2', label='d7')
c8 = Connection(heat_exchanger_a, 'out1', gas_sink, 'in1', label='c8')
# solar cycle
c9 = Connection(af_solar_out, 'out1', valve_9, 'in1', label='c9')
d9 = Connection(valve_9, 'out1', merge_7_9_10, 'in1', label='d9')
c10 = Connection(merge_7_9_10, 'out1', splitter_10_11_12, 'in1', label='c10')
c12 = Connection(splitter_10_11_12, 'out1', merge_12_13_21, 'in1', label='c12')
c21 = Connection(merge_12_13_21, 'out1', oil_sink_21, 'in1', label='c21')
c81 = Connection(oil_source_81, 'out1', splitter_20_81_82, 'in1', label='c81')
c82 = Connection(splitter_20_81_82, 'out1', merge_18_82_83, 'in1', label='c82')
c83 = Connection(merge_18_82_83, 'out1', oil_recycle_pump1, 'in1', label='c83')
c84 = Connection(oil_recycle_pump1, 'out1', splitter_6_84_85, 'in1', label='c84')
c85 = Connection(splitter_6_84_85, 'out1', af_solar_in, 'in1', label='c85')
c86 = Connection(af_solar_in, 'out1', solar_collector4, 'in1', label='c86')
c87 = Connection(solar_collector4, 'out1', solar_collector3, 'in1', label='c87')
c88 = Connection(solar_collector3, 'out1', solar_collector2, 'in1', label='c88')
c89 = Connection(solar_collector2, 'out1', solar_collector1, 'in1', label='c89')
c90 = Connection(solar_collector1, 'out1', af_solar_out, 'in1', label='c90')
# salt cycle
c11 = Connection(splitter_10_11_12, 'out2', heat_exchanger_b, 'in1', label='c11')
c13 = Connection(heat_exchanger_c, 'out2', merge_12_13_21, 'in2', label='c13')
c14 = Connection(heat_exchanger_b, 'out2', hot_salt_tank, 'in1', label='c14')
c15 = Connection(hot_salt_tank, 'out1', heat_exchanger_c, 'in1', label='c15')
c16 = Connection(cold_salt_tank, 'out1', heat_exchanger_b, 'in2', label='c16')
c17 = Connection(heat_exchanger_c, 'out1', salt_cycle_closer, 'in1', label='c17')
d17 = Connection(salt_cycle_closer, 'out1', cold_salt_tank, 'in1', label='d17')
c18 = Connection(heat_exchanger_b, 'out1', merge_18_82_83, 'in2', label='c18')
c19 = Connection(oil_recycle_pump2, 'out1', heat_exchanger_c, 'in2', label='c19')
c20 = Connection(splitter_20_81_82, 'out2', oil_recycle_pump2, 'in1', label='c20')
# generate network
nw = Network(p_unit="bar", T_unit='C', h_unit="kJ / kg", m_unit='kg / s', iterinfo=True)
nw.add_conns(c1, c2, c3, c4, c5, c6, c7, d7, c8)
nw.add_conns(c9, d9, c10, c12, c21, c81, c82, c83, c84, c85, c86, c87, c88, c89, c90)
nw.add_conns(c11, c13, c14, c15, c16, c17, d17, c18, c19, c20)

# set properties
# components
heat_exchanger_a.set_attr(dp1=0.02, dp2=0.002, DTL=6)
heat_exchanger_b.set_attr(dp1=6, dp2=0, DTL=7)
heat_exchanger_c.set_attr(dp2=0, DTU=9)
hot_salt_tank.set_attr(T_out=384, T_in=387)
cold_salt_tank.set_attr(T_out=292, T_in=296)
salt_cycle_closer.set_attr(mass_conservation=True)
af_solar_in.set_attr(Ki=156)
af_solar_out.set_attr(Ki=1/156)
solar_collector1.set_attr(D=0.2, L=4, dp=0, eta_opt=0.40, fA=890, E=1000, hf=0, Tamb=25)
solar_collector2.set_attr(D=0.2, L=4, dp=0, eta_opt=0.40, fA=890, E=1000, hf=0, Tamb=25)
solar_collector3.set_attr(D=0.2, L=4, dp=0, eta_opt=0.40, fA=890, E=1000, hf=0, Tamb=25)
solar_collector4.set_attr(D=0.2, L=4, dp=0, eta_opt=0.40, fA=890, E=1000, hf=0, Tamb=25)
combustion.set_attr(dp=0, eta=0.99)
gas_turbine.set_attr(eta_s=0.88)
air_compressor.set_attr(eta_s=0.8)
oil_recycle_pump1.set_attr(eta_s=0.8)
oil_recycle_pump2.set_attr(eta_s=0.8)
# connections
# solar cycle
c9.set_attr(m=1100)
c21.set_attr(T=393)  # can't be set with c11.T together
# gas cycle
c1.set_attr(m=400, p=1, T=20, fluid={"Ar": 0.0129, "N2": 0.7553, "CO2": 0.0004, "O2": 0.2314})  # 400
c2.set_attr(m=11.5, p=1, T=20, fluid={"CH4": 1})  # 11.5
c3.set_attr(p=15)
c5.set_attr(p=0.9)
c6.set_attr(p=35, fluid={'DowthermA': 1})  # !!
# salt cycle
# c11.set_attr(T=393)  #  'DowthermA'  MD4M
c14.set_attr(p=1, m=450, fluid={"Solar Salt": 1})  # m=953
c19.set_attr(p=21)  #  'DowthermA'  MD4M
c15.set_attr(m=450)  # m=10
c81.set_attr(T=295)



nw.solve(mode='design', max_iter=100, algo_factor=0.01,
        plot_iteration=False, print_results=True,
        init_path="salt_testing_design_",
        # design_path="salt_testing_design_"
        )
# nw.save(f"salt_testing_design_3")

