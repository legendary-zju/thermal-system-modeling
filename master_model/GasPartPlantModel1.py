from Aurora.components import CycleCloser, HeatExchanger, Evaporator
from Aurora.components import Sink
from Aurora.components import Source
from Aurora.components import Condenser
from Aurora.components import Deaerator, DeaeratorSimple
from Aurora.components import Desuperheater
from Aurora.components import SimpleHeatExchanger
from Aurora.components import DiabaticCombustionChamber
from Aurora.components import Merge
from Aurora.components import Splitter
from Aurora.components import Separator, DropletSeparator, Drum
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



class GasPartPlant1:
    def __init__(self, name):
        self.name = name

        logger.define_logging(
            logpath=f"{self.name}_loggings", log_the_path=True, log_the_version=True,
            screen_level=logging.INFO, file_level=logging.DEBUG)

        self.nw = Network(p_unit="bar", T_unit='C', h_unit="kJ / kg", m_unit='kg / s', iterinfo=True)

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
        self.gas_re_combustion = DiabaticCombustionChamber('gas_re_combustion')
        self.gas_sink = Sink('gas_sink')

        # # # # connection
        # # # gas cycle section
        self.fuel_source_splitter = Connection(self.fuel_source, 'out1', self.fuel_splitter, 'in1',
                                               label='fuel_source_splitter')

        # # fuel section1
        self.fuel1_splitter_compressor = Connection(self.fuel_splitter, 'out1', self.fuel_compressor, 'in1',
                                                    label='fuel1_splitter_compressor')
        self.fuel1_compressor_combustion = Connection(self.fuel_compressor, 'out1', self.fuel_combustion, 'in2',
                                                      label='fuel1_compressor_combustion')
        # fuel1 air section
        # air filter
        self.fuel1_air_source_compressor = Connection(self.air_source, 'out1', self.air_compressor, 'in1',
                                                      label='fuel1_air_source_compressor')
        self.fuel1_air_compressor_combustion = Connection(self.air_compressor, 'out1', self.fuel_combustion, 'in1',
                                                          label='fuel1_air_compressor_combustion')
        # gas turbine
        self.gas_turbine_in = Connection(self.fuel_combustion, 'out1', self.gas_turbine, 'in1', label='gas_turbine_in')

        # # fuel section2
        self.gas_turbine_out = Connection(self.gas_turbine, 'out1', self.gas_re_combustion, 'in1',
                                          label='gas_turbine_out')
        self.fuel2_splitter_valve = Connection(self.fuel_splitter, 'out2', self.fuel_add_valve, 'in1',
                                               label='fuel2_splitter_valve')
        self.fuel2_valve_combustion = Connection(self.fuel_add_valve, 'out1', self.gas_re_combustion, 'in2',
                                                 label='fuel2_valve_combustion')
        self.gas_out = Connection(self.gas_re_combustion, 'out1', self.gas_sink, 'in1', label='gas_out')

        self.nw.add_conns(self.fuel_source_splitter, self.fuel1_splitter_compressor, self.fuel1_compressor_combustion,
                          self.fuel1_air_source_compressor, self.fuel1_air_compressor_combustion, self.gas_turbine_in,
                          self.gas_turbine_out, self.fuel2_splitter_valve, self.fuel2_valve_combustion, self.gas_out)

    def set_properties(self):
        # component
        self.gas_turbine.set_attr(eta_s=0.88)
        self.air_compressor.set_attr(eta_s=0.85)
        self.fuel_compressor.set_attr(eta_s=0.85)
        self.fuel_combustion.set_attr(eta=0.99, pr=1)
        self.gas_re_combustion.set_attr(eta=0.99, pr=1)
        # connection
        # gas
        self.fuel_source_splitter.set_attr(T=15, p=4.5, fluid={"CO2": 0.05, "CH4": 0.92, "H2": 0.03})  # "CO2": 0.04, "CH4": 0.92, "H2": 0.04
        self.fuel1_compressor_combustion.set_attr(m=1.401, p=15)
        self.fuel1_air_source_compressor.set_attr(T=15, p=0.978, m=79.86, fluid={"Ar": 0.0129, "N2": 0.7553, "CO2": 0.0004, "O2": 0.2314})
        self.fuel1_air_compressor_combustion.set_attr(p=15)
        self.gas_turbine_out.set_attr(p=1.1)
        # self.gas_turbine_in.set_attr(T=1000)
        self.fuel2_valve_combustion.set_attr(m=0.102, p=4)

    def solve(self, mode='design', max_iter=50):
        self.set_properties()
        self.nw.solve(mode=mode, max_iter=max_iter, plot_iteration=False, print_results=True)

if __name__ == '__main__':
    gas_part_plant1 = GasPartPlant1('gas_part_plant1')
    gas_part_plant1.solve()

