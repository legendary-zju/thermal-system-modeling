from Aurora.components import CycleCloser
from Aurora.components import Sink
from Aurora.components import Source
from Aurora.components import Condenser
from Aurora.components import Deaerator, DeaeratorSimple
from Aurora.components import Desuperheater
from Aurora.components import SimpleHeatExchanger, HeatExchanger, Evaporator, ExtractHeatExchanger, BoilerSimple
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

class NuclearPlantPureSteamCyclePlant1:
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
        self.boiler = BoilerSimple('boiler')
        # heat exchanger
        self.heatexchanger_a = HeatExchanger('heatexchanger_a', nodes_num=40)
        self.heatexchanger_b = HeatExchanger('heatexchanger_b', nodes_num=40)
        self.heatexchanger_c = HeatExchanger('heatexchanger_c', nodes_num=40)
        self.heatexchanger_d = HeatExchanger('heatexchanger_d', nodes_num=40)
        self.heatexchanger_e = HeatExchanger('heatexchanger_e', nodes_num=40)
        self.heatexchanger_f = HeatExchanger('heatexchanger_f', nodes_num=40)
        self.heatexchanger_g = HeatExchanger('heatexchanger_g', nodes_num=40)
        self.heatexchanger_h = HeatExchanger('heatexchanger_h', nodes_num=40)
        self.heatexchanger_i = HeatExchanger('heatexchanger_i', nodes_num=40)
        self.heatexchanger_j = HeatExchanger('heatexchanger_j', nodes_num=40)
        self.heatexchanger_k = HeatExchanger('heatexchanger_k', nodes_num=40)
        # condenser
        self.condenser = Condenser('condenser')
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
        # 