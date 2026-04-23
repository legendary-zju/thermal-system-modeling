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

# components
salt_source = Source('salt_source')
salt_sink = Sink('salt_sink')
oil_source = Source('oil_source')
oil_sink = Sink('oil_sink')
heat_exchanger_b = HeatExchanger('heat_exchanger_b')
# connections
c11 = Connection(oil_source, 'out1', heat_exchanger_b, 'in1', label='c11')
c18 = Connection(heat_exchanger_b, 'out1', oil_sink, 'in1', label='c18')
c16 = Connection(salt_source, 'out1', heat_exchanger_b, 'in2', label='c16')
c14 = Connection(heat_exchanger_b, 'out2', salt_sink, 'in1', label='c14')

nw = Network(p_unit="bar", T_unit='C', h_unit="kJ / kg", m_unit='kg / s', iterinfo=True)
nw.add_conns(c11, c18, c16, c14)

# set properties
heat_exchanger_b.set_attr(dp1=0.01, dp2=0, DTL=7)
c11.set_attr(p=35, fluid={'DowthermA': 1})  #  'DowthermA'
c14.set_attr(m=953, p=1, fluid={"Solar Salt": 1})
c11.set_attr(m=550, T=393)
c16.set_attr(T=292)

nw.solve(mode='design', max_iter=100, algo_factor=0.01,
        plot_iteration=False, print_results=True,
        )

