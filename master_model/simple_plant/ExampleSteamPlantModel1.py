from Aurora.components import CycleCloser
from Aurora.components import Sink
from Aurora.components import Source
from Aurora.components import Condenser
from Aurora.components import Desuperheater
from Aurora.components import SimpleHeatExchanger
from Aurora.components import Merge
from Aurora.components import Splitter
from Aurora.components import Pump
from Aurora.components import Turbine
from Aurora.connections import Bus
from Aurora.connections import Connection
from Aurora.networks import Network

from Aurora.tools.characteristics import CharLine
from Aurora.tools.fluid_properties.wrappers import IAPWSWrapper

from Aurora.tools.characteristics import load_default_char
from Aurora.tools.characteristics import CharLine
from Aurora.tools import logger
import logging


class ExampleSteamPlantModel1:
    def __init__(self, name):
        self.name = name

        logger.define_logging(
            logpath=f"{self.name}_loggings", log_the_path=True, log_the_version=True,
            screen_level=logging.INFO, file_level=logging.DEBUG)

        self.nw = Network(p_unit="bar", T_unit='C', h_unit="kJ / kg", m_unit='kg / s', iterinfo=True)

        # components
        # main cycle
        self.sg = SimpleHeatExchanger("steam generator")
        self.cc = CycleCloser("cycle closer")
        self.hpt = Turbine("high pressure turbine")
        self.sp1 = Splitter("splitter 1", num_out=2)
        self.mpt = Turbine("mid pressure turbine")
        self.sp2 = Splitter("splitter 2", num_out=2)
        self.lpt = Turbine("low pressure turbine")
        self.con = Condenser("condenser")
        self.pu1 = Pump("feed water pump")
        self.fwh1 = Condenser("feed water preheater 1")
        self.fwh2 = Condenser("feed water preheater 2")
        self.dsh = Desuperheater("desuperheater")
        self.me2 = Merge("merge2", num_in=2)
        self.pu2 = Pump("feed water pump 2")
        self.pu3 = Pump("feed water pump 3")
        self.me = Merge("merge", num_in=2)

        # cooling water
        self.cwi = Source("cooling water source")
        self.cwo = Sink("cooling water sink")

        # connections
        # main cycle
        self.c0 = Connection(self.sg, "out1", self.cc, "in1", label="0")
        self.c1 = Connection(self.cc, "out1", self.hpt, "in1", label="1")
        self.c2 = Connection(self.hpt, "out1", self.sp1, "in1", label="2")
        self.c3 = Connection(self.sp1, "out1", self.mpt, "in1", label="3", state="g")
        self.c4 = Connection(self.mpt, "out1", self.sp2, "in1", label="4")
        self.c5 = Connection(self.sp2, "out1", self.lpt, "in1", label="5")
        self.c6 = Connection(self.lpt, "out1", self.con, "in1", label="6")
        self.c7 = Connection(self.con, "out1", self.pu1, "in1", label="7", state="l")
        self.c8 = Connection(self.pu1, "out1", self.fwh1, "in2", label="8", state="l")
        self.c9 = Connection(self.fwh1, "out2", self.me, "in1", label="9", state="l")
        self.c10 = Connection(self.me, "out1", self.fwh2, "in2", label="10", state="l")
        self.c11 = Connection(self.fwh2, "out2", self.dsh, "in2", label="11", state="l")
        self.c12 = Connection(self.dsh, "out2", self.me2, "in1", label="12", state="l")
        self.c13 = Connection(self.me2, "out1", self.sg, "in1", label="13", state="l")

        self.nw.add_conns(
            self.c0, self.c1, self.c2, self.c3, self.c4, self.c5, self.c6, self.c7, self.c8, self.c9, self.c10,
            self.c11, self.c12, self.c13
        )

        # preheating
        self.c21 = Connection(self.sp1, "out2", self.dsh, "in1", label="21")
        self.c22 = Connection(self.dsh, "out1", self.fwh2, "in1", label="22")
        self.c23 = Connection(self.fwh2, "out1", self.pu2, "in1", label="23")
        self.c24 = Connection(self.pu2, "out1", self.me2, "in2", label="24")

        self.c31 = Connection(self.sp2, "out2", self.fwh1, "in1", label="31")
        self.c32 = Connection(self.fwh1, "out1", self.pu3, "in1", label="32")
        self.c33 = Connection(self.pu3, "out1", self.me, "in2", label="33")

        self.nw.add_conns(self.c21, self.c22, self.c23, self.c24, self.c31, self.c32, self.c33)

        # cooling water
        self.c41 = Connection(self.cwi, "out1", self.con, "in2", label="41")
        self.c42 = Connection(self.con, "out2", self.cwo, "in1", label="42")

        self.nw.add_conns(self.c41, self.c42)

    def set_properties(self):
        self.hpt.set_attr(eta_s=0.9)
        self.mpt.set_attr(eta_s=0.9)
        self.lpt.set_attr(eta_s=0.9)
        self.pu1.set_attr(eta_s=0.8)
        self.pu2.set_attr(eta_s=0.8)
        self.pu3.set_attr(eta_s=0.8)
        self.sg.set_attr(pr=0.92)
        self.con.set_attr(pr1=1, pr2=0.99, DTU=5)
        self.fwh1.set_attr(pr1=1, pr2=0.99, DTU=5)
        self.fwh2.set_attr(pr1=1, pr2=0.99, DTU=5)
        self.dsh.set_attr(pr1=0.99, pr2=0.99)
        self.c0.set_attr(m=200, T=650, p=100, fluid={"water": 1})
        self.c2.set_attr(p=20)
        self.c4.set_attr(p=3)
        self.c6.set_attr(p=0.05)
        self.c41.set_attr(T=20, p=3, fluid={"water": 1})
        self.c42.set_attr(T=28)


        power_eta_doc = {'high pressure turbine': 0.9,
                         'mid pressure turbine': 0.9,
                         'low pressure turbine': 0.9,
                         'feed water pump': 0.8,
                         'feed water pump 2': 0.8,
                         'feed water pump 3': 0.8},
        design_param_group_doc = {'1': {'Component': {"steam generator": {'pr': 0.92},
                                                      "condenser": {"pr1": 1, "pr2": 0.99, "ttd_u": 5},
                                                      "feed water preheater 1": {"pr1": 1, "pr2": 0.99, "ttd_u": 5},
                                                      "feed water preheater 2": {"pr1": 1, "pr2": 0.99, "ttd_u": 5},
                                                      "desuperheater": {"pr1": 0.99, "pr2": 0.99}},
                                        'Connection': {'0': {'m': 200, 'T': 650, 'p': 100, 'fluid': {"water": 1}},
                                                       '2': {'p': 20},
                                                       '4': {'p': 3},
                                                       '41': {'T': 20, 'p': 3, 'fluid': {"INCOMP::Water": 1}},
                                                       '42': {'T': 28, 'p0': 3, 'h0': 100}}
                                        }},
        pass

    def solve(self, mode='design', max_iter=50):
        self.set_properties()
        self.nw.solve(mode=mode, max_iter=max_iter, plot_iteration=False, print_results=True, init_path=f"{self.name}_design")
        # self.nw.save(f"{self.name}_design")

if __name__ == "__main__":
    example_steam_plant1 = ExampleSteamPlantModel1('example_steam_plant1')
    example_steam_plant1.solve()
