# -*- coding: utf-8
import importlib.resources
import os

__datapath__ = os.path.join(importlib.resources.files("Aurora"), "data")
__version__ = '0.1.1 - GCC\'s Nature'

# Aurora data and connections import
from . import connections  # noqa: F401
from . import data  # noqa: F401
# Aurora components imports
from .components.fluid_components import combustion, deaerators, distributors, endpoints, heat_exchangers, piping, turbomachinery
from .components import component  # noqa: F401
from .components import reactors  # noqa: F401
from .components import subsystem  # noqa: F401
# Aurora networks imports
from .networks import network  # noqa: F401
# Aurora tools imports
from .tools import characteristics  # noqa: F401
from .tools import data_containers  # noqa: F401
from .tools import fluid_properties  # noqa: F401
from .tools import global_vars  # noqa: F401
from .tools import helpers  # noqa: F401
from .tools import logger  # noqa: F401
