# -*- coding: utf-8

"""Module class ElectricComponent.
"""

import math

import numpy as np

from Aurora.components.component import Component
from Aurora.components.component import component_registry

from Aurora.tools import logger
from Aurora.tools.characteristics import CharLine
from Aurora.tools.characteristics import CharMap
from Aurora.tools.characteristics import load_default_char as ldc
from Aurora.tools.data_containers import ComponentCharacteristicMaps as dc_cm
from Aurora.tools.data_containers import ComponentCharacteristics as dc_cc
from Aurora.tools.data_containers import ComponentProperties as dc_cp
from Aurora.tools.data_containers import GroupedComponentCharacteristics as dc_gcc
from Aurora.tools.data_containers import GroupedComponentProperties as dc_gcp
from Aurora.tools.data_containers import SimpleDataContainer as dc_simple
from Aurora.tools.data_containers import Constraints as dc_cons
from Aurora.tools.document_models import generate_latex_eq

from Aurora.tools.global_vars import ERR
from Aurora.tools.helpers import bus_char_derivative
from Aurora.tools.helpers import bus_char_evaluation
from Aurora.tools.helpers import newton_with_kwargs
from Aurora.tools.helpers import convert_to_SI
from Aurora.tools.helpers import convert_from_SI


@component_registry
class ElectricComponent(Component):
    r"""
    Class ElectricComponent is the base class of all AURORA electric components.
    """

    @staticmethod
    def is_frequency_branch_source():
        return False

    def start_electric_frequency_branch(self):
        msg = f'The electric component {self.__class__.__name__}: {self.label} has no electric frequency branch start attribute.'
        logger.error(msg)
        raise NotImplementedError(msg)

    def component_type(self):
        return ['electric']

    def interface_type(self, port_id):
        return 'electric'

    def numeric_deriv(self, func, dx, conn=None, **kwargs):  # calculate discrete derivatives
        r"""
        Calculate partial derivative of the function func to dx.
        For details see :py:func:`Aurora.tools.helpers._numeric_deriv`
        """
        pass

    def get_char_expr(self, param, type='rel', inconn=0, outconn=0):  # need to be rewritten !!!
        r"""
        Generic method to access characteristic function parameters (ratio).

        Parameters
        ----------
        param : str
            Parameter for characteristic function evaluation.

        type : str
            Type of expression:

            - :code:`rel`: relative to design value
            - :code:`abs`: absolute value

        inconn : int
            Index of inlet connection.

        outconn : int
            Index of outlet connection.

        Returns
        -------
        expr : float
            Value of expression
        """
        pass







