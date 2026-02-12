# -*- coding: utf-8

"""Module for data container classes.

The DataContainer class and its subclasses are used to store component or
connection properties.
"""
import numpy as np

from Aurora.tools import logger


class DataContainer:
    """
    The DataContainer is parent class for all data containers.

    Parameters
    ----------
    **kwargs :
        See the class documentation of desired DataContainer for available
        keywords.

    Note
    ----
    The initialisation method (:code:`__init__`), setter method
    (:code:`set_attr`) and getter method (:code:`get_attr`) are used for
    instances of class DataContainer and its children. AURORA uses different
    :code:`DataContainer` classes for specific objectives:

    - component characteristics
      :py:class:`aurora.tools.data_containers.ComponentCharacteristics`
    - component characteristic maps
      :py:class:`aurora.tools.data_containers.ComponentCharacteristicMaps`
    - component properties
      :py:class:`aurora.tools.data_containers.ComponentProperties`
    - grouped component properites
      :py:class:`aurora.tools.data_containers.GroupedComponentProperties`
    - fluid composition
      :py:class:`aurora.tools.data_containers.FluidComposition`
    - fluid properties
      :py:class:`aurora.tools.data_containers.FluidProperties`

    Grouped component properties are used, if more than one component property
    has to be specified in order to apply one equation, e.g. pressure drop in
    pipes by specified length, diameter and roughness. If you specify all three
    of these properties, the DataContainer for the group will be created
    automatically!

    For the full list of available parameters for each data container, see its
    documentation.

    Example
    -------
    The examples below show the different (sub-)classes of DataContainers
    available.


    """

    def __init__(self, **kwargs):
        var = self.attr()  # get the Dictionary of available attributes (dictionary keys) with default values.
        self.label = None
        # default values
        for key in var.keys():
            self.__dict__.update({key: var[key]})  # set default value of children data container class
        # assign self_choice properties
        self.assign_choice_(**kwargs)
        # assign default properties
        self.set_attr(**kwargs)

    def set_attr(self, **kwargs):
        """
        Sets, resets or unsets attributes of a DataContainer type object.

        Parameters
        ----------
        **kwargs :
            See the class documentation of desired DataContainer for available
            keywords.
        """
        var = self.attr()
        # specify values
        for key in kwargs:
            if hasattr(self, key):  # key in var:
                self.__dict__.update({key: kwargs[key]})
            else:
                msg = (
                    f"Datacontainer of type {self.__class__.__name__} has no "
                    f"attribute \"{key}\"."
                )
                logger.error(msg)
                raise KeyError(msg)

    def get_attr(self, key):
        """
        Get the value of a DataContainer's attribute.

        Parameters
        ----------
        key : str
            The attribute you want to retrieve.

        Returns
        -------
        out :
            Specified attribute.
        """
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            msg = (
                f"Datacontainer of type {self.__class__.__name__} has no "
                f"attribute \"{key}\"."
            )
            logger.error(msg)
            raise KeyError(msg)

    def assign_choice_(self, **kwargs):
        if hasattr(self, "choice") and 'choice' in kwargs:
            for key in kwargs['choice']:
                self.__dict__.update({key: default_choice_func_})
                self.func_params.update({key: {}})

    @staticmethod
    def attr():
        """
        Return the available attributes for a DataContainer type object.

        Returns
        -------
        out : dict
            Dictionary of available attributes (dictionary keys) with default
            values.
        """
        return {}

    def _serialize(self):
        return {}


class ComponentCharacteristics(DataContainer):
    """
    Data container for component characteristics.

    Parameters
    ----------
    func : Aurora.components.characteristics.characteristics
        Function to be applied for this characteristics, default: None.

    is_set : boolean
        Should this equation be applied?, default: is_set=False.

    param : str
        Which parameter should be applied as the x value?
        default: method='default'.
    """

    @staticmethod
    def attr():
        """
        Return the available attributes for a ComponentCharacteristics
        type object.

        Returns
        -------
        out : dict
            Dictionary of available attributes (dictionary keys) with default
            values.
        """
        return {
            'char_func': None,
            'is_set': False,
            "take_effect": default_take_effect,
            'solve_isolated': default_solve_isolated,
            'param': None,  # variable
            'func_params': {},  # applied location and type
            'func': None,
            'variables_columns': None,
            'deriv': None,
            'repair_matrix': default_repair_matrix,
            'tensor': None,
            'char_params': {'type': 'rel', 'inconn': 0, 'outconn': 0},  # location and type of variable
            'row': None,
            'num_eq': 0,
            'scale': 1e0,
            'latex': None,
        }

    def _serialize(self):
        export = {}
        if self.char_func is not None:
            export.update({"char_func": self.char_func._serialize()})  # self.char_func._serialize(): attribute dict of charline

        for k in ["is_set", "param", "char_params"]:
            export.update({k: self.get_attr(k)})
        return export


class ComponentCharacteristicMaps(DataContainer):
    """
    Data container for characteristic maps.

    Parameters
    ----------
    func : Aurora.components.characteristics.characteristics
        Function to be applied for this characteristic map, default: None.

    is_set : boolean
        Should this equation be applied?, default: is_set=False.

    param : str
        Which parameter should be applied as the x value?
        default: method='default'.
    """

    @staticmethod
    def attr():
        """
        Return the available attributes for a ComponentCharacteristicMaps type
        object.

        Returns
        -------
        out : dict
            Dictionary of available attributes (dictionary keys) with default
            values.
        """
        return {
            'char_func': None,
            'is_set': False,
            'param': None,
            'func_params': {},
            'func': None,
            'variables_columns': None,
            "take_effect": default_take_effect,
            'solve_isolated': default_solve_isolated,
            'deriv': None,
            'repair_matrix': default_repair_matrix,
            'tensor': None,
            'row': None,
            'num_eq': 0,
            'scale': 1e0,
            'latex': None,
        }

    def _serialize(self):
        export = {}
        if self.char_func is not None:
            export.update({"char_func": self.char_func._serialize()})  # self.char_func._serialize(): attribute dict of char map

        for k in ["is_set", "param"]:
            export.update({k: self.get_attr(k)})
        return export


class ComponentProperties(DataContainer):
    """
    Data container for component properties.

    Parameters
    ----------
    val : float
        Value for this component attribute, default: val=1.

    val_SI : float
        Value in SI_unit (available for temperatures only, unit transformation
        according to network's temperature unit), default: val_SI=0.   ????

    is_set : boolean
        Has the value for this attribute been set?, default: is_set=False.

    is_var : boolean
        Is this attribute part of the system variables?, default: is_var=False.

    d : float
        Interval width for numerical calculation of partial derivative towards
        this attribute, it is part of the system variables, default d=1e-4.

    min_val : float
        Minimum value for this attribute, used if attribute is part of the
        system variables, default: min_val=1.1e-4.

    max_val : float
        Maximum value for this attribute, used if attribute is part of the
        system variables, default: max_val=1e12.
    """

    @staticmethod
    def attr():
        """
        Return the available attributes for a ComponentProperties type object.

        Returns
        -------
        out : dict
            Dictionary of available attributes (dictionary keys) with default
            values.
        """
        return {
            'val': 1,
            'val_SI': 0,
            'is_set': False,
            'is_var': False,
            "take_effect": default_take_effect,
            'solve_isolated': default_solve_isolated,
            'initialized': False,
            'd': 1e-4,
            'min_val': -1e12,
            'max_val': 1e12,
            'design': np.nan,
            'is_property': False,
            'is_result': False,
            'func_params': {},
            'func': None,
            'variables_columns': None,
            'deriv': None,
            'repair_matrix': default_repair_matrix,
            'tensor': None,
            'unit': None,
            'SI_unit': None,
            'property_data': None,
            'scale': 1e0,
            'var_scale': 1e0,
            'dimension': 1,
            "J_col": None,
            'row': None,
            'num_eq': 0,
            'latex': None
        }

    def _serialize(self):
        keys = self._serializable_keys()
        return {k: self.get_attr(k) for k in keys}

    @staticmethod
    def _serializable_keys():
        return [
            "val", "val_SI", "is_set", "d", "min_val", "max_val", "is_var",
        ]

    def extract_(self, index):
        if self.dimension == 1:
            if index == 0:
                return self.val_SI
            else:
                msg = f"Has something wrong in extracting {self.label}."
                raise ValueError(msg)
        else:
            return self.val_SI[index]

    def vest_(self, index, value):
        if self.dimension == 1:
            if index == 0:
                self.val_SI = value
            else:
                msg = f"Has something wrong in vesting {self.label}."
                raise ValueError(msg)
        else:
            self.val_SI[index] = value


class Constraints(DataContainer):
    """
    Data container for component constraints.
    """

    @staticmethod
    def attr():
        """
        Return the available attributes for a Constraints type object.
        Returns
        -------
        out : dict
            Dictionary of available attributes (dictionary keys) with default
            values.
        """
        return {
            'func': None,
            "take_effect": default_take_effect,
            "solve_isolated": default_solve_isolated,
            'variables_columns': None,
            'fluid_composition_list': [],
            'deriv': None,
            'repair_matrix': default_repair_matrix,
            'tensor': None,
            'constant_deriv': False,
            'latex': None,
            'row': None,
            'num_eq': 0,
            'unit': None,
            'SI_unit': None,
            'char': None,
            'scale': 1e0
        }


class GroupedComponentProperties(DataContainer):
    """
    Data container for grouped component parameters.

    Parameters
    ----------
    is_set : boolean
        Should the equation for this parameter group be applied?
        default: is_set=False.

    method : str
        Which calculation method for this parameter group should be used?
        default: method='default'.

    elements : list
        Which component properties are part of this component group?
        default elements=[].
    """

    @staticmethod
    def attr():
        """
        Return the available attributes for a GroupedComponentProperties type
        object.

        Returns
        -------
        out : dict
            Dictionary of available attributes (dictionary keys) with default
            values.
        """
        return {
            'is_set': False,
            "take_effect": default_take_effect,
            "solve_isolated": default_solve_isolated,
            'elements': [],
            'func_params': {},
            'func': None,
            'variables_columns': None,
            'deriv': None,
            'repair_matrix': default_repair_matrix,
            'tensor': None,
            'row': None,
            'num_eq': 0,
            'scale': 1e0,
            'unit': None,
            'SI_unit': None,
            'latex': None
        }


class GroupedComponentCharacteristics(DataContainer):
    """
    Data container for grouped component characteristics.

    Parameters
    ----------
    is_set : boolean
        Should the equation for this parameter group be applied?
        default: is_set=False.

    elements : list
        Which component properties are part of this component group?
        default elements=[].
    """

    @staticmethod
    def attr():
        """
        Return the available attributes for a GroupedComponentCharacteristics
        type object.

        Returns
        -------
        out : dict
            Dictionary of available attributes (dictionary keys) with default
            values.
        """
        return {
            'is_set': False,
            "take_effect": default_take_effect,
            "solve_isolated": default_solve_isolated,
            'elements': [],
            'func_params': {},
            'func': None,
            'variables_columns': None,
            'deriv': None,
            'repair_matrix': default_repair_matrix,
            'tensor': None,
            'row': None,
            'num_eq': 0,
            'scale': 1e0,
            'unit': None,
            'SI_unit': None,
            'latex': None
        }


class FitCoefficient(DataContainer):
    """
    Data container for fit coefficient.
    Parameters
    ----------
    choice: list
        Contain other properties defined in specific component.
    """

    def __call__(self, **kwargs):
        if self.get_attr(self.rule):
            kwargs_ = self.func_params[self.rule].copy()
            kwargs_.update(kwargs)
            return self.get_attr(self.rule)(**kwargs_)
        else:
            msg = f'Has no func in fit coefficient.'
            raise AttributeError(msg)

    @staticmethod
    def attr():
        """
        Return the available attributes for a FitCoefficient
        type object.

        Returns
        -------
        out : dict
            Dictionary of available attributes (dictionary keys) with default
            values.
        """
        return {
            'is_set': False,
            'choice': [],
            'rule': 'constant',
            'static': None,
            'constant': None,
            'default': None,
            'charline': None,
            'charmap': None,
            'self_defined': None,
            'func_params': {
                'static': {},
                'constant': {},
                'default': {},
                'charline': {},
                'charmap': {},
                'self_defined': {},
            },
        }


class FluidComposition(DataContainer):
    """
    Data container for fluid composition.

    Parameters
    ----------
    val : dict
        Mass fractions of the fluids in a mixture, default: val={}.
        Pattern for dictionary: keys are fluid name, values are mass fractions.

    val0 : dict
        Starting values for mass fractions of the fluids in a mixture,
        default: val0={}. Pattern for dictionary: keys are fluid name, values
        are mass fractions.

    is_set : dict
        Which fluid mass fractions have been set, default is_set={}.
        Pattern for dictionary: keys are fluid name, values are True or False.

    balance : boolean
        Should the fluid balance equation be applied for this mixture?
        default: False.
    """

    @staticmethod
    def attr():
        """
        Return the available attributes for a FluidComposition type object.

        Returns
        -------
        out : dict
            Dictionary of available attributes (dictionary keys) with default
            values.
        """
        return {
            'val': dict(),
            'val0': dict(),
            'scale': 1e0,
            'is_set': set(),
            'initialized': set(),
            'design': dict(),  # design value of fluid composition
            'wrapper': dict(),  # fluid calculating back_end information: fluid wrapper
            'back_end': dict(),  # just like INCOMP::Water
            'engine': dict(),  # the engine of calculating fluid properties
            "is_var": set(),
            "J_col": dict(),  # the column in Jacobin matrix
        }

    def _serialize(self):
        export = {"val": self.val}
        export["is_set"] = list(self.is_set)
        export["engine"] = {k: e.__name__ for k, e in self.engine.items()}
        export["back_end"] = {k: b for k, b in self.back_end.items()}
        return export


class FluidProperties(DataContainer):
    """
    Data container for fluid properties.

    Parameters
    ----------
    val : float
        Value in user specified unit (or network unit) if unit is unspecified,
        default: val=np.nan.

    val0 : float
        Starting value in user specified unit (or network unit) if unit is
        unspecified, default: val0=np.nan.

    val_SI : float
        Value in SI_unit, default: val_SI=0.

    is_set : boolean
        Has the value for this property been set? default: is_set=False.

    unit : str
        Unit for this property, default: unit=None.

    unit_set : boolean
        Has the unit for this property been specified manually by the user?
        default: unit_set=False.
    """

    @staticmethod
    def attr():
        r"""
        Return the available attributes for a FluidProperties type object.

        Returns
        -------
        out : dict
            Dictionary of available attributes (dictionary keys) with default
            values.
        """
        return {
            'val': np.nan,
            'val0': np.nan,
            'val_SI': 0,
            'min_val': -1e12,
            'max_val': 1e12,
            'unit': None,
            'SI_unit': None,
            'is_set': False,
            "is_var": False,
            'is_result': False,
            'property_data': None,
            "take_effect": default_take_effect,
            'solve_isolated': default_solve_isolated,
            'initialized': False,
            "is_shared": False,
            "shared_connection": [],
            'design': np.nan,  # design value(SI) of fluid property for iteration calculation
            "func_params": {},
            "func": None,
            'variables_columns': None,
            "deriv": None,
            'repair_matrix': default_repair_matrix,
            'tensor': None,
            "constant_deriv": False,  #
            'row': None,
            'scale': 1e0,
            "num_eq": 0,
            'dimension': 1,
            "J_col": None,  # the column in Jacobin matrix
            "_solved": False,  # pre_solve, some properties may be pre_solved (be calculated by setting not by iteration)
            "latex": None
        }

    def _serialize(self):
        keys = ["val", "val0", "val_SI", "is_set", "unit"]
        return {k: self.get_attr(k) for k in keys}

    def extract_(self, index):
        if self.dimension == 1:
            if index == 0:
                return self.val_SI
            else:
                msg = f"Has something wrong in extracting {self.label}."
                raise ValueError(msg)
        else:
            return self.val_SI[index]

    def vest_(self, index, value):
        if self.dimension == 1:
            if index == 0:
                self.val_SI = value
            else:
                msg = f"Has something wrong in vesting {self.label}."
                raise ValueError(msg)
        else:
            self.val_SI[index] = value


class ElectricProperties(DataContainer):

    def extract_(self, index):
        if self.dimension == 1:
            if index == 0:
                return self.val_SI
            else:
                msg = f"Has something wrong in extracting {self.label}."
                raise ValueError(msg)
        else:
            return self.val_SI[index]

    def vest_(self, index, value):
        if self.dimension == 1:
            if index == 0:
                self.val_SI = value
            else:
                msg = f"Has something wrong in vesting {self.label}."
                raise ValueError(msg)
        else:
            self.val_SI[index] = value


class ReferencedFluidProperties(DataContainer):

    @staticmethod
    def attr():
        r"""
        Return the available attributes for a FluidProperties type object.

        Returns
        -------
        out : dict
            Dictionary of available attributes (dictionary keys) with default
            values.
        """
        return {
            "ref": None,   # the object of Ref class, containing the referent connection object
            "unit": None,
            'SI_unit': None,
            "is_set": False,
            "take_effect": default_take_effect,
            'solve_isolated': default_solve_isolated,
            "func_params": {},
            "func": None,
            'variables_columns': None,
            "deriv": None,
            'repair_matrix': default_repair_matrix,
            'tensor': None,
            'row': None,
            "num_eq": 0,
            'scale': 1e0,
            "_solved": False  # 
        }

    def _serialize(self):
        if self.ref is not None:  #
            keys = ["is_set", "unit"]
            export = {k: self.get_attr(k) for k in keys}
            export["conn"] = self.ref.obj.label  # the label of referent connection object
            export["factor"] = self.ref.factor  # the proportionality coefficient
            export["delta"] = self.ref.delta  # the constant difference value
            return export
        else:
            return {}


class SimpleDataContainer(DataContainer):
    """
    Simple data container without data type restrictions to val field.

    Parameters
    ----------
    val : no specific datatype
        Value for the property, no predefined datatype.

    is_set : boolean
        Has the value for this property been set? default: is_set=False.
    """

    @staticmethod
    def attr():
        r"""
        Return the available attributes for a SimpleDataContainer type object.

        Returns
        -------
        out : dict
            Dictionary of available attributes (dictionary keys) with default
            values.
        """
        return {
            "val": np.nan,
            "is_set": False,
            "func_params": {},
            "func": None,
            'variables_columns': None,
            "take_effect": default_take_effect,
            'solve_isolated': default_solve_isolated,
            "deriv": None,
            'repair_matrix': default_repair_matrix,
            'tensor': None,
            'row': None,
            "num_eq": 0,
            'scale': 1e0,
            "var_scale": 1e0,
            "_solved": False,
            "latex": None
        }

    def _serialize(self):
        return {"val": self.val,
                "is_set": self.is_set}

def default_take_effect():
    return True

def default_solve_isolated(**kwargs):
    return False

def default_repair_matrix(property):
    msg = f"has not defined repair derive for {property.label}"
    raise ValueError(msg)

def default_choice_func_(**kwargs):
    msg = f'has no fit attribute '
    raise ValueError(msg)