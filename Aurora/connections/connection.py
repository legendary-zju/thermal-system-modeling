# -*- coding: utf-8
"""Module of class Connection and class Ref.
"""

import numpy as np

from Aurora.components.component import Component
# from Aurora.connections.fluid_connection import FluidConnection
# from Aurora.connections.electric_connection import ElectricConnection
from Aurora.tools import fluid_properties as fp
from Aurora.tools import logger
from Aurora.tools.data_containers import DataContainer as dc
from Aurora.tools.data_containers import FluidComposition as dc_flu
from Aurora.tools.data_containers import FluidProperties as dc_prop
from Aurora.tools.data_containers import ReferencedFluidProperties as dc_ref
from Aurora.tools.data_containers import SimpleDataContainer as dc_simple

from Aurora.tools.global_vars import ERR
from Aurora.tools.global_vars import min_derive
from Aurora.tools.global_vars import fluid_property_data as fpd
from Aurora.tools.global_vars import electromagnetic_property_data as epd
from Aurora.tools.global_vars import property_scale as ps
from Aurora.tools.helpers import AURORAConnectionError
from Aurora.tools.helpers import AURORANetworkError
from Aurora.tools.helpers import convert_from_SI


class Connection:
    r"""
    Class connection is the container for fluid properties between components.

    Parameters
    ----------
    design : list
        List containing design parameters (stated as string).

    offdesign : list
        List containing offdesign parameters (stated as string).

    design_path : str
        Path to individual design case for this connection.

    local_offdesign : boolean
        Treat this connection in offdesign mode in a design calculation.

    local_design : boolean
        Treat this connection in design mode in an offdesign calculation.

    printout : boolean
        Include this connection in the network's results printout.

    label : str
        Label of the connection. The default value is:
        :code:`'source:source_id_target:target_id'`.

    Note
    ----
    - The specification of values for design and/or offdesign is used for
      automatic switch from design to offdesign calculation: All parameters
      given in 'design', e.g. :code:`design=['T', 'p']`, are unset in any
      offdesign calculation, parameters given in 'offdesign' are set for
      offdesign calculation.

    - The property state is applied on pure fluids only. If you specify the
      desired state of the fluid at a connection the convergence check will
      adjust the enthalpy values of that connection for the first
      iterations in order to meet the state requirement.

    """
    def __new__(cls, source, outlet_id, target, inlet_id, label=None, **kwargs):
        """Determine class type before objective generation"""
        # type -> subclass
        from Aurora.connections.fluid_connection import FluidConnection
        from Aurora.connections.electric_connection import ElectricConnection
        type_to_class = {
            "fluid": FluidConnection,
            "electrical": ElectricConnection,
        }
        type_set = {source.interface_type(outlet_id), target.interface_type(inlet_id)}
        if len(type_set) != 1:
            msg = f'Connection type {type_set} of {label} not supported between {source.label}--{outlet_id} and {target.label}--{inlet_id}.'
            raise AURORAConnectionError(msg)
        component_type = type_set.pop()
        if component_type in type_to_class:
            subclass = type_to_class[component_type]
            return super().__new__(subclass)
        else:
            msg = f'Unknown connection type {component_type} in Connection: {label} generation'
            raise NotImplementedError(msg)

    def __init__(self, source, outlet_id, target, inlet_id,
                 label=None, **kwargs):
        self._check_types(source, outlet_id, target, inlet_id)
        self._check_self_connect(source, target)  # check whether the source is the target
        self._check_connector_id(source, outlet_id, source.outlets())
        self._check_connector_id(target, inlet_id, target.inlets())
        self.label = f"{source.label}:{outlet_id}_{target.label}:{inlet_id}"
        if label is not None:
            self.label = label
            if not isinstance(label, str):
                msg = "Please provide the label as string."
                logger.error(msg)
                raise TypeError(msg)
        # set specified values
        self.source = source
        self.source_id = outlet_id
        self.target = target
        self.target_id = inlet_id
        # defaults
        self.new_design = True  # load another design mode of single connection as reference
        self.design_path = None
        self.init_path = None
        self.design = []  # design variables
        self.offdesign = []  # offdesign variables
        self.local_design = False
        self.local_offdesign = False
        self.printout = True
        # set default values for kwargs
        self.property_data = self.get_parameters()  # all properties dict
        self.parameters = {
            k: v for k, v in self.get_parameters().items()
            if hasattr(v, "func") and v.func is not None}  # contain the user_set no_primary properties
        self.property_data0 = [x + '0' for x in self.property_data.keys()]
        self.__dict__.update(self.property_data)
        self.variables_properties_ = {
            'fluid': ["m", "p", "h"],
            'electric': ["U", "I", "f"]
        }
        msg = (
            f"Created connection from {self.source.label} ({self.source_id}) "
            f"to {self.target.label} ({self.target_id})."
        )
        logger.debug(msg)
        self.set_attr(**kwargs)

    def _check_types(self, source, outlet_id, target, inlet_id):
        # check input parameters
        if not (isinstance(source, Component) and
                isinstance(target, Component)):
            msg = (
                "Error creating connection. Check if source and target are "
                "Aurora.components."
            )
            logger.error(msg)
            raise TypeError(msg)
        # check the accuracy of type of source-connection-target
        if not source.interface_type(outlet_id) == target.interface_type(inlet_id):
            msg = (f'Has error in topological constructure generation due to '
                   f'{source.interface_type(outlet_id)} component: {source.label} connected to {target.interface_type(inlet_id)} component: {target.label} '
                   f'by {self.__class__.__name__}: {self.label}.')
            logger.error(msg)
            raise AURORAConnectionError(msg)

    def _check_self_connect(self, source, target):
        if source == target:
            msg = (
                "Error creating connection. Cannot connect component "
                f"{source.label} to itself."
            )
            logger.error(msg)
            raise AURORAConnectionError(msg)

    def _check_connector_id(self, component, connector_id, connecter_locations):
        if connector_id not in connecter_locations:
            msg = (
                "Error creating connection. Specified connector for "
                f"{component.label} ({connector_id}) is not available. Choose "
                f"from " + ", ".join(connecter_locations) + "."
            )
            logger.error(msg)
            raise ValueError(msg)

    def connection_type(self):
        msg = f'Has something wrong in defining connection type of {self.__class__.__name__}: {self.label}.'
        logger.error(msg)
        raise AURORAConnectionError(msg)

    def get_parameters(self):
        return {}

    def set_attr(self, **kwargs):
        r"""
        Set, reset or unset attributes of a connection.

        Parameters
        ----------
        design : list
            List containing design parameters (stated as string).

        offdesign : list
            List containing offdesign parameters (stated as string).

        design_path : str
            Path to individual design case for this connection.

        local_offdesign : boolean
            Treat this connection in offdesign mode in a design calculation.

        local_design : boolean
            Treat this connection in design mode in an offdesign calculation.

        printout : boolean
            Include this connection in the network's results printout.

        Note
        ----
        - The specification of values for design and/or offdesign is used for
          automatic switch from design to offdesign calculation: All parameters
          given in 'design', e.g. :code:`design=['T', 'p']`, are unset in any
          offdesign calculation, parameters given in 'offdesign' are set for
          offdesign calculation.
        - The property state is applied on pure fluids only. If you specify the
          desired state of the fluid at a connection the convergence check will
          adjust the enthalpy values of that connection for the first
          iterations in order to meet the state requirement.
        """
        # set specified values
        for key in kwargs:
            if key == 'label':
                msg = 'Label can only be specified on instance creation.'
                logger.error(msg)
                raise AURORAConnectionError(msg)
            elif 'unit' in key:
                prop = key.split('_unit')[0]
                if prop in self.property_data:
                    data = self.get_attr(prop)
                    if not hasattr(data, 'property_data'):
                        msg = f'Has something wrong in code definition or calling of {prop} in {self.__class__.__name__}: {self.label}.'
                        logger.error(msg)
                        raise AttributeError(msg)
                    if prop in fpd.keys():
                        if kwargs[key] in fpd[prop]['units']:
                            data.unit = kwargs[key]
                        else:
                            msg = f'Has not defined fluid property unit: {kwargs[key]} for {prop} in {self.__class__.__name__}: {self.label}.'
                            logger.error(msg)
                            raise AttributeError(msg)
                    elif prop in epd.keys():
                        if kwargs[key] in epd[prop]['units']:
                            data.unit = kwargs[key]
                        else:
                            msg = f'Has not defined electronic property unit: {kwargs[key]} for {prop} in {self.__class__.__name__}: {self.label}.'
                            logger.error(msg)
                            raise AttributeError(msg)
                    else:
                        msg = f'The {self.__class__.__name__}: {self.label} has something wrong in defining {key}.'
                        logger.error(msg)
                        raise NotImplementedError(msg)
                else:
                    msg = f'The {self.__class__.__name__}: {self.label} has no attribute {key}.'
                    logger.error(msg)
                    raise NotImplementedError(msg)
            elif key in self.property_data or key in self.property_data0:
                self._parameter_specification(key, kwargs[key])
            # design/offdesign parameter list
            elif key in ['design', 'offdesign']:
                if not isinstance(kwargs[key], list):
                    msg = f"Please provide the {key} parameters as list!"
                    logger.error(msg)
                    raise TypeError(msg)
                elif set(kwargs[key]).issubset(self.property_data.keys()):
                    self.__dict__.update({key: kwargs[key]})
                else:
                    params = ', '.join(self.property_data.keys())
                    msg = (
                        "Available parameters for (off-)design specification "
                        f"are: {params}."
                    )
                    logger.error(msg)
                    raise ValueError(msg)
            # design path
            elif key == 'design_path':  # no checking ????
                self.__dict__.update({key: kwargs[key]})
                self.new_design = True  # set another design path, reference another mode
            # other boolean keywords
            elif key in ['printout', 'local_design', 'local_offdesign']:
                if not isinstance(kwargs[key], bool):
                    msg = ('Please provide the ' + key + ' as boolean.')
                    logger.error(msg)
                    raise TypeError(msg)
                else:
                    self.__dict__.update({key: kwargs[key]})
            # invalid keyword
            else:
                msg = 'Connection has no attribute ' + key + '.'
                logger.error(msg)
                raise KeyError(msg)

    def _parameter_specification(self, key, value):
        try:
            float(value)
            is_numeric = True
        except (TypeError, ValueError):
            is_numeric = False

        if value is None:  # unset the property of connection
            self.get_attr(key).set_attr(is_set=False)
            if f"{key}_ref" in self.property_data:  # unset reference property
                self.get_attr(f"{key}_ref").set_attr(is_set=False)
            if key in self.variables_properties_[self.connection_type()]:  # m,p,h are the variables iterated by the solve_modul
                self.get_attr(key).is_var = True  # if m,p,h not been set, then will be var
        elif is_numeric:  # set the numeric property
            # value specification
            if key in self.property_data:
                self.get_attr(key).set_attr(is_set=True, val=value)
                if key in self.variables_properties_[self.connection_type()]:  # m,p,h are the variables iterated by the solve_modul
                    self.get_attr(key).is_var = False
            # starting value specification
            else:  # property_data0
                self.get_attr(key.replace('0', '')).set_attr(val0=value)
                self.get_attr(key.replace('0', '')).initialized = True
        # reference object
        elif isinstance(value, Ref):  # reference of other connections
            if f"{key}_ref" not in self.property_data:
                msg = f"Referencing {key} is not implemented."
                logger.error(msg)
                raise NotImplementedError(msg)
            else:
                self.get_attr(f"{key}_ref").set_attr(ref=value)
                self.get_attr(f"{key}_ref").set_attr(is_set=True)
        # invalid datatype for keyword
        else:
            msg = f"Wrong datatype for keyword argument {key}."
            logger.error(msg)
            raise TypeError(msg)

    def get_attr(self, key):
        r"""
        Get the value of a connection's attribute.

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
            msg = 'Connection has no attribute \"' + key + '\".'
            logger.error(msg)
            raise KeyError(msg)

    def _serialize(self):
        export = {}
        export.update({"source": self.source.label})
        export.update({"target": self.target.label})
        for k in self._serializable():
            export.update({k: self.get_attr(k)})
        for k in self.property_data:
            data = self.get_attr(k)
            export.update({k: data._serialize()})
        return {self.label: export}

    @staticmethod
    def _serializable():
        return [
            "source_id", "target_id",
            "design_path", "design", "offdesign", "local_design", "local_offdesign",
            "printout"
        ]

    def primary_ref_func(self, **kwargs):
        variable = kwargs["variable"]
        self.get_attr(variable)
        ref = self.get_attr(f"{variable}_ref").ref
        return self.get_attr(variable).val_SI - (ref.obj.get_attr(variable).val_SI * ref.factor + ref.delta_SI)

    def primary_ref_variables_columns(self, **kwargs):
        variables_columns1 = []
        variable = kwargs["variable"]
        ref = self.get_attr(f"{variable}_ref").ref
        if self.get_attr(variable).is_var:
            variables_columns1.append(self.get_attr(variable).J_col)
        if ref.obj.get_attr(variable).is_var:
            variables_columns1.append(ref.obj.get_attr(variable).J_col)
        variables_columns1.sort()
        return [variables_columns1]

    def primary_ref_take_effect(self, **kwargs):
        variable = kwargs["variable"]
        pass

    def primary_ref_solve_isolated(self, **kwargs):
        variable = kwargs["variable"]
        ref = self.get_attr(f"{variable}_ref").ref
        if self.get_attr(variable).is_var and ref.obj.get_attr(variable).is_var:
            return False
        elif self.get_attr(variable).is_var and not ref.obj.get_attr(variable).is_var:
            self.get_attr(variable).val_SI = (ref.obj.get_attr(variable).val_SI * ref.factor + ref.delta_SI)
            self.get_attr(variable).is_set = True
            self.get_attr(variable).is_var = False
            self.get_attr(f"{variable}_ref").is_set = False
            return True
        elif not self.get_attr(variable).is_var and ref.obj.get_attr(variable).is_var:
            ref.obj.get_attr(variable).val_SI = (self.get_attr(variable).val_SI - ref.delta_SI) / ref.factor
            ref.obj.get_attr(variable).is_set = True
            ref.obj.get_attr(variable).is_var = False
            self.get_attr(f"{variable}_ref").is_set = False
            return True
        elif self.get_attr(variable).is_var and ref.obj.get_attr(variable).is_var:
            return False
        else:
            self.get_attr(f"{variable}_ref").is_set = False
            return True

    def primary_ref_deriv(self, increment_filter, k, **kwargs):
        variable = kwargs["variable"]
        ref = self.get_attr(f"{variable}_ref").ref
        if self.get_attr(variable).is_var:
            self.network.jacobian[k, self.get_attr(variable).J_col] = 1
        if ref.obj.get_attr(variable).is_var:
            self.network.jacobian[k, ref.obj.get_attr(variable).J_col] = -ref.factor

    def primary_ref_tensor(self, increment_filter, k, **kwargs):
        pass

    def summarize_equations(self):
        """ Perform connection initialization in network preprocessing. """
        self.num_eq = 0
        self.it = 0
        self.equations = {}  # parameter(not include m, p, h, f) index dict of connection
        # serve for the count of self_set equations and no_primary variables
        for parameter in self.parameters:  # not include m, p, h, fluid composition
            container = self.get_attr(parameter)
            if container.is_set and container.take_effect():
                # if not set, the no_primary variable/equation won't be considered
                # some properties may be pre_solved (be calculated by setting not by iteration)
                self.equations[self.num_eq] = parameter
                self.num_eq += self.parameters[parameter].num_eq
        # logger for debug
        msg = f"The connection: {self.label} has {self.num_eq} equations."
        logger.debug(msg)
        # equations containing no_primary equation set, index: parameter
        for k, parameter in self.equations.items():
            data = self.get_attr(parameter)
            data.label = f"<connection property>: {parameter} of {self.label}"
            self.network.sorted_equations_module_container.append(data)

    def calc_results(self):
        pass


class Ref:
    r"""
    A reference object is used to reference (unknown) properties of connections
    to other connections.

    For example, reference the mass flow of one connection :math:`\dot{m}` to
    another mass flow :math:`\dot{m}_{ref}`:

    .. math::

        \dot{m} = \dot{m}_\mathrm{ref} \cdot \mathrm{factor} + \mathrm{delta}

    Parameters
    ----------
    obj : Aurora.connections.connection.Connection
        Connection to be referenced.

    factor : float
        Factor to multiply specified property with.

    delta : float
        Delta to add after multiplication.
    """

    def __init__(self, ref_obj, factor, delta):
        if not isinstance(ref_obj, Connection):
            msg = 'First parameter must be object of type connection.'
            logger.error(msg)
            raise TypeError(msg)
        if not (isinstance(factor, int) or isinstance(factor, float)):
            msg = 'Second parameter must be of type int or float.'
            logger.error(msg)
            raise TypeError(msg)
        if not (isinstance(delta, int) or isinstance(delta, float)):
            msg = 'Thrid parameter must be of type int or float.'
            logger.error(msg)
            raise TypeError(msg)
        self.obj = ref_obj
        self.factor = factor
        self.delta = delta
        self.delta_SI = None  # set in the initialise: init_set_properties in network
        msg = (
            f"Created reference object with factor {self.factor} and delta "
            f"{self.delta} referring to connection {ref_obj.label}"
        )
        logger.debug(msg)

    def get_attr(self, key):
        r"""
        Get the value of a reference attribute.

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
            msg = 'Reference has no attribute \"' + key + '\".'
            logger.error(msg)
            raise KeyError(msg)
