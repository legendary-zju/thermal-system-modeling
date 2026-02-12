from Aurora.tools import logger


class Node:
    """
    Derivative node of components.
    """
    def __new__(cls, component, comp_type, **kwargs):
        """Determine class type before objective generation"""
        from Aurora.nodes.fluid_node import FluidNode
        from Aurora.nodes.electric_node import ElectricNode
        type_to_class = {
            "fluid": FluidNode,
            "electrical": ElectricNode,
        }
        if comp_type in type_to_class:
            subclass = type_to_class[comp_type]
            return super().__new__(subclass)
        else:
            msg = f'Unknown component type {comp_type} in Node generation of {component.__class__.__name__} {component.label}'
            raise NotImplementedError(msg)

    def __init__(self, component, comp_type, **kwargs):
        self.component = component
        self.type_ = comp_type
        self.properties = self.set_properties()
        self.__dict__.update(self.properties)

    def set_properties(self):
        return {

        }

    def set_attr(self):
        pass

    def get_attr(self, key):
        r"""
        Get the value of a node's attribute.

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
            msg = f'{self.__class__.__name__} has no attribute \"' + key + '\".'
            logger.error(msg)
            raise KeyError(msg)

