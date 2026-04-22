# -*- coding: utf-8

"""Module for free fluid property container.
Aurora/tools/fluid_properties/property_container.py

SPDX-License-Identifier: MIT
"""

import numpy as np
from scipy.interpolate import interp1d, interp2d, LinearNDInterpolator, RegularGridInterpolator


class FluidPropertyContainer:
    def __init__(self, value=None, type='constant', intro={}):
        """
        Container for fluid properties.

        value: default value
        type: type of calculation
             - constant
             - polynomial
             - correlation
             - table_lookup
        intro: fit coefficient
        """
        self.type = type
        self.value = value
        self.intro = intro
        self.correlation_types = {
            'polynomial': self._polynomial,
            'exponential': self._exponential,
            'linear_expansion': self._linear_expansion,
            'power_law': self._power_law,
            'sutherland': self._sutherland,
            'table_lookup': self._table_lookup
        }
        self._interpolator = None
        if self.type == 'table_lookup':
            self._initialize_interpolator()

    def __call__(self, **kwargs):
        if self.type == 'constant':
            return self.value
        elif self.type in self.correlation_types:
            return self.correlation_types[self.type](**kwargs)
        else:
            return self.value

    def _polynomial(self, **kwargs):
        """polynomial calculation"""
        parameter = self.intro.get('parameters', ['T'])[0]
        x = kwargs[parameter]
        coeffs = self.intro.get('coefficients', [])
        result = 0.0
        for i, coeff in enumerate(coeffs):
            result += coeff * (x ** i)
        return result

    def _exponential(self, **kwargs):
        """exponential associated formula"""
        parameter = self.intro.get('parameters', ['T'])[0]
        x = kwargs[parameter]
        A = self.intro.get('A', 1.0)
        B = self.intro.get('B', 0.0)
        C = self.intro.get('C', 0.0)
        return A * np.exp(B / x + C * x)

    def _linear_expansion(self, **kwargs):
        """linear expansion associated formula"""
        parameter = self.intro.get('parameters', ['T'])[0]
        x = kwargs[parameter]
        value0 = self.intro.get('value0', 0.0)
        alpha = self.intro.get('alpha', 1.0)
        belta = self.intro.get('belta', 0.0)
        return value0 * (1- alpha * (x - belta))

    def _power_law(self, **kwargs):
        """power law calculation"""
        parameter = self.intro.get('parameters', ['T'])[0]
        x = kwargs[parameter]
        A = self.intro.get('A', 1.0)
        n = self.intro.get('n', 0.0)
        return A * (x ** n)

    def _sutherland(self, **kwargs):
        """Sutherland experimental formula"""
        parameter = self.intro.get('parameters', ['T'])[0]
        x = kwargs[parameter]
        mu0 = self.intro.get('mu0', 1.0e-3)
        T0 = self.intro.get('T0', 273.15)
        S = self.intro.get('S', 110.4)
        return mu0 * (x / T0) ** 1.5 * (T0 + S) / (x + S)

    def _initialize_interpolator(self):
        """initialize interpolator"""
        table_data = self.intro.get('table', {})
        if not table_data:
            return
        method = self.intro.get('method', 'linear')
        bounds_error = self.intro.get('bounds_error', False)
        fill_value = self.intro.get('fill_value', 'extrapolate')
        parameters = self.intro.get('parameters', ['T'])
        # check dimension
        if 'data' in table_data:
            # 新格式：多维数据
            axes = table_data.get('axes', [])
            data = np.array(table_data['data'])
            if len(axes) == 1:
                # 1D interpolation
                x = np.array(axes[0])
                y = np.array(data)
                self._interpolator = interp1d(x, y, kind=method,
                                              bounds_error=bounds_error,
                                              fill_value=fill_value)
            elif len(axes) == 2:
                # 2D interpolation
                x = np.array(axes[0])
                y = np.array(axes[1])
                z = np.array(data)
                if method == 'linear':
                    self._interpolator = RegularGridInterpolator((x, y), z,
                                                                 method='linear',
                                                                 bounds_error=bounds_error,
                                                                 fill_value=np.nan if fill_value == 'extrapolate' else fill_value)
                else:
                    self._interpolator = interp2d(x, y, z, kind=method)
            else:
                # nD interpolation
                points = np.array(table_data['points'])
                values = np.array(table_data['values'])
                self._interpolator = LinearNDInterpolator(points, values)
        elif 'x' in table_data and 'y' in table_data:
            # old form: 1D interpolation
            x = np.array(table_data['x'])
            y = np.array(table_data['y'])
            self._interpolator = interp1d(x, y, kind=method,
                                          bounds_error=bounds_error,
                                          fill_value=fill_value)

    def _table_lookup(self, **kwargs) -> float:
        """
        table lookup interpolation

        supported table format：
        1. 1D table：{'x': [x1, x2, ...], 'y': [y1, y2, ...]}
        2. 2D table：{'axes': [[x1, x2, ...], [y1, y2, ...]], 'data': [[z11, z12, ...], ...]}
        3. nD table：{'points': [[x1, y1, z1], ...], 'values': [v1, v2, ...]}

        parameters:
            kwargs: containing all parameters required
                   such as：T=600.0, p=101325.0, x=0.5
        """
        if self._interpolator is None:
            self._initialize_interpolator()
        if self._interpolator is None:
            return self.value if self.value is not None else 0.0
        # get parameters
        parameters = self.intro.get('parameters', ['T'])
        missing_params = [p for p in parameters if p not in kwargs]
        if missing_params:
            raise ValueError(f"missing parameters: {missing_params} at fluid property container defined by self")
        args = [kwargs[p] for p in parameters]
        # interpolation
        try:
            if len(parameters) == 1:  # 1D interpolation
                return float(self._interpolator(args[0]))
            elif len(parameters) == 2:  # 2D interpolation
                if isinstance(self._interpolator, RegularGridInterpolator):
                    return float(self._interpolator(args))
                else:
                    return float(self._interpolator(args[0], args[1]))
            else:  # nD interpolation
                return float(self._interpolator(args))
        except (ValueError, TypeError) as e:
            fallback_method = self.intro.get('fallback_method', 'nearest')
            if fallback_method == 'nearest':
                return self._nearest_lookup(**kwargs)
            elif fallback_method == 'constant':
                return self.value if self.value is not None else 0.0
            else:
                msg = f"fallback_method {fallback_method} not recognized at fluid property container defined by self" + f"{e}"
                raise ValueError(msg)

    def _nearest_lookup(self, **kwargs) -> float:
        """nearest lookup interpolation"""
        table_data = self.intro.get('table', {})
        parameters = self.intro.get('parameters', ['T'])
        if 'data' in table_data and 'axes' in table_data:
            axes = table_data['axes']
            data = np.array(table_data['data'])
            if len(parameters) == 1:
                x_vals = np.array(axes[0])
                target_x = kwargs[parameters[0]]
                idx = np.abs(x_vals - target_x).argmin()
                return float(data[idx])
            elif len(parameters) == 2:
                x_vals = np.array(axes[0])
                y_vals = np.array(axes[1])
                target_x = kwargs[parameters[0]]
                target_y = kwargs[parameters[1]]
                idx_x = np.abs(x_vals - target_x).argmin()
                idx_y = np.abs(y_vals - target_y).argmin()
                return float(data[idx_x, idx_y])
        elif 'x' in table_data and 'y' in table_data:
            x_vals = np.array(table_data['x'])
            y_vals = np.array(table_data['y'])
            target_x = kwargs[parameters[0]]
            idx = np.abs(x_vals - target_x).argmin()
            return float(y_vals[idx])
        # default
        return self.value if self.value is not None else 0.0
