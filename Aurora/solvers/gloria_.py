# -*- coding: utf-8

import numpy as np
import pandas as pd
import copy as cp
import random
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# matplotlib.use('Qt5Agg')
import math
from numpy.linalg import norm
from time import time

from Aurora.tools import logger

from .particles_ import Particles


class Gloria:
    def __init__(self, n, m, problem_obj, lb, ub, cl, cu):
        self.n, self.m = n, m
        self.problem_obj = problem_obj
        self.lb, self.ub = lb, ub
        self.cl, self.cu = cl, cu

        self.variables_dict = self.problem_obj.network.variables_dict
        self.constraints_groups = self.problem_obj.network.constraints

        self.x = np.zeros(self.n)
        self.increment = np.zeros(self.n)
        self.jacobin = np.zeros([self.n, self.n])
        self.jacobin_slipy = np.zeros([self.n, self.n])
        self.residual = np.zeros(self.n)
        self.residual_slipy = np.zeros(self.n)
        self.cov_matrix_exp = np.eye(self.n)
        self.particles_pool = []
        self.particles_state = True

        self.iter = 0
        self.original_factor = [0.5, 0.05, 0.005, 0.5]
        self.original_factor_std = [0.1, 0.01, 0.001, 0.1]

        self.progress = True
        self.converged = False

        self.classify_variables()
        self.add_option()

    def classify_variables(self):
        """
        Generate the index of each variable class.

        Returns
        -------

        """
        self.m_index = [k for k, v in self.variables_dict.items() if v["variable"] == "m"]
        self.p_index = [k for k, v in self.variables_dict.items() if v["variable"] == "p"]
        self.h_index = [k for k, v in self.variables_dict.items() if v["variable"] == "h"]
        self.fl_index = [k for k, v in self.variables_dict.items() if v["variable"] == "fluid"]
        self.cp_index = [k for k in self.variables_dict if k not in self.m_index + self.p_index + self.h_index + self.fl_index]

    def add_option(self, min_iter: int = 5, max_iter: int = 50, tol: float = 1e-8,
                   max_particle : int = 9, min_belief : float = 0.05,
                   max_layer: int = 20, plot_iteration: bool = False,
                   iterinfo: bool = True):
        """
        Set algorithm configuration.

        Parameters
        ----------
        min_iter : minimum number of iterations
        max_iter: max iteration number
        tol: tolerance number
        max_particle: maximum number of particles
        min_belief: minimum belief of the particles
        max_layer: maximum number of particle lawyers
        plot_iteration: plot iteration
        iterinfo: iteration information

        Returns
        -------

        """
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.tol = tol
        self.max_particle = max_particle
        self.min_belief = min_belief
        self.max_layer = max_layer
        self.plot_iteration = plot_iteration
        self.iterinfo = iterinfo

    def set_original_x(self, variables):
        """
        Looking for more suitable variables.

        Parameters
        ----------
        variables: np.array

        Returns
        -------

        """
        if not self.feasible_bounds_check(variables, 'original variables generation'):
            raise ValueError('The variables is not feasible in the starting')
        self.x = variables
        self.cov_matrix_exp = 100 * np.eye(self.n)
        pass

    def generate_particles(self, layer):
        """
        Generate the particle objects.

        Parameters
        -------
        layer: int

        Returns
        -------

        """
        self.particles_pool = []
        self.particles_pool_load = []
        # generate factor bounds
        self.generate_factor_bounds()
        for i in range(self.max_particle):
            particle = Particles((layer, i), self.n)
            particle.variables = self.x.copy()
            particle.residual_slipy = self.residual_slipy.copy()
            particle.jacobin_slipy = self.jacobin_slipy.copy()
            # distribute factors for each particle
            self.distribute_factor(particle)
            # containing particles
            self.particles_pool.append(particle)

    def feasible_bounds_check(self, variables, func_inf=''):
        """
        Check the feasibility of variables.

        Parameters
        ----------
        variables: np.array
        func_inf: str

        Returns
        -------
        the feasibility of variables of particles: bool

        """
        # bounds check
        if np.all(variables > self.lb) and np.all(variables < self.ub):
            try:
                constraints_vector = self.problem_obj.constraints(variables)
                # constraints check
                if np.all(constraints_vector > self.cl) and np.all(constraints_vector < self.cu):
                    return True
                else:
                    cl_violations = np.where(constraints_vector <= self.cl)[0].tolist()
                    cu_violations = np.where(constraints_vector >= self.cu)[0].tolist()
                    cl_vio_info = [self.constraints_groups[index]['info']
                                   + f"--{constraints_vector[index]} should under {self.cl[index]}"
                                   for index in cl_violations]
                    cu_vio_info = [self.constraints_groups[index]['info']
                                   + f"--{constraints_vector[index]} should upper {self.cu[index]}"
                                   for index in cu_violations]
                    logger.debug(
                        f"Constraints violation in {func_inf}. "
                        f"Lower-bound violations:" + "[\n  " + ",\n  ".join(cl_vio_info) + "\n],"
                        f"Upper-bound violations:" + "[\n  " + ",\n  ".join(cu_vio_info) + "\n]."
                    )
                    return False
            except ValueError as e:
                logger.debug(f"Has something wrong with constraints vector generation in {func_inf} due to {e}")
                return False
        else:
            logger.debug(f"The variables violate variables bounds in {func_inf}")
            return False

    def prior_calculation(self, cov_matrix_exp, layer):
        """
        Calculate the prior belief of all particles at beginning.

        Parameters
        ----------
        cov_matrix_exp: np.array
        layer: int

        Returns
        -------

        """
        x = self.x.copy()
        if self.feasible_bounds_check(x, f"Particles prior belief generation of layer: {layer}"):
            try:
                residual = self.problem_obj.get_residual(x)
                prior_belief = (self.sigmoid_estimate(self.problem_obj.constraints(x),
                                                      self.problem_obj.constraints_k_list,
                                                      self.problem_obj.constraints_epsilon_list)
                                * self.exp_estimate(cov_matrix_exp, residual, self.n))
                for particle in self.particles_pool:
                    particle.prior_belief = prior_belief
                    particle.residual_temp = residual.copy()
                    particle.feasible = True
            except ValueError as e:
                logger.debug(f'Has something wrong with residual calculation in prior belief generation of layer: {layer}: {e}')
                prior_belief = 0
                self.particles_state = False
                for particle in self.particles_pool:
                    particle.prior_belief = prior_belief
                    particle.feasible = False
        else:
            prior_belief = 0
            self.particles_state = False
            for particle in self.particles_pool:
                particle.prior_belief = prior_belief
                particle.feasible = False
            logger.debug(f'The prior belief generation of all particles has been defeated of layer: {layer}.')

    def predict_calculation(self, cov_matrix_exp, layer):
        """
        Calculate the predict belief of particles.

        Parameters
        ----------
        cov_matrix_exp: np.array
        layer: int

        Returns
        -------

        """
        for particle in self.particles_pool:
            variables = particle.variables
            residual = particle.residual_temp + particle.jacobin_temp @ particle.delta_variables  # the near residual
            if particle.feasible and self.feasible_bounds_check(variables, f'predict belief generation of particle: {particle.index} of layer: {layer}.'):
                predict_belief = (self.sigmoid_estimate(self.problem_obj.constraints(variables),
                                                        self.problem_obj.constraints_k_list,
                                                        self.problem_obj.constraints_epsilon_list)
                                  * self.exp_estimate(cov_matrix_exp, residual, self.n))
                particle.predict_belief = predict_belief
                particle.feasible = True
            else:
                particle.predict_belief = 0
                particle.feasible = False

    def posterior_calculation(self, cov_matrix_exp, layer):
        """
        Calculate the posterior belief of particles.

        Parameters
        ----------
        cov_matrix_exp: np.array
        layer: int

        Returns
        -------

        """
        for particle in self.particles_pool:
            x = particle.variables
            if particle.feasible:  #  and self.feasible_bounds_check(x, f'posterior belief generation of particle: {particle.index} of layer: {layer}.'):
                try:
                    residual = self.problem_obj.get_residual(x)
                    particle.residual_temp = residual.copy()
                    particle.feasible = True
                    particle.posterior_belief = (self.sigmoid_estimate(self.problem_obj.constraints(x),
                                                                       self.problem_obj.constraints_k_list,
                                                                       self.problem_obj.constraints_epsilon_list)
                                                 * self.exp_estimate(cov_matrix_exp, residual, self.n))
                except ValueError as e:
                    logger.debug(f"Has something wrong with residual calculation of particle: {particle.index}"
                                   f" in posterior belief generation of layer: {layer}: {e}")
                    particle.posterior_belief = 0
                    particle.feasible = False
            else:
                logger.debug(f"The posterior belief generation of particle: {particle.index} of layer: {layer} "
                               f"has been defeated due to feasible constraints check.")
                particle.posterior_belief = 0
                particle.feasible = False

    def movement(self, layer):
        """
        Move the particles.

        Parameters
        -------
        layer: int

        Returns
        -------

        """
        for particle in self.particles_pool:
            x = particle.variables
            if particle.feasible:
                try:
                    particle.jacobin_temp = self.problem_obj.get_jacobin(x).copy()
                    # slipy function groups
                    self.slipy_cg(particle)
                    # update variables
                    particle.variables = x + particle.delta_variables
                except ValueError as e:
                    particle.feasible = False
                    logger.error(f'Has something wrong with jacobin calculation of particle: {particle.index} in layer: {layer} due to: {e}')
            else:
                logger.error(f"Has something wrong with regeneration of particle: {particle.index} in layer: {layer}.")

    @staticmethod
    def exp_estimate(cov_matrix, residual, n):
        """
        Calculate the exp estimate based on residual.

        Parameters
        ----------
        cov_matrix
        residual: np.array
        n: int

        Returns
        -------

        """
        p_res = np.exp(-0.5 * residual.T @ np.linalg.inv(cov_matrix) @ residual) / (((2 * np.pi) ** (0.5 * n)) * (np.linalg.det(cov_matrix) ** 0.5))
        return p_res

    @staticmethod
    def sigmoid_estimate(g_x_list, k_list, epsilon_list):
        """
        Calculate the sigmoid estimate based on bounds and constraints.

        Parameters
        ----------
        g_x_list: np.array
        k_list: np.array
        epsilon_list: np.array

        Returns
        -------

        """
        def element_func(g_x, k_i, epsilon_i):
            return k_i / (np.exp(-0.5 * k_i * (g_x - epsilon_i)) + 1)
        p_cons = np.prod(element_func(g_x_list, k_list, epsilon_list))
        return p_cons

    def slipy_cg(self, particle):
        """
        Calculate the increment of variables based on the slippy CG algorithm.

        Parameters
        ----------
        particle: Particle object

        Returns
        -------

        """
        alpha = particle.alpha
        beta = particle.beta
        gamma = particle.gamma
        particle.jacobin_slipy = (1 - alpha) * particle.jacobin_slipy + alpha * particle.jacobin_temp
        particle.residual_slipy = (1 - beta) * particle.residual_slipy + beta * particle.residual_temp
        jacobin = particle.jacobin_slipy.copy()
        residual = particle.residual_slipy.copy()
        try:
            hessian = jacobin.T @ jacobin + gamma * np.eye(self.n)
            increment = np.zeros([self.n])
            gradient = - jacobin.T @ residual
            direction = - gradient
            for _ in range(self.n):
                alpha_k = ((residual.T @ jacobin @ direction - increment.T @ hessian @ direction) /
                           (direction.T @ hessian @ direction))
                increment = increment + alpha_k * direction
                gradient = hessian @ increment - jacobin.T @ residual
                belta_k = ((gradient.T @ hessian @ direction) /
                           (direction.T @ hessian @ direction))
                direction = - gradient + belta_k * direction
                if np.any(np.isnan(increment)):
                    raise ValueError
            increment = - increment
        except ValueError:
            particle.feasible = False  #
            increment = np.zeros([self.n])

        particle.delta_variables = increment * particle.zeta

    def generate_factor_bounds(self):
        """
        Generate factor bounds before generating particles.

        Returns
        -------

        """
        self.alpha_low = max(0, self.original_factor[0] - 3 * self.original_factor_std[0])
        self.alpha_high = min(1, self.original_factor[0] + 3 * self.original_factor_std[0])

        self.beta_low = max(0, self.original_factor[1] - 3 * self.original_factor_std[1])
        self.beta_high = min(0.1, self.original_factor[1] + 3 * self.original_factor_std[1])

        self.gamma_low = max(0, self.original_factor[2] - 3 * self.original_factor_std[2])
        self.gamma_high = min(0.01, self.original_factor[2] + 3 * self.original_factor_std[2])

        self.zeta_low = max(0, self.original_factor[3] - 3 * self.original_factor_std[3])
        self.zeta_high = min(1, self.original_factor[3] + 3 * self.original_factor_std[3])

    def distribute_factor(self, particle):
        """
        Distribute factor value for each particle.

        Parameters
        ----------
        particle: Particle object

        Returns
        -------

        """
        particle.alpha = np.random.uniform(self.alpha_low, self.alpha_high)
        particle.beta = np.random.uniform(self.beta_low, self.beta_high)
        particle.gamma = np.random.uniform(self.gamma_low, self.gamma_high)
        particle.zeta = np.random.uniform(self.zeta_low, self.zeta_high)

    def update_particles(self, layer):
        """
        Update elite particles, obsolete inferior particles.

        Parameters
        ----------
        layer

        Returns
        -------

        """
        if self.particles_state and layer > 0:
            # adaptive adjustment
            decay = 0.95 ** min(layer, 10)  #
            # clip
            self.generate_factor_bounds()

            for index, particle in enumerate(self.particles_pool):
                # inherit elite particles
                if particle.is_elite and particle.feasible:
                    weight = self.elite_weights[index]
                    perturbation_factor = max(0.001, (1 - weight) * decay)
                    particle.alpha = np.clip(
                        particle.alpha * random.gauss(1.0, 0.18 * perturbation_factor),
                        self.alpha_low, self.alpha_high
                    )
                    particle.beta = np.clip(
                        particle.beta * random.gauss(1.0, 0.1 * perturbation_factor),
                        self.beta_low, self.beta_high
                    )
                    particle.gamma = np.clip(
                        particle.gamma * random.gauss(1.0, 0.15 * perturbation_factor),
                        self.gamma_low, self.gamma_high
                    )
                    particle.zeta = np.clip(
                        particle.zeta * random.gauss(1.0, 0.1 * perturbation_factor),
                        self.zeta_low, self.zeta_high
                    )
                # regenerate inferior particles
                else:
                    if self.elite_weights and random.choice([True, False]):
                        template_idx = random.choices(
                            [i for i in self.elite_weights.keys()],
                            weights=list(self.elite_weights.values())
                        )[0]
                        template_particle = self.particles_pool[template_idx]
                        particle.alpha = np.clip(
                            template_particle.alpha * random.gauss(1.0, 0.15),
                            self.alpha_low, self.alpha_high
                        )
                        particle.beta = np.clip(
                            template_particle.beta * random.gauss(1.0, 0.15),
                            self.beta_low, self.beta_high
                        )
                        particle.gamma = np.clip(
                            template_particle.gamma * random.gauss(1.0, 0.15),
                            self.gamma_low, self.gamma_high
                        )
                        particle.zeta = np.clip(
                            template_particle.zeta * random.gauss(1.0, 0.15),
                            self.zeta_low, self.zeta_high
                        )
                    else:
                        self.distribute_factor(particle)
                # recover
                particle.index = (layer, index)
                particle.jacobin_slipy = self.jacobin_slipy.copy()
                particle.residual_slipy = self.residual_slipy.copy()
                particle.variables = self.x.copy()
                particle.feasible = True
                particle.is_elite = False
        else:
            if layer > 0:
                logger.debug(f"Not updating elite particles in layer: {layer}.")

    @staticmethod
    def value_measure(predict_belief, posterior_belief, prev_posterior):
        """
        Value of measuring particle behavior.

        Parameters
        ----------
        predict_belief
        posterior_belief
        prev_posterior

        Returns
        -------

        """
        if predict_belief == 0 or posterior_belief == 0:
            return - np.inf
        else:
            base_score = posterior_belief * 100
            # differential punishment
            prediction_error = abs(predict_belief - posterior_belief) / (posterior_belief + 1e-30)
            accuracy_penalty = 2 * max(prediction_error, 1e-10)  #
            # progress award
            if prev_posterior == 0:
                improvement_bonus = 1
            else:
                relative_improve = (prev_posterior) / (posterior_belief + 1e-30)
                improvement_bonus = 5 * (relative_improve)  #
            #
            measure_value = math.log10(base_score) - math.log10(accuracy_penalty) - math.log10(improvement_bonus)
            return measure_value

    def optimal_value_measure(self, layer):
        """
        Optimize the action of algorithm.

        Returns
        -------

        """
        max_belief = 0
        value_space = np.zeros([self.max_particle])
        #### calculate value of each particle ####
        for index, particle in enumerate(self.particles_pool):
            # saved belief
            prev_belief = particle.prior_belief if layer == 0 else particle.prev_posterior
            # belief value function
            value_space[index] = self.value_measure(particle.predict_belief,
                                                    particle.posterior_belief,
                                                    prev_belief)
            # save posterior belief
            particle.prev_posterior = particle.posterior_belief

        ##### some method to look for optimal particle factor value ####
        # identify elite particles (30% better)
        valid_values = value_space[value_space > -np.inf]
        if valid_values.size == 0:
            elite_threshold = -np.inf
            contain_elite = False
        else:
            elite_threshold = np.percentile(valid_values, 70)
            contain_elite = False  # just check this decision round
            # single decision optimization
            for i, particle in enumerate(self.particles_pool):
                # check elite particles
                if particle.feasible and value_space[i] >= elite_threshold:
                    particle.is_elite = True
                    contain_elite = True
                # update global optimum particle
                if not self.fund_best_value and particle.feasible:
                    self.best_value = value_space[i]
                    self.fund_best_value = True
                if self.fund_best_value and particle.feasible and value_space[i] >= self.best_value:
                    self.best_value = value_space[i]
                    self.best_factor_space = [particle.alpha, particle.beta, particle.gamma, particle.zeta]  # document best decision
                    self.best_particle = [cp.deepcopy(particle)]
                # check the best particle of one layer
                if particle.feasible and particle.posterior_belief > max_belief:
                    # update the best belief
                    max_belief = particle.posterior_belief

        #### log ####
        # save max posterior belief
        self.multi_belief_list.append(max_belief)
        # check the state of decision optimization
        if not contain_elite or max_belief < self.min_belief:
            self.particles_state = False
            logger.warning(f"Has something wrong with no feasible particles in layer: {layer}, "
                           f"through factor: {self.original_factor}, particles will been regenerated in next layer.")
        # adaptive adjustment based on elite particle
        self.elite_indices = [i for i, p in enumerate(self.particles_pool) if p.is_elite and p.feasible]
        if self.best_particle:
            best_particle_info = (f"{self.best_particle[0].posterior_belief}--"
                                  f"[{self.best_particle[0].alpha}, {self.best_particle[0].beta}, "
                                  f"{self.best_particle[0].gamma}, {self.best_particle[0].zeta}]--"
                                  f"{self.best_particle[0].index}")
        else:
            best_particle_info = "<<NOT FOUND>>"
        # log the information of single decision optimization layer
        logger.debug(f"layer: {layer}, "
                     f"whether elite particle subsist: {contain_elite}, "
                     f"number of elite particles: {len(self.elite_indices)}, "
                     f"the maximum belief: {max_belief}, "
                     f"the info--best particle: {best_particle_info}.")

        #### set original factor for next trying ####
        if self.elite_indices and any(value_space[self.elite_indices] > - np.inf):
            # contain the properties of elite particles
            elite_alphas = [self.particles_pool[i].alpha for i in self.elite_indices]
            elite_betas = [self.particles_pool[i].beta for i in self.elite_indices]
            elite_gammas = [self.particles_pool[i].gamma for i in self.elite_indices]
            elite_zetas = [self.particles_pool[i].zeta for i in self.elite_indices]
            weights = np.array([value_space[i] for i in self.elite_indices])
            min_weights = np.min(weights)
            adjust_weights = weights - min_weights + 1e-10
            adjust_weights = adjust_weights / np.sum(adjust_weights)
            self.original_factor = [
                np.average(elite_alphas, weights=adjust_weights),
                np.average(elite_betas, weights=adjust_weights),
                np.average(elite_gammas, weights=adjust_weights),
                np.average(elite_zetas, weights=adjust_weights)  #
            ]
            self.original_factor_std = [
                np.sqrt(
                    np.average((np.array(elite_alphas) - self.original_factor[0]) ** 2, weights = adjust_weights)),
                np.sqrt(
                    np.average((np.array(elite_betas) - self.original_factor[1]) ** 2, weights = adjust_weights)),
                np.sqrt(
                    np.average((np.array(elite_gammas) - self.original_factor[2]) ** 2, weights = adjust_weights)),
                np.sqrt(
                    np.average((np.array(elite_zetas) - self.original_factor[3]) ** 2, weights=adjust_weights))
            ]
            self.elite_weights = dict(zip(self.elite_indices.copy(), adjust_weights))
        else:
            # will determine the factor of regenerated particles, so will be inherited from last ?
            self.original_factor = [0.5, 0.05, 0.005, 0.5]
            self.original_factor_std = [0.1, 0.01, 0.001, 0.1]
            self.elite_weights = {}

        #### post measure of single layer ####
        # obsolete particles randomly
        num_elite = len(self.elite_indices)
        if num_elite > 0.8 * len(self.particles_pool):
            reset_indices = random.sample(range(len(self.particles_pool)),
                                          max(2, len(self.particles_pool) // 5))
            for i in reset_indices:
                self.particles_pool[i].is_elite = False

    def single_layer(self, layer):
        """
        Single trying to implement the decision.

        Parameters
        ----------
        layer

        Returns
        -------

        """
        self.movement(layer)  # move particles variously due to their factor
        self.predict_calculation(self.cov_matrix_exp, layer)
        self.posterior_calculation(self.cov_matrix_exp, layer)
        self.optimal_value_measure(layer)

    def multi_layers(self):
        """
        Multiple iterations trying to look for the best decision.

        Returns
        -------

        """
        self.multi_belief_list = []
        self.best_factor_space = []
        # contain optimum particle
        self.best_particle = []
        self.particles_state = True  #
        self.fund_best_value = False  # check in all layers

        ##### optimization start #####
        self.generate_particles(0)
        self.prior_calculation(self.cov_matrix_exp, 0)
        if not self.particles_state:  # maybe residual calculation errors with feasible variables
            logger.error(f"Has something wrong in prior belief calculation in {self.iter + 1}th iteration.")
        self.particles_pool_load = cp.deepcopy(self.particles_pool)
        x_temp = self.x.copy()
        # multiple trying to look for the best factor configuration
        for layer in range(self.max_layer):
            logger.debug(f"-----layer: {layer} starts-----")
            # update particles properties while layer starts
            self.update_particles(layer)  # set factors: alpha、beta、zeta
            # regenerate particles while decision iteration failing
            if not self.particles_state or (layer > 5 and max(self.multi_belief_list[-5:]) - min(self.multi_belief_list[-5:]) < self.tol):
                logger.debug(f"Restarting search at layer {layer}")
                self.particles_state = True
                self.particles_pool = cp.deepcopy(self.particles_pool_load)
                # regenerate factor bounds
                self.generate_factor_bounds()
                for index, particle in enumerate(self.particles_pool):
                    particle.index = (layer, index)
                    # redistribute factor for each particle
                    self.distribute_factor(particle)
            #### particles trying once ####
            self.single_layer(layer)
            # decision iteration done
            if layer > 0 and self.particles_state and ((self.multi_belief_list[-1] - self.multi_belief_list[-2]) > self.tol**2
                              or (self.multi_belief_list[-1] > self.multi_belief_list[-2] > self.multi_belief_list[-1]* 0.99)):
                logger.debug(f"Has reached the best decision: {self.best_factor_space} in layer: {layer + 1}.")
                break
            # decision iteration reaches the max layer
            if layer == self.max_layer - 1:
                logger.warning(f"Searching for best decision has reached max layer: {self.max_layer}, "
                               f"the decision: {self.best_factor_space} maybe the best decision.")
        #### optimization done ####
        if self.best_particle:
            self.x = self.best_particle[0].variables.copy()
            self.residual = self.best_particle[0].residual_temp.copy()
            self.residual_slipy = self.best_particle[0].residual_slipy.copy()
            self.jacobin = self.best_particle[0].jacobin_temp.copy()
            self.jacobin_slipy = self.best_particle[0].jacobin_slipy.copy()
            self.increment = self.x - x_temp
        else:
            self.x = x_temp.copy()
            self.increment = np.zeros(self.n)
            self.residual = self.particles_pool_load[0].residual_temp.copy()
            logger.warning(f"Has no optimum decision been found in iteration: {self.iter + 1}.")

    def solve(self, original_x, print_results=True):
        """
        Main function to application the algorithm.

        Parameters
        ----------
        original_x: np.array
        print_results: bool

        Returns
        -------
        variables: np.array
        increment: np.array
        residual: np.array
        converged: bool
        progress: bool

        """
        self.residual_history = []
        self.increment_history = []
        self.set_original_x(original_x)
        self.start_time = time()
        # iteration start
        if self.iterinfo:
            self.iterinfo_head(print_results)
        # loop
        for self.iter in range(self.max_iter):
            # algorithm core
            self.multi_layers()
            # iteration process
            if self.iterinfo:
                self.iterinfo_body(print_results)
            # document iterated information
            self.residual_history.append(self.residual)
            self.increment_history.append(self.increment)
            # convergence check
            if self.iter >= self.min_iter - 1 and (np.array([norm(res) for res in self.residual_history[-2:]]) < (self.tol ** 0.5)).all():
                self.converged = True  #
                break
            # progress check
            if self.iter > 40:  #
                if (
                    all(
                        np.array([norm(res) for res in self.residual_history[(self.iter - 3):]]) >= norm(self.residual_history[-3]) * 0.99
                    ) and norm(self.residual_history[-1]) >= norm(self.residual_history[-2]) * 0.99
                ):
                    self.progress = False
                    break
        self.end_time = time()
        # iteration done
        if self.iterinfo:
            self.iterinfo_tail(print_results)
        # reaching max iteration
        if self.iter == self.max_iter - 1:
            msg = (
                f"Reached maximum iteration count ({self.max_iter})), calculation stopped. "
                f"Residual value is " +
                "{:.2e}".format(norm(self.residual))
            )
            logger.warning(msg)
        # plot the increment and residual
        self.increment_history = np.array(self.increment_history)
        self.residual_history = np.array(self.residual_history)
        if self.plot_iteration:
            self.plot_iterations([self.increment_history, self.residual_history],
                                 ['increment_history', 'residual_all_history'])
        return self.x, self.increment, self.residual, self.converged, self.progress

    def iterinfo_head(self, print_results=True):
        """Print head of convergence progress."""
        # Start with defining the format here
        self.iterinfo_fmt = ' {iter:5s} | {residual:10s} | {progress:10s} '
        self.iterinfo_fmt += '| {massflow:10s} | {pressure:10s} | {enthalpy:10s} '
        self.iterinfo_fmt += '| {fluid:10s} | {component:10s} '
        # Use the format to create the first logging entry
        msg = self.iterinfo_fmt.format(
            iter='iter',
            residual='residual',
            progress='progress',
            massflow='massflow',
            pressure='pressure',
            enthalpy='enthalpy',
            fluid='fluid',
            component='component'
        )
        logger.progress(0, msg)
        msg2 = '-' * 7 + '+------------' * 7

        logger.progress(0, msg2)
        if print_results:
            print('\n' + msg + '\n' + msg2)
        return

    def iterinfo_body(self, print_results=True):
        """Print convergence progress."""
        iter_str = str(self.iter + 1)
        residual_norm = norm(self.residual)  # /self.num_vars
        residual = 'NaN'
        progress = 'NaN'
        massflow = 'NaN'
        pressure = 'NaN'
        enthalpy = 'NaN'
        fluid = 'NaN'
        component = 'NaN'

        progress_val = -1

        if not np.isnan(residual_norm):
            residual = '{:.2e}'.format(residual_norm)

            if norm(self.increment):
                massflow = '{:.2e}'.format(norm(self.increment[self.m_index]))
                pressure = '{:.2e}'.format(norm(self.increment[self.p_index]))
                enthalpy = '{:.2e}'.format(norm(self.increment[self.h_index]))
                fluid = '{:.2e}'.format(norm(self.increment[self.fl_index]))
                component = '{:.2e}'.format(norm(self.increment[self.cp_index]))

            # This should not be hardcoded here.
            if residual_norm > np.finfo(float).eps * 100:
                progress_min = math.log(self.tol)
                progress_max = math.log(self.tol ** 0.5) * -1
                progress_val = math.log(max(residual_norm, self.tol ** 0.5)) * -1
                # Scale to 0-1
                progres_scaled = (
                    (progress_val - progress_min)
                    / (progress_max - progress_min)
                )
                progress_val = max(0, min(1, progres_scaled))
                # Scale to 100%
                progress_val = int(progress_val * 100)
            else:
                progress_val = 100

            progress = '{:d} %'.format(progress_val)

        msg = self.iterinfo_fmt.format(
            iter=iter_str,
            residual=residual,
            progress=progress,
            massflow=massflow,
            pressure=pressure,
            enthalpy=enthalpy,
            fluid=fluid,
            component=component
        )
        logger.progress(progress_val, msg)
        if print_results:
            print(msg)
        return

    def iterinfo_tail(self, print_results=True):
        """Print tail of convergence progress."""
        num_iter = self.iter + 1
        clc_time = self.end_time - self.start_time
        num_ips = num_iter / clc_time if clc_time > 1e-10 else np.inf
        msg = '-' * 7 + '+------------' * 7
        logger.progress(100, msg)
        msg = (
            "Total iterations: {0:d}, Calculation time: {1:.2f} s, "
            "Iterations per second: {2:.2f}"
        ).format(num_iter, clc_time, num_ips)
        logger.debug(msg)
        if print_results:
            print(msg)
        return

    def plot_iterations(self, obj_list, titles=None, figsize=(12, 8)):
        """
        draw multiple 3D curve surface of iteration

        参数：
        obj_list (list) - 包含多个要绘制的二维数组的列表
        titles (list) - 每个子图的标题列表，长度应与obj_list一致
        figsize (tuple) - 整个画布尺寸，默认(12,8)
        """
        # input check
        if not isinstance(obj_list, (list, tuple)):
            obj_list = [obj_list]

        if titles and len(titles) != len(obj_list):
            raise ValueError("标题数量必须与对象数量一致")

        # generate plot configuration
        num_plots = len(obj_list)
        ncols = min(3, num_plots)  # 每行最多3个子图
        nrows = (num_plots + ncols - 1) // ncols

        # generate plot
        fig = plt.figure(figsize=figsize)
        axes = []
        for i in range(num_plots):
            ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')
            axes.append(ax)

        # axis parameter
        t = np.arange(obj_list[0].shape[0])  # iteration num
        n = np.arange(self.n)  # parameters index
        tt, nn = np.meshgrid(t, n, indexing='ij')

        # draw plots
        for idx, (ax, obj) in enumerate(zip(axes, obj_list)):
            # check dimension
            if obj.shape != (len(t), len(n)):
                raise ValueError(f"对象{idx}维度不匹配，应为({len(t)}, {len(n)})")

            # draw curve surface
            surf = ax.plot_surface(tt, nn, obj,
                                   cmap=plt.cm.viridis,
                                   rstride=1, cstride=1,
                                   linewidth=0, antialiased=True)

            # add color strip
            fig.colorbar(surf, ax=ax, shrink=0.6, label='Value')

            # set label
            ax.set(xlabel='Iteration Step',
                   ylabel='Variable Index',
                   zlabel='Value Magnitude')

            # add title
            if titles:
                ax.set_title(titles[idx], pad=15)

        plt.tight_layout()
        plt.show()
    