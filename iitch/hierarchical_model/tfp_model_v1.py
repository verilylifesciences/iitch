# Copyright 2019 Verily Life Sciences LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implementation of a model of mosquito dispersion, growth, and death.

We are looking to estimate a number of parameters relating to how released
wolbachia infected mosquitoes move through the environment and inhibit
population growth. We observe male and female trap numbers, and thus will look
to infer the contribution of wolbachia infected males.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

# Collection of the observed data used to define the model
# n = number of traps
# z = number of unique spatial locations
# t = number of time steps
ObservedData = collections.namedtuple(
    'ObservedData',
    [
        # np array ~ [n] - Unique indices for each trap location
        'trap_indices',
        # np array ~ [n] - Duration the trap was placed
        'trap_durations',
        # np array ~ [n] - Number of male mosquitoes captured
        'm_counts',
        # np array ~ [n] - Number of female mosquitoes captured
        'f_counts',
        # np array ~ [n] - 0/1 indicating absence/presence of bait in trap
        'baited_traps',
        # np array ~ [n] - Trap location latitude
        'trap_lat',
        # np array ~ [n] - Trap location longitude
        'trap_lng',
        # np array ~ [n] - Day trap was collected
        'trap_collection_day',
        # np array ~ [t, z] -  Tensor describing release events over space and
        # time
        'release_tensor',
        # list of tuples (t, idx) where traps were placed
        'obs_locations',
        # total number of locations in a single time step
        'num_locations',
        # number of time steps
        'time_steps',
        # pairwise squared distance matrix between midpoints of partitions in
        # space
        'squared_dist_matrix',
        # pandas range interval that defines the partitioning with respect to
        # lat
        'lat_seq',
        # pandas range interval that defines the partitioning with respect to
        # lng
        'lng_seq',
        # pandas range interval that defines the partitioning with respect to
        # time
        'time_seq'
    ])

ModelPriorParameters = collections.namedtuple(
    'ModelPriorParameters',
    [
        # ~ Beta
        'baited_male_trapping_efficiency',
        # ~ Beta
        'unbaited_male_trapping_efficiency',
        # ~ Beta
        'baited_female_trapping_efficiency',
        # ~ Beta
        'unbaited_female_trapping_efficiency',
        # Log dispersion coefficients that dictate relative attractivity of
        # points in space ~ LogMultivariateNormal
        'log_dispersion_coeffs',
        # Log wildtype male population initial conditions
        # ~ LogMultivariateNormal
        'log_init_male_wt_intensity',
        # Log wildtype female population initial conditions
        # ~ LogMultivariateNormal
        'log_init_female_intensity',
        # Average number of mosquitoes produced by a female per time step
        # ~ Gamma
        'growth_rate',
        # Mating competitiveness index
        # ~ Gamma
        'frieds_index',
        # Average proportion of wildtype mosquitoes that survive per time
        # step ~ Unif
        'wt_survival_rate',
        # Average proportion of sterile infected mosquitoes that survive per
        # time step ~ Unif
        'sterile_survival_rate'
    ])

ModelParameters = collections.namedtuple('ModelParameters', [
    'male_abundances', 'female_abundances', 'baited_male_trapping_efficiency',
    'unbaited_male_trapping_efficiency', 'baited_female_trapping_efficiency',
    'unbaited_female_trapping_efficiency', 'dispersion_coeffs',
    'init_male_wt_intensity', 'init_female_intensity', 'growth_rate',
    'frieds_index', 'wt_survival_rate', 'sterile_survival_rate'
])


def bounded_bijection(lower, upper, validate_args, name=None):
  """Creates bijector that bounds data between an upper and lower bound.

  Arguments:
    lower: lower bound of the bijection
    upper: upper bound of the bijection
    validate_args: validate the values passed into the bijector
    name: name of bijector

  Returns:
    A bijector
  """
  return tfb.Chain([
      tfb.AffineScalar(shift=lower),
      tfb.AffineScalar(scale=upper - lower),
      tfb.Sigmoid()
  ],
                   name=name,
                   validate_args=validate_args)


class MosquitoTrapModelv1(object):
  """Implementation of mosquito dispersion, growth, death model.

  Given the observed data and prior parameters, constructs a likelihood that
  may then be used in sampling routines, optimization, etc. The primary function
  of interest is the joint_log_prob function which takes as arguments the
  unknown parameters of the model.

  """

  def __init__(self,
               obs_data,
               prior_parameters,
               validate_args=False):
    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
      self.validate_args = validate_args
      self.dtype = np.float32
      self.obs_data = obs_data
      self.prior_parameters = prior_parameters
      self.make_prior()
      self.make_unconstraining_bijectors()
      self.joint_log_prob = tf.make_template(
          name_='joint_log_prob', func_=self._joint_log_prob)

  def make_prior(self):
    """Generates the joint prior distribution of the model."""
    self.prior_dist = tfd.JointDistributionNamed(
        dict(
            baited_male_trapping_efficiency=tfd.Beta(
                concentration1=self.prior_parameters
                .baited_male_trapping_efficiency['concentration1'],
                concentration0=self.prior_parameters
                .baited_male_trapping_efficiency['concentration0'],
                validate_args=self.validate_args,
                name='rv_baited_male_trapping_efficiency'),
            unbaited_male_trapping_efficiency=tfd.Beta(
                concentration1=self.prior_parameters
                .unbaited_male_trapping_efficiency['concentration1'],
                concentration0=self.prior_parameters
                .unbaited_male_trapping_efficiency['concentration0'],
                validate_args=self.validate_args,
                name='rv_unbaited_male_trapping_efficiency'),
            baited_female_trapping_efficiency=tfd.Beta(
                concentration1=self.prior_parameters
                .baited_female_trapping_efficiency['concentration1'],
                concentration0=self.prior_parameters
                .baited_female_trapping_efficiency['concentration0'],
                validate_args=self.validate_args,
                name='rv_baited_female_trapping_efficiency'),
            unbaited_female_trapping_efficiency=tfd.Beta(
                concentration1=self.prior_parameters
                .unbaited_female_trapping_efficiency['concentration1'],
                concentration0=self.prior_parameters
                .unbaited_female_trapping_efficiency['concentration0'],
                validate_args=self.validate_args,
                name='rv_unbaited_female_trapping_efficiency'),
            log_dispersion_coeffs=tfd.MultivariateNormalFullCovariance(
                loc=self.prior_parameters.log_dispersion_coeffs['loc'],
                covariance_matrix=self.prior_parameters
                .log_dispersion_coeffs['covariance_matrix'],
                validate_args=self.validate_args,
                name='rv_log_dispersion_coeffs',
            ),
            log_init_male_wt_intensity=tfd.MultivariateNormalFullCovariance(
                loc=self.prior_parameters.log_init_male_wt_intensity['loc'],
                covariance_matrix=self.prior_parameters
                .log_init_male_wt_intensity['covariance_matrix'],
                validate_args=self.validate_args,
                name='rv_init_male_wt_log_intensity'),
            log_init_female_intensity=tfd.MultivariateNormalFullCovariance(
                loc=self.prior_parameters.log_init_female_intensity['loc'],
                covariance_matrix=self.prior_parameters
                .log_init_female_intensity['covariance_matrix'],
                validate_args=self.validate_args,
                name='rv_init_female_log_intensity'),
            growth_rate=tfd.Gamma(
                concentration=self.prior_parameters
                .growth_rate['concentration'],
                rate=self.prior_parameters.growth_rate['rate'],
                validate_args=self.validate_args,
                name='rv_growth_rate'),
            frieds_index=tfd.Gamma(
                concentration=self.prior_parameters
                .frieds_index['concentration'],
                rate=self.prior_parameters.frieds_index['rate'],
                validate_args=self.validate_args,
                name='rv_frieds_index'),
            wt_survival_rate=tfd.Uniform(
                low=self.prior_parameters.wt_survival_rate['low'],
                high=self.prior_parameters.wt_survival_rate['high'],
                validate_args=self.validate_args,
                name='rv_wt_survival_rate'),
            sterile_survival_rate=tfd.Uniform(
                low=self.prior_parameters.sterile_survival_rate['low'],
                high=self.prior_parameters.sterile_survival_rate['high'],
                validate_args=self.validate_args,
                name='rv_sterile_survival_rate')))

  def _generate_dispersion_matrix(self, dispersion_coeffs):
    """Generates a dispersion matrix using the dispersion coefficients.

    The dispersion matrix is a right stochastic matrix so that multiplying by
    it results in redistribution of the current population.

    Arguments:
      dispersion_coeffs: Coefficients that indicate a degree of attractivity
        of a location

    Returns:
      A dispersion matrix that redistributes weights of a vector it is
      multiplied by
    """
    squared_dist_matrix = self.obs_data.squared_dist_matrix
    num_locations = self.obs_data.num_locations
    divisor = tf.reshape(
        tf.tile(dispersion_coeffs, tf.constant([num_locations])),
        (num_locations, num_locations))
    disp_matrix = tf.exp(-(squared_dist_matrix / divisor))
    row_sums = tf.reshape(tf.reduce_sum(disp_matrix, axis=1), (-1, 1))
    disp_matrix = tf.divide(disp_matrix, row_sums)
    return disp_matrix

  def _generate_growth_matrix(self, female_intensity, male_s_intensity,
                              male_wt_intensity, frieds_index, growth_rate):
    """Generates a growth matrix.

    The growth matrix calculates the effective sterile population which is then
    used to calculate the probability of a productive mating in a given
    spatial subunit. This is then multiplied by the proportion of the female
    population and the growth rate to result in a growth matrix.

    Arguments:
      female_intensity: vector of female intensities
      male_s_intensity: vector of sterile male intensities
      male_wt_intensity: vector of wildtype male intensities
      frieds_index: mating competitiveness constant
      growth_rate: number of mosquitoes produced by a female per time step

    Returns:
      A matrix of growth values by which the current population is
      multiplied by.
    """
    num_locations = self.obs_data.num_locations
    effective_sterile_pop = frieds_index * male_s_intensity
    wt_mating_probabilities = tf.divide(
        male_wt_intensity, male_wt_intensity + effective_sterile_pop)
    wt_mated = female_intensity * wt_mating_probabilities
    growth_values = (growth_rate * wt_mated) / female_intensity
    growth_matrix = tf.eye(num_locations) * growth_values
    return growth_matrix

  def _generate_wt_survival_matrix(self, wt_survival_rate):
    """Generates a survival matrix.

    Arguments:
      wt_survival_rate: The proportion of the wildtype population that survives
      to the next time step.

    Returns:
      An identity matrix scaled by the survival rate
    """
    return tf.eye(self.obs_data.num_locations) * wt_survival_rate

  def _generate_sterile_survival_matrix(self, sterile_survival_rate):
    """Generates a survival matrix.

    Arguments:
      sterile_survival_rate: The proportion of the population that survives to
      the next time step.

    Returns:
      An identity matrix scaled by the survival rate
    """
    return tf.eye(self.obs_data.num_locations) * sterile_survival_rate

  def _calculate_wt_intensity_evolution(self, intensity, disp_matrix,
                                        growth_matrix, survival_matrix):
    """Calculates the evolution of the wildtype intensity vector.

    Arguments:
      intensity: Vector of current intensities
      disp_matrix: Dispersion matrix defined above
      growth_matrix: Growth matrix defined above
      survival_matrix: Survival matrix defined above

    Returns:
      The resulting intensity from a single time step
    """
    next_intensity = tf.matmul(
        disp_matrix,
        tf.matmul(growth_matrix, tf.matmul(survival_matrix, intensity)))
    return next_intensity

  def _calculate_sterile_intensity_evolution(self, intensity, release,
                                             disp_matrix, survival_matrix):
    """Calculates the evolution of the sterile intensity vector.

    Arguments:
      intensity: Vector of current intensities
      release: A matrix of releases that occurred in the time period
      disp_matrix: Dispersion matrix defined above
      survival_matrix: Survival matrix defined above

    Returns:
      The resulting intensity from a single time step
    """
    next_intensity = tf.matmul(disp_matrix, tf.matmul(survival_matrix,
                                                      intensity)) + release
    return next_intensity

  def _make_intensity_process(self, dispersion_coeffs, wt_survival_rate,
                              sterile_survival_rate, init_male_wt_intensity,
                              init_female_intensity, frieds_index, growth_rate):
    """Function to apply the process from t_0 to t_n as defined by the model.

    Arguments:
      dispersion_coeffs: Dispersion coefficients of the model
      wt_survival_rate: Wildtype population survival rate
      sterile_survival_rate: Sterile population survival rate
      init_male_wt_intensity: Male wildtype population intensity initial
        conditions
      init_female_intensity: Female wildtype population intensity initial
        conditions
      frieds_index: Mating competitiveness constant
      growth_rate: Average number of mosquitoes produced by a female per time
        step

    Returns:
      The resulting male and female intensity evolutions
    """
    num_locations = self.obs_data.num_locations

    disp_matrix = self._generate_dispersion_matrix(dispersion_coeffs)
    wt_survival_matrix = self._generate_wt_survival_matrix(wt_survival_rate)
    sterile_survival_matrix = self._generate_sterile_survival_matrix(
        sterile_survival_rate)

    male_wt_intensity = [tf.reshape(init_male_wt_intensity, [num_locations, 1])]
    male_s_intensity = [
        tf.reshape(self.obs_data.release_tensor[0], [num_locations, 1])
    ]
    female_intensity = [tf.reshape(init_female_intensity, [num_locations, 1])]

    for t in range(self.obs_data.time_steps - 1):
      release = self.obs_data.release_tensor[t]

      growth_matrix = self._generate_growth_matrix(female_intensity[-1],
                                                   male_s_intensity[-1],
                                                   male_wt_intensity[-1],
                                                   frieds_index, growth_rate)

      mwt_intensity_step = self._calculate_wt_intensity_evolution(
          male_wt_intensity[-1], disp_matrix, growth_matrix, wt_survival_matrix)

      male_wt_intensity.append(mwt_intensity_step)

      f_intensity_step = self._calculate_wt_intensity_evolution(
          female_intensity[-1], disp_matrix, growth_matrix, wt_survival_matrix)

      female_intensity.append(f_intensity_step)

      ms_intensity_step = self._calculate_sterile_intensity_evolution(
          male_s_intensity[-1], release, disp_matrix, sterile_survival_matrix)

      male_s_intensity.append(ms_intensity_step)

    male_intensity = (
        tf.stack(male_wt_intensity, axis=0) +
        tf.stack(male_s_intensity, axis=0))

    female_intensity = tf.stack(female_intensity, axis=0)
    return (male_intensity, female_intensity)

  def _make_process_likelihood(self, dispersion_coeffs, wt_survival_rate,
                               sterile_survival_rate, init_male_wt_intensity,
                               init_female_intensity, frieds_index,
                               growth_rate):
    """Make the likelihood of the abundances given the process.

    Arguments:
      dispersion_coeffs: Dispersion coefficients
      wt_survival_rate: Wildtype survival rate
      sterile_survival_rate: Sterile survival rate
      init_male_wt_intensity: Initial male wildtype intensity conditions
      init_female_intensity: Initial female wildtype intensity conditions
      frieds_index: Mating competitiveness constant
      growth_rate: Average number of mosquitoes produced by a female per time
        step

    Returns:
      distributions conditioned on the input parameters

    """
    male_intensity, female_intensity = self._make_intensity_process(
        dispersion_coeffs, wt_survival_rate, sterile_survival_rate,
        init_male_wt_intensity, init_female_intensity, frieds_index,
        growth_rate)
    rv_male_abundance = tfd.Poisson(
        rate=tf.squeeze(
            tf.gather_nd(male_intensity, self.obs_data.obs_locations)))
    rv_female_abundance = tfd.Poisson(
        rate=tf.squeeze(
            tf.gather_nd(female_intensity, self.obs_data.obs_locations)))
    return (rv_male_abundance, rv_female_abundance)

  def _make_obs_likelihood(self, baited_male_trapping_efficiency,
                           unbaited_male_trapping_efficiency,
                           baited_female_trapping_efficiency,
                           unbaited_female_trapping_efficiency, male_abundances,
                           female_abundances):
    """Make the likelihood of the observations given the abundances.

    Arguments:
      baited_male_trapping_efficiency: Proportion of male abundance that is
        captured by a baited trap per day
      unbaited_male_trapping_efficiency: Proportion of male abundance that is
        captured by a unbaited trap per day
      baited_female_trapping_efficiency: Proportion of female abundance that is
        captured by a baited trap per day
      unbaited_female_trapping_efficiency: Proportion of female abundance that
        is captured by a unbaited trap per day
      male_abundances: Vector of true male abundances at observed locations
      female_abundances: Vector of true female abundances at observed locations
    Returns:
      distributions conditioned on the abundances
    """
    male_trapping_efficiency_vec = (
        baited_male_trapping_efficiency * self.obs_data.baited_traps +
        unbaited_male_trapping_efficiency * (1. - self.obs_data.baited_traps))

    female_trapping_efficiency_vec = (
        baited_female_trapping_efficiency * self.obs_data.baited_traps +
        unbaited_female_trapping_efficiency * (1. - self.obs_data.baited_traps))

    male_obs_abundance = tf.squeeze(
        tf.gather(male_abundances, self.obs_data.trap_indices))

    female_obs_abundance = tf.squeeze(
        tf.gather(female_abundances, self.obs_data.trap_indices))

    rv_male_observations = tfd.Binomial(
        male_obs_abundance,
        probs=self.obs_data.trap_durations * male_trapping_efficiency_vec)

    rv_female_observations = tfd.Binomial(
        female_obs_abundance,
        probs=self.obs_data.trap_durations * female_trapping_efficiency_vec)

    return (rv_male_observations, rv_female_observations)

  def _joint_log_prob(self, parameters_tensor):
    """Calculates the joint log probability of the input parameters.

    Primary function used for inference

    Arguments:
      parameters_tensor: A 1D tensor representing all the parameters to be input

    Returns:
      the log probability of the parameters under the model and observed data
    """
    (male_abundances, female_abundances, baited_male_trapping_efficiency,
     unbaited_male_trapping_efficiency, baited_female_trapping_efficiency,
     unbaited_female_trapping_efficiency, dispersion_coeffs,
     init_male_wt_intensity, init_female_intensity, growth_rate, frieds_index,
     wt_survival_rate, sterile_survival_rate
    ) = self.convert_tensor_to_parameters(parameters_tensor)

    rv_male_abundance, rv_female_abundance = self._make_process_likelihood(
        dispersion_coeffs, wt_survival_rate, sterile_survival_rate,
        init_male_wt_intensity, init_female_intensity, frieds_index,
        growth_rate)

    rv_male_observations, rv_female_observations = self._make_obs_likelihood(
        baited_male_trapping_efficiency, unbaited_male_trapping_efficiency,
        baited_female_trapping_efficiency, unbaited_female_trapping_efficiency,
        male_abundances, female_abundances)

    log_prob_parts = [
        self.prior_dist.log_prob({
            'baited_male_trapping_efficiency':
                baited_male_trapping_efficiency,
            'unbaited_male_trapping_efficiency':
                unbaited_male_trapping_efficiency,
            'baited_female_trapping_efficiency':
                baited_female_trapping_efficiency,
            'unbaited_female_trapping_efficiency':
                unbaited_female_trapping_efficiency,
            'dispersion_coeffs':
                dispersion_coeffs,
            'init_male_wt_intensity':
                init_male_wt_intensity,
            'init_female_intensity':
                init_female_intensity,
            'growth_rate':
                growth_rate,
            'frieds_index':
                frieds_index,
            'wt_survival_rate':
                wt_survival_rate,
            'sterile_survival_rate':
                sterile_survival_rate
        })
    ]

    log_prob_parts.extend([
        tf.reshape(
            rv_male_abundance.log_prob(tf.transpose(male_abundances)), [-1]),
        tf.reshape(
            rv_female_abundance.log_prob(tf.transpose(female_abundances)), [-1])
    ])

    log_prob_parts.extend([
        rv_male_observations.log_prob(self.obs_data.m_counts),
        rv_female_observations.log_prob(self.obs_data.f_counts)
    ])

    sum_log_prob = tf.reduce_sum(tf.concat(log_prob_parts, axis=-1), axis=-1)
    return sum_log_prob

  def convert_tensor_to_parameters(self, parameter_tensor):
    """Convert the input parameter tensor into easier to use parameters.

    Arguments:
      parameter_tensor: Parameters collapsed into a 1D tensor

    Returns:
      The unpacked parameter_tensor into a ModelParameters object
    """
    num_observations = len(self.obs_data.obs_locations)
    offset = 0
    male_abundances = parameter_tensor[offset:offset + num_observations]
    offset += num_observations
    female_abundances = parameter_tensor[offset:offset + num_observations]
    offset += num_observations
    baited_male_trapping_efficiency = parameter_tensor[offset:offset + 1]
    offset += 1
    unbaited_male_trapping_efficiency = parameter_tensor[offset:offset + 1]
    offset += 1
    baited_female_trapping_efficiency = parameter_tensor[offset:offset + 1]
    offset += 1
    unbaited_female_trapping_efficiency = parameter_tensor[offset:offset + 1]
    offset += 1
    dispersion_coeffs = parameter_tensor[offset:offset +
                                         self.obs_data.num_locations]
    offset += self.obs_data.num_locations
    init_male_wt_intensity = parameter_tensor[offset:offset +
                                              self.obs_data.num_locations]
    offset += self.obs_data.num_locations
    init_female_intensity = parameter_tensor[offset:offset +
                                             self.obs_data.num_locations]
    offset += self.obs_data.num_locations
    growth_rate = parameter_tensor[offset:offset + 1]
    offset += 1
    frieds_index = parameter_tensor[offset:offset + 1]
    offset += 1
    wt_survival_rate = parameter_tensor[offset:offset + 1]
    offset += 1
    sterile_survival_rate = parameter_tensor[offset:offset + 1]
    offset += 1
    return ModelParameters(
        male_abundances, female_abundances, baited_male_trapping_efficiency,
        unbaited_male_trapping_efficiency, baited_female_trapping_efficiency,
        unbaited_female_trapping_efficiency, dispersion_coeffs,
        init_male_wt_intensity, init_female_intensity, growth_rate,
        frieds_index, wt_survival_rate, sterile_survival_rate)

  def convert_parameters_to_tensor(self, model_parameters):
    """Flatten model parameters into a 1D tensor.

    Arguments:
      model_parameters: ModelParameters object

    Returns:
      a 1D tensor of the model parameters
    """
    return tf.concat(model_parameters, axis=-1)

  def make_unconstraining_bijectors(self):
    """Generates bijectors to constrain sampling.

    Monte Carlo sampling efficiency can be increased by model reparameterization
    that results in decorrelation of variables and rescaling to a similar scale.
    Using bijectors can achieve a similar result. Here we implement a number
    of bijections to restrict sampling to "reasonable" values (i.e. positive or
    restricted scales)
    """
    trapping_efficiency_upper_limit = self.dtype(
        1. /
        np.max(self.obs_data.trap_durations))  # longest time trap is set out
    mean_count = self.dtype(
        np.mean(self.obs_data.m_counts + self.obs_data.f_counts))
    upper_count_limit = self.dtype(mean_count * 1e6)
    max_count = self.dtype(
        np.max(self.obs_data.m_counts + self.obs_data.f_counts))
    self.unconstraining_bijectors = []

    num_observations = len(self.obs_data.obs_locations)

    # male abundances bijections
    for _ in range(num_observations):
      self.unconstraining_bijectors.append(tfb.Exp())

    # female abundances bijections
    for _ in range(num_observations):
      self.unconstraining_bijectors.append(tfb.Exp())

    # baited male trapping efficiency bijection
    self.unconstraining_bijectors.append(
        bounded_bijection(
            self.dtype(0.000001), trapping_efficiency_upper_limit,
            self.validate_args))

    # unbaited male trapping efficiency bijetion
    self.unconstraining_bijectors.append(
        bounded_bijection(
            self.dtype(0.000001), trapping_efficiency_upper_limit,
            self.validate_args))

    # baited female trapping efficiency bijection
    self.unconstraining_bijectors.append(
        bounded_bijection(
            self.dtype(0.000001), trapping_efficiency_upper_limit,
            self.validate_args))

    # unbaited female trapping efficiency bijection
    self.unconstraining_bijectors.append(
        bounded_bijection(
            self.dtype(0.000001), trapping_efficiency_upper_limit,
            self.validate_args))

    # dispersion coeffs bijection
    for _ in range(self.obs_data.num_locations):
      self.unconstraining_bijectors.append(
          tfb.Chain(
              (tfb.AffineScalar(shift=self.dtype(1.),
                                scale=self.dtype(6.)), tfb.Exp())))

    # init male wt intensity bijection
    for _ in range(self.obs_data.num_locations):
      self.unconstraining_bijectors.append(
          tfb.Chain((tfb.AffineScalar(
              shift=self.dtype(max_count),
              scale=self.dtype(upper_count_limit)), tfb.Exp())))

    # init female intensity bijection
    for _ in range(self.obs_data.num_locations):
      self.unconstraining_bijectors.append(
          tfb.Chain((tfb.AffineScalar(
              shift=self.dtype(max_count),
              scale=self.dtype(upper_count_limit)), tfb.Exp())))

    # growth rate bijection
    self.unconstraining_bijectors.append(
        tfb.Chain(
            (tfb.AffineScalar(shift=self.dtype(2.),
                              scale=self.dtype(100.)), tfb.Exp())))

    # frieds index bijection
    self.unconstraining_bijectors.append(
        bounded_bijection(
            self.dtype(0.00001), self.dtype(.99), self.validate_args))

    # wt survival rate bijection
    self.unconstraining_bijectors.append(
        bounded_bijection(
            self.dtype(0.001), self.dtype(.99), self.validate_args))

    # sterile survival rate bijection
    self.unconstraining_bijectors.append(
        bounded_bijection(
            self.dtype(0.001), self.dtype(.99), self.validate_args))
