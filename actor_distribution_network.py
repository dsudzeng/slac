from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin.tf
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.distributions import tanh_bijector_stable
from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.utils import common

class AffineScalar(tfp.bijectors.Bijector):
    def __init__(self, shift=None, scale=None, validate_args=False, name="affine_scalar"):
        # Set default values for shift and scale
        self.shift = shift if shift is not None else 0.0
        self.scale = scale if scale is not None else 1.0
        super(AffineScalar, self).__init__(
            forward_min_event_ndims=0,
            validate_args=validate_args,
            name=name
        )

    def _forward(self, x):
        return self.scale * x + self.shift

    def _inverse(self, y):
        return (y - self.shift) / self.scale

    def _forward_log_det_jacobian(self, x):
        return tf.math.log(tf.abs(self.scale))

    def _inverse_log_det_jacobian(self, y):
        return -tf.math.log(tf.abs(self.scale))



class SquashToSpecDistribution(tfp.distributions.TransformedDistribution):
  def __init__(self, distribution, spec, name="SquashToSpecDistribution"):
    self.action_means, self.action_magnitudes = common.spec_means_and_magnitudes(
        spec)
    bijectors = [
        AffineScalar(
            shift=self.action_means, scale=self.action_magnitudes),
        tanh_bijector_stable.Tanh()
    ]
    bijector_chain = tfp.bijectors.Chain(bijectors)
    super(SquashToSpecDistribution, self).__init__(
        distribution, bijector_chain, name=name)

  def _mode(self):
    mode = self.distribution.mode()
    mode = self.bijector.forward(mode)
    return mode


@gin.configurable
class ActorDistributionNetwork(network.DistributionNetwork):
  """Creates an actor network."""

  def __init__(self,
               input_tensor_spec,
               output_tensor_spec,
               fc_layer_params=(256, 256),
               conv_layer_params=None,
               activation_fn=tf.keras.activations.relu,
               name='ActorDistributionNetwork'):
    """Creates an instance of `ActorDistributionNetwork`.

    Args:
      input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
        input.
      output_tensor_spec: A nest of `tensor_spec.BoundedTensorSpec` representing
        the output.
      fc_layer_params: Optional list of fully_connected parameters, where each
        item is the number of units in the layer.
      conv_layer_params: Optional list of convolution layers parameters, where
        each item is a length-three tuple indicating (filters, kernel_size,
        stride).
      activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu, ...
      name: A string representing name of the network.

    Raises:
      ValueError: If `input_tensor_spec` or `output_tensor_spec` contains more
        than one spec.
    """
    super(ActorDistributionNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=(),
        output_spec=output_tensor_spec,
        name=name)

    if len(tf.nest.flatten(input_tensor_spec)) > 1:
      raise ValueError('Only a single observation is supported by this network')

    flat_action_spec = tf.nest.flatten(output_tensor_spec)
    if len(flat_action_spec) > 1:
      raise ValueError('Only a single action is supported by this network')
    self._single_action_spec = flat_action_spec[0]

    # TODO(kbanoop): Replace mlp_layers with encoding networks.
    self._mlp_layers = utils.mlp_layers(
        conv_layer_params,
        fc_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=tf.compat.v1.keras.initializers.glorot_uniform(),
        name='input_mlp')

    self._mlp_layers.append(
        tf.keras.layers.Dense(
            2 * self._single_action_spec.shape.num_elements(),
            activation=None,
            kernel_initializer=tf.compat.v1.keras.initializers.glorot_uniform(),
            name='normal_projection_layer'))

  def call(self, observation, step_type=(), network_state=()):
    del step_type  # unused.
    output = tf.cast(tf.nest.flatten(observation)[0], tf.float32)
    for layer in self._mlp_layers:
      output = layer(output)

    shift, log_scale_diag = tf.split(output, 2, axis=-1)
    log_scale_diag = tf.clip_by_value(log_scale_diag, -20, 2)

    base_distribution = tfp.distributions.MultivariateNormalDiag(
        loc=shift,  scale_diag=tf.exp(log_scale_diag))
    distribution = SquashToSpecDistribution(
        base_distribution, self._single_action_spec)

    distribution = tf.nest.pack_sequence_as(self.output_spec, [distribution])
    return distribution, network_state
