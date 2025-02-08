import jax
import jax.numpy as jnp

def get_sineGaussian(t: Array, theta: Array):
  """
  Generate single sine Gaussian wavelet in time domain.

  theta = [t_0, f_0, Q, A, phi_0]
  where t_0: central time of wavelet
  f_0: frequency at t=t_0
  Q: wavelet quality factor
  A: wavelet amplitude
  phi_0: wavelet phase at t=t_0

  Returns:
  --------
    h0 (array): strain
  """
  # Define parameters
  t_0, f_0, Q, A, phi_0 = theta
  tau = Q/2*jnp.pi*f_0
  h0 = A*jnp.exp((t-t_0)^2/tau^2)*jnp.cos(2*jnp.pi*f_0*(t-t_0)+phi_0)
  return h0
  
