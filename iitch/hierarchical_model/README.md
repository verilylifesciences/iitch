# iitch Hierarchical Model

This directory contains implementations of a model of mosquito dispersion, growth, and death run on [Tensorflow Probability](https://www.tensorflow.org/probability).

We are looking to estimate a number of parameters relating to how released
wolbachia infected mosquitoes move through the environment and inhibit
population growth. We observe male and female trap numbers, and thus will look
to infer the contribution of wolbachia infected males.

## Model

### Priors

<!-- GitHub can't display Latex and g3docs, can't pull in images from the repo.
So keep two versions of the equations.
-->


![Priors](doc_images/priors.png)


<!-- 
$$ \begin{align*} 
\textrm{(Trapping Efficiency)} \theta^Y &\sim Beta(\alpha_{\theta^Y}, \beta_{\theta^Y}) \\
\textrm{(Initial intesity)} \log{\lambda_0} &\sim N(\sigma_\lambda, \Sigma_\lambda) \\
\textrm{(Dispersion Coefficients)} \log{\tau} &\sim N(\sigma_\tau, \Sigma_\tau) \\
\textrm{(Growth Rate)} \eta &\sim N(\sigma_\eta, \sigma_\nu^2) \\
\textrm{(Mating Competitiveness)} c &\sim Gamma(\alpha_c, \beta_c) \\
\textrm{(Survival Rate)} \theta^D &\sim Unif(\alpha_{\theta^D}, \beta_{\theta^D})
\end{align*}$$ 
 -->


### Process Model


![Process Model](doc_images/process_model.png)


<!-- 
$$ \begin{align*}
\lambda_{t,m_{wt}} | \lambda_{t-1,m_{wt'}},\lambda_{t-1,f} &= B(\tau)G(\lambda_{t-1};\theta^G)D(\theta^D)\lambda_{t-1,m_{wt}} \\
\lambda_{t,f} | \lambda_{t-1,m_{wt'}},\lambda_{t-1,f} &= B(\tau)G(\lambda_{t-1};\theta^G)D(\theta^D)\lambda_{t-1,f} \\
\lambda_{t,m_s} | \lambda_{t-1,m_s} &= B(\tau)D(\theta^D)\lambda_{t-1,m_s} + R(t) \\
B_{i,j}(\tau) &\sim \exp(-\frac{d_{i,j}^2}{\tau(s_i)}) \textrm{(Right Stochastic Matrix)} \\
G_{i,j}(\lambda_{t,m_{wt}}, \lambda_{t,f};\eta, c) &= \eta(\lambda_{t,f}\frac{\lambda_{t,m_{wt}}}{\lambda_{t,m_{wt}} + c\lambda_{t,m_s}}) \\
D_{i,i}(\theta^D) &= \mathbb{I}\theta^D \\
R(t) &= \textrm{Release matrix at time t} \\
\end{align*}$$
 -->

### Obervation model


![Observation Model](doc_images/observation_model.png)


<!-- 
$$ \begin{align*}
Z_{t,m} | Y_{t,m},\theta_m^Y &\sim Bin(Y_{t,m}, T_t\theta_m^Y) \\
Z_{t,f} | Y_{t,f},\theta_f^Y &\sim Bin(Y_{t,f}, T_t\theta_f^Y) \\
Y_{t,m} | \lambda_{t,m_{wt}}, \lambda_{t,m_s} &\sim Poi(H(\lambda_{t,m_{wt}} + \lambda_{t,m_s})) \\
Y_{t,f} | \lambda_{t,f} &\sim Poi(H\lambda_{t,f}) \\
\end{align*}$$
 -->

## Example Usage

Here's an example of how to fit some trapping data with this model using [Hamiltonian Monte Carlo](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo). Please see the code for the details about the model parameters and data format.

```
from iitch.hierarchical_model import MosquitoTrapModelv2
import tensorflow as tf
import tensorflow_probability as tfp

...

model = MosquitoTrapModelv2(observed_trapping_data, priors)

kernel = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=model.joint_log_prob,
    step_size=0.02,
    num_leapfrog_steps=3)
    
states, kernel_results = tfp.mcmc.sample_chain(
  num_results=10000,
  num_burnin_steps=5000,
  current_state=model.convert_parameters_to_tensor(priors),
  kernel=kernel)
```

Please see the TF Probability [MCMC documentation](https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/sample_chain) about how to interpret the output of the sampling.
