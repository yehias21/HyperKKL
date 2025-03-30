# HyperKKL

## Model
The aim of HyperKKL is to take advantage of information provided in the model:
- **Temporal aspect of the data itself:** data is trained in random sample fashion (aka assume no correlation between the points) mathematically presentation as 
P(xt+1|xt ) = 0
- **Online error**: output error produced during inference: y_hat-y error signal
- **Input signal (u)** 
- **Training and sampling techniques**: Employing curriculum learning and autoregressive (teacher forcing) has been shown to enhance the model accuracy.

### Extended Idea

- We want to generalize for any kind of Input, Idea proposed:
  - If input is very complex in nature to be learned, take the spectral decomposition of it (e.g Fourier basis)
- The problem with the input is in the encoder and not the decoder, as the decoder maps from Z space to X space. The encoder is the one that needs to be more complex (have a dynamic weights), because it deals with the input.
- Use the differential equation as a hard constraint not soft constraint --> Can I prove that the hypernetwork guarantee such thing?
### Extended Obstacles and solutions
- **Problem:** What about the sampling of the signals, different sampling rates, will need retraining. \
**Solution:** Fix a maximum sampling, as in reality the sampling rate of the sensor/oscillator is physically limited.
### Baseline



## Data generation and creation

### Data Generation 

- The goal is to generate a dataset, a mapping between the system_state and the observer_state (X, Z)
- These points are sampled from multiple trajectories of the system, note that each trajectory represents a new distribution of 


### Code Design
## Evaluation technique
### Uncertainty Quantification
### Hypothesis testing - Permutation test - Ablation study
## Implementation Timeline
### Steps
- [X] Generate the data as proposed
- [X] Develop the Hyper-network model
- [ ] Finish the training loop
- [ ] Simulation
- [ ] Evaluation
- [ ] Iteration
### Refinement 
- [ ] Test code for each code part + visualization technique
- [ ] Refactoring