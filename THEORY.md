# Mathematical 
## Hypernetworks
- Use the hypernetwork as hard physics constraint? theoretical guarantee? (e.g most probably as an approximation)
### Why does it work?
- The concept of soft weight sharing 
- Our model is data conditioned, It's very related to multiple concepts as Meta-Learning, conditional neural networks
- sharing weights
## Physics points
- Why do we isolate the physics points from regression points in the loss function? 
- Can I use the physics points as self-supervised, pretraining technique?
- How to make the physics constraint hard and not soft (i.e. applicable in the inference time)
## Normalization
- why do we normalize the data of different trajectories all together? isn't it better to normalize for each trajectory separately?

### Relevant models
- Hypergradient
- Model agnostic Meta Learning: Gradient Based
- Bi-level optimization
- Self-tuning networks
# Programing design

## Normalizer class

- Inheriting from torch.nn.Module and use forward method, why?
- When to know to pass the Dataset to the Normalizer class only, or add a set_stats 
function that takes a dataset and calculates the mean and std of the dataset ?