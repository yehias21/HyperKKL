# Todo
- [ ] 


# Model Loading

- 2 models: encoder and decoder
  - different update technique
- Hypernetwork --> maybe 0, 1, 2
  - Encoder --> easy
  - Decoder --> Need to know strategy (chunked, full, delta), 
 and the number of parameters to update

- Hypothesis: Main network object == observer x,z,u,y,t in training

structural problem 
- Model initiation: MLP, Transformer, RNN
- Hypernetwork: count, structure of the decoder 

Control flow problem
 - Model alone
 - Hypernetwork update model, 
 - Model frozen + hypernetwork

### Flow of control
1. train.py -call-> load_model(cfg.model)
- Return -> one module only (Main_mapper, observer as blackbox)
- during Training: Pass x,z,u,y,t in a dict, return y_hat, x_hat, z_hat
2. load_model: 
 -  black box
 - return Main_mapper

- Structure of the mappers     | 
                                ---> How the hypernetwork_decoder will be like ---> control flow forward   
- Update scheme I want to use  |
- Constraint: mappers and hypernetworks are in the end a nn.Moddule, i.e: we can have MLP for hypernetwork and mappers 
- I want to write the minimum amount of code ---> abstraction

Hypenetwork --> is only updated by backpropagation
Decoder bta3 elhypernetwork  how to initialize it ?
mappers can be updated by multiple things (e.g: hypernetwork, backpropagation, etc)

### Decoder Hypernetwork
- Input: Digest of Encoder 
- Output: Decoder:  weights
  - FULL: Return the weights of the mapper on one time? code problem: a module to cut and reshape
  - Chuncked: Return the weights of the mapper on multiple time? code problem: a module to cut and reshape
  - Dictionary: a module