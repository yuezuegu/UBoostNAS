# U-Boost NAS: Utilization-Boosted Differentiable Neural Architecture Search

This repo contains the source code for the U-Boost NAS method presented in this [paper](https://arxiv.org/abs/2203.12412). It optimizes hardware resource utilization with task accuracy and latency to maximize inference performance. It estimates hardware utilization using a novel computational model for DNN accelerators.

## Code structure/note
* `src/search`: contains the code for microarchitecture and channel search stages
* `src/vanilla`: contains the code for final training stage
* `src/data`: contains the code for dataloaders and preprocessing for various datasets
* `src/hardware`: contains the code for hardware model and cycle-accurate hardware simulations

## NAS
This code implements the microarchitecture search as in DARTS-like methods and channel search using DMaskNAS method.

## Hardware model
Computes the following runtime for convolutional cells:
<!-- $$
\texttt{RUNTIME} = \left\lceil \frac{k_1k_2c}{s_1} \right\rceil \left\lceil \frac{f}{s_2} \right\rceil hwB 
$$ --> 

<div align="center"><img width=300 style="background: white;" src="https://render.githubusercontent.com/render/math?math=%5Ctexttt%7BRUNTIME%7D%20%3D%20%5Cleft%5Clceil%20%5Cfrac%7Bk_1k_2c%7D%7Bs_1%7D%20%5Cright%5Crceil%20%5Cleft%5Clceil%20%5Cfrac%7Bf%7D%7Bs_2%7D%20%5Cright%5Crceil%20hwB%20"></div>
with "matrixification" of the tensors,  where:

* B is the number of batches
* h is the height of the input
* w is the width of the input
* c is the number of channels
* f is the number of filters 
* k1 is one dimension of the kernel
* k2 is the other dimension of the kernel
* s1 is one dimension of the systolic array
* s2 is the other dimension of the systolic array

However, the ceil function is not differentiable and can only be used as a collection of point estimates. This hinders the neural architecture seach and allows only for evolutionary or reinforcement learning methods, which require orders of magnitude more computational resources compared to differentiable methods. For this reason, the ceil function is replaced with a soft approximation, the `smooth ceiling`:

<!-- $$
f_{T, \mathbf{ w }}(x)=\sum_{i} \frac{1}{1+\exp{(-T (x-w_i))}}
$$ --> 

<div align="center"><img width=300 style="background: white;" src="https://render.githubusercontent.com/render/math?math=f_%7BT%2C%20%5Cmathbf%7B%20w%20%7D%7D(x)%3D%5Csum_%7Bi%7D%20%5Cfrac%7B1%7D%7B1%2B%5Cexp%7B(-T%20(x-w_i))%7D%7D"></div>


for wi intervals between zero and a fixed value. This model corresponds more with the realistic case.

TODO: explain a bit more the realistic case.

## How to run

`python main.py #--help for information about optional arguments`


## Citation
If you use this code, please cite our [paper](https://arxiv.org/abs/2203.12412).
