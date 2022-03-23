# U-Boost NAS: Utilization-Boosted Differentiable Neural Architecture Search

This repo contains the source code for the U-Boost NAS method presented in [cite]. It optimizes hardware resource utilization with task accuracy and latency to maximize inference performance. It estimates hardware utilization using a novel computational model for DNN accelerators.



## Code structure/note
* `src/search`: contains the code for microarchitecture and channel search stages
* `src/vanilla`: contains the code for final training stage
* `src/data`: contains the code for dataloaders and preprocessing for various datasets
* `src/hardware`: contains the code for hardware model and cycle-accurate hardware simulations

## NAS
This code implements the microarchitecture search as in DARTS-like methods and channel search using DMaskNAS method [2].

## Hardware model
Computes the following runtime for convolutional cells:
$$
\texttt{RUNTIME} = \left\lceil \frac{k_1k_2c}{s_1} \right\rceil \left\lceil \frac{f}{s_s} \right\rceil hwB 
$$
with "matrixification" of the tensors,  where:

* $B$ is the number of batches
* $h$ is the height of the input
* $w$ is the width of the input
* $c$ is the number of channels
* $f$ is the number of filters 
* $k_1$ is one dimension of the kernel
* $k_2$ is the other dimension of the kernel
* $s_1$ is one dimension of the systolic array
* $s_2$ is the other dimension of the systolic array

However, the ceil function is not differentiable and can only be used as a collection of point estimates. This hinders the neural architecture seach and allows only for evolutionary or reinforcement learning methods, which require orders of magnitude more computational resources compared to differentiable methods. For this reason, the ceil function is replaced with a soft approximation, the `smooth ceiling`:
$$
f_{T, \mathbf{ w }}(x)=\sum_{i} \frac{1}{1+\exp{(-T (x-w_i))}}
$$ 
for $w_i$ intervals between zero and a fixed value. This model corresponds more with the realistic case.

TODO: explain a bit more the realistic case.

## How to run

`python main.py #--help for information about optional arguments`


## References

[1]: Wan A, Dai X, Zhang P, He Z, Tian Y, Xie S, Wu B, Yu M, Xu T, Chen K, Vajda P. Fbnetv2: Differentiable neural architecture search for spatial and channel dimensions. InProceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition 2020 (pp. 12965-12974).
