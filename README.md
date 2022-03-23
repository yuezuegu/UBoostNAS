# DmaskingNAS


## Code structure/note
* `model`: in our case LeNet. NAS is performed on this with search space defined by `arch_def`
* `hw_model`: simulates the device performance (i.e. latency) with the architecture defined by the `model`.
* `loss_fn`: the loss function that drives NAS: $\mathcal{L}=\mathcal{L}_{data}+\lambda\cdot\mathcal{L}_{latency}$ 

## NAS
The NAS is implemented with DARTS-like method, see [1]. 

> In each epoch, we train the network weights with 80% of training samples using SGD. We then train the
Gumbel Softmax sampling parameter $\alpha$ with the remaining 20% using Adam.

## Differences with prior work
Prior work oversimplifies the hardware model latency measurements with constant or monotonically non-decreasing function wrt no. of channels

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

## Next steps
1. code the part that trains until completion the "winner" arch of the NAS





## References

[1]: Wan A, Dai X, Zhang P, He Z, Tian Y, Xie S, Wu B, Yu M, Xu T, Chen K, Vajda P. Fbnetv2: Differentiable neural architecture search for spatial and channel dimensions. InProceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition 2020 (pp. 12965-12974).