# BSRBF_KAN

In this repo, we use Efficient KAN (https://github.com/Blealtan/efficient-kan/ and FAST-KAN (https://github.com/ZiyaoLi/fast-kan/) to create BSRBF_KAN, which combines B-Spline (**BS**) and Radial Basic Function (**RBF**) for Kolmogorov-Arnold Networks (KANs). We will publish our paper soon.

# Requirements 
* numpy==1.26.4
* numpyencoder==0.3.0
* torch==2.3.0+cu118
* torchvision==0.18.0+cu118
* tqdm==4.66.4
  
# How to combine
We do layer normalization for the input and then combine 3 outputs (base_output, bs_output, and rbf_output).

```
def forward(self, x):
        # layer normalization
        x = self.layernorm(x)
        
        # base
        base_output = F.linear(self.base_activation(x), self.base_weight)
        
        # b_splines
        bs_output = self.b_splines(x).view(x.size(0), -1)
        
        # rbf
        rbf_output = self.rbf(x)
        rbf_output = torch.reshape(rbf_output, (rbf_output.shape[0], -1))
        
        # combine
        bsrbf_output = bs_output + rbf_output
        bsrbf_output = F.linear(bsrbf_output, self.spline_weight)

        return base_output + bsrbf_output
```
# Training

## Parameters
* *mode*: working mode ("train" or "test").
* *model_name*: type of model (bsrbf_kan, efficient_kan, fast_kan, faster_kan).
* *epochs*: the number of epochs.
* *batch_size*: the training batch size.
* *n_input*: The number of input neurons.
* *n_hidden*: The number of hidden neurons. We use only 1 hidden layer. You can modify the code (run_mnist.py) for more layers.
* *n_output*: The number of output neurons (classes). For MNIST, there are 10 classes.
* *grid_size*: The size of grid (default: 5). Use with bsrbf_kan and efficient_kan.
* *spline_order*: The order of spline (default: 3). Use with bsrbf_kan and efficient_kan.
* *num_grids*: The number of grids, equals grid_size + spline_order (default: 8). Use with fast_kan and faster_kan.
* *device*: use "cuda" or "cpu".

## Commands
```python run_mnist.py --mode "train" --model_name "bsrbf_kan" --epochs 15 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --grid_size 5 --spline_order 3```

```python run_mnist.py --mode "train" --model_name "efficient_kan" --epochs 15 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --grid_size 5 --spline_order 3```

```python run_mnist.py --mode "train" --model_name "fast_kan" --epochs 15 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --num_grids 8```

```python run_mnist.py --mode "train" --model_name "faster_kan" --epochs 15 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --num_grids 8```

```python run_mnist.py --mode "train" --model_name "gottlieb_kan" --epochs 15 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --spline_order 3```

# Test on MNIST
We trained the models in **15 epochs on GeForce RTX 3060 Ti** (with other default parameters; see Commands). In general, BSRBF_KAN is stable and converges the best, but it requires more training time than other networks except Gottlieb_KAN. While Gottlieb_KAN can achieve the highest accuracy values, its performance is unstable.

## Best of 5 training times
| Network | Total Layers | Training Accuracy | Val Accuracy | Macro F1 | Macro Precision | Macro Recall | Training time (seconds) |
 | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
 | bsrbf_kan | 2 (768, 64, 10) | **100.0** | 97.63 | 97.6 | 97.61 | 97.59 | 222 |
 | fast_kan | 2 (768, 64, 10) | 99.94 | 97.38 | 97.34 | 97.35 | 97.34 | 102 |
 | faster_kan | 2 (768, 64, 10) | 98.52 | 97.38 | 97.36 | 97.37 | 97.35 | **93** |
 | efficient_kan | 2 (768, 64, 10) | 99.34 | 97.54 | 97.5 | 97.5 | 97.51 | 122 |
 | gottlieb_kan | 3 (768, 64, 64, 10) | 99.66 | **97.78** | **97.74** | **97.74** | **97.73** | 269 |

## Average of 5 training times
 | Network | Total Layers | Training Accuracy | Val Accuracy | Macro F1 | Macro Precision | Macro Recall | Training time (seconds) |
 | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
 | bsrbf_kan | 2 (768, 64, 10) | **100.00 ± 0.00** | **97.55 ± 0.03** | **97.51 ± 0.03** | **97.52 ± 0.03** | **97.50 ± 0.03** | 231 |
 | fast_kan | 2 (768, 64, 10) | 99.94 ± 0.01 | 97.25 ± 0.03 | 97.21 ± 0.03 | 97.22 ± 0.03 | 97.21 ± 0.03 | 101 |
 | faster_kan | 2 (768, 64, 10) | 98.48 ± 0.01 | 97.28 ± 0.06 | 97.25 ± 0.06 | 97.26 ± 0.06 | 97.24 ± 0.06 | **93** |
 | efficient_kan | 2 (768, 64, 10) | 99.37 ± 0.04 | 97.37 ± 0.07 | 97.33 ± 0.07 | 97.34 ± 0.07 | 97.33 ± 0.07 | 120 |
 | gottlieb_kan | 3 (768, 64, 64, 10) | 98.44 ± 0.61 | 97.19 ± 0.22 | 97.14 ± 0.23 | 97.16 ± 0.22 | 97.13 ± 0.23 | 221 |
 
# References
* https://github.com/Blealtan/efficient-kan
* https://github.com/AthanasiosDelis/faster-kan
* https://github.com/ZiyaoLi/fast-kan/
* https://github.com/seydi1370/Basis_Functions
* https://github.com/KindXiaoming/pykan (the original KAN)

# Acknowledgements
We especially thank the contributions of https://github.com/Blealtan/efficient-kan, https://github.com/ZiyaoLi/fast-kan/, and https://github.com/seydi1370/Basis_Functions for their great work in KANs.

# Contact
If you have any questions, please contact: tahoangthang@gmail.com. If you want to know more about me, please visit website: https://tahoangthang.com.
