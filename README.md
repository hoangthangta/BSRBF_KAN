# BSRBF_KAN

In this repo, we use Efficient KAN (https://github.com/Blealtan/efficient-kan/ and FAST-KAN (https://github.com/ZiyaoLi/fast-kan/) to create BSRBF_KAN, which combines B-Spline (**BS**) and Radial Basic Function (**RBF**) for Kolmogorov-Arnold Networks (KANs). Our paper is available at https://arxiv.org/abs/2406.11173.

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
        rbf_output = self.rbf(x).view(x.size(0), -1)
        
        # combine
        bsrbf_output = bs_output + rbf_output
        bsrbf_output = F.linear(bsrbf_output, self.spline_weight)

        return base_output + bsrbf_output
```
# Training

## Parameters
* *mode*: working mode ("train" or "test").
* *ds_name*: dataset name ("mnist" or "fashion_mnist").
* *model_name*: type of model (bsrbf_kan, efficient_kan, fast_kan, faster_kan).
* *epochs*: the number of epochs.
* *batch_size*: the training batch size.
* *n_input*: The number of input neurons.
* *n_hidden*: The number of hidden neurons. We use only 1 hidden layer. You can modify the code (run.py) for more layers.
* *n_output*: The number of output neurons (classes). For MNIST, there are 10 classes.
* *grid_size*: The size of grid (default: 5). Use with bsrbf_kan and efficient_kan.
* *spline_order*: The order of spline (default: 3). Use with bsrbf_kan and efficient_kan.
* *num_grids*: The number of grids, equals grid_size + spline_order (default: 8). Use with fast_kan and faster_kan.
* *device*: use "cuda" or "cpu".

## Commands
```python run.py --mode "train" --ds_name "mnist" --model_name "bsrbf_kan" --epochs 15 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --grid_size 5 --spline_order 3```

```python run.py --mode "train" --ds_name "mnist" --model_name "efficient_kan" --epochs 15 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --grid_size 5 --spline_order 3```

```python run.py --mode "train" --ds_name "mnist" --model_name "fast_kan" --epochs 15 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --num_grids 8```

```python run.py --mode "train" --ds_name "mnist" --model_name "faster_kan" --epochs 15 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --num_grids 8```

```python run.py --mode "train" --ds_name "mnist" --model_name "gottlieb_kan" --epochs 15 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --spline_order 3```

```python run.py --mode "train" --ds_name "mnist" --model_name "mlp" --epochs 15 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10```

# Test on MNIST
We trained the models in **15 epochs on GeForce RTX 3060 Ti** (with other default parameters; see Commands). In general, BSRBF_KAN is stable and converges the best, but it requires more training time than other networks except Gottlieb_KAN. While achieving the highest accuracy values, Gottlieb_KAN's performance is unstable.

## Best of 5 training sessions
| Network | Total Layers | Training Accuracy | Val Accuracy | Macro F1 | Macro Precision | Macro Recall | Training time (seconds) | # Params
 | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
 | bsrbf_kan | 2 (768, 64, 10) | **100.0** | 97.63 | 97.6 | 97.61 | 97.59 | 222 | 459040 |
 | fast_kan | 2 (768, 64, 10) | 99.94 | 97.38 | 97.34 | 97.35 | 97.34 | 102 | 459114 |
 | faster_kan | 2 (768, 64, 10) | 98.52 | 97.38 | 97.36 | 97.37 | 97.35 | **93** | 408224 |
 | efficient_kan | 2 (768, 64, 10) | 99.34 | 97.54 | 97.5 | 97.5 | 97.51 | 122 | 508160 |
 | gottlieb_kan | 3 (768, 64, 64, 10) | 99.66 | **97.78** | **97.74** | **97.74** | **97.73** | 269 | 219927 | 
 | MLP   | 99.42 |   97.69 | 97.66 | 273 | **52512** |


## Average of 5 training sessions
 | Network | Total Layers | Training Accuracy | Val Accuracy | Macro F1 | Macro Precision | Macro Recall | Training time (seconds) |
 | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
 | bsrbf_kan | 2 (768, 64, 10) | **100.00 ± 0.00** | **97.55 ± 0.03** | **97.51 ± 0.03** | **97.52 ± 0.03** | **97.50 ± 0.03** | 231 |
 | fast_kan | 2 (768, 64, 10) | 99.94 ± 0.01 | 97.25 ± 0.03 | 97.21 ± 0.03 | 97.22 ± 0.03 | 97.21 ± 0.03 | 101 |
 | faster_kan | 2 (768, 64, 10) | 98.48 ± 0.01 | 97.28 ± 0.06 | 97.25 ± 0.06 | 97.26 ± 0.06 | 97.24 ± 0.06 | **93** |
 | efficient_kan | 2 (768, 64, 10) | 99.37 ± 0.04 | 97.37 ± 0.07 | 97.33 ± 0.07 | 97.34 ± 0.07 | 97.33 ± 0.07 | 120 |
 | gottlieb_kan | 3 (768, 64, 64, 10) | 98.44 ± 0.61 | 97.19 ± 0.22 | 97.14 ± 0.23 | 97.16 ± 0.22 | 97.13 ± 0.23 | 221 |

# Test on FashionMNIST
Training on MNIST seems easy, making it difficult to compare BSRBF-KAN accurately to other models; therefore, we would like to work on FashionMNIST. We trained the models in **25 epochs on GeForce RTX 3060 Ti** (with other default parameters; see Commands). Like MNIST, BSRBF_KAN is also stable and converges the best. FastKAN achieves the best performance.

## Best of 5 training sessions
 | Network | Training Accuracy | Val Accuracy | Macro F1 | Macro Precision | Macro Recall | Training time (seconds) |
 | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
 | bsrbf_kan | **99.3** | 89.59 | 89.54 | 89.55 | 89.57 | 219 |
 | fast_kan | 98.27 | **89.62** | 89.6 | 89.6 | 89.63 | 160 |
 | faster_kan | 94.4 | 89.39 | 89.3 | 89.31 | 89.34 | 157 |
 | efficient_kan | 94.83 | 89.11 | 89.04 | 89.03 | 89.09 | 182 |
 | gottlieb_kan | 93.79 | 87.69 | 87.61 | 87.6 | 87.66 | 241 |
 | mlp | 93.58 | 88.51 | 88.44 | 88.42 | 88.48 | **147** |

## Average of 5 training sessions
| Network | Training Accuracy | Val Accuracy | Macro F1 | Macro Precision | Macro Recall | Training time (seconds) |
 | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
 | bsrbf_kan | **99.19 ± 0.03** | 89.33 ± 0.07 | 89.29 ± 0.07 | 89.30 ± 0.07 | 89.33 ± 0.07 | 211 |
 | fast_kan | 98.19 ± 0.04 | **89.42 ± 0.07** | 89.38 ± 0.07 | 89.38 ± 0.07 | 89.41 ± 0.07 | 162 |
 | faster_kan | 94.40 ± 0.01 | 89.26 ± 0.06 | 89.17 ± 0.07 | 89.17 ± 0.07 | 89.23 ± 0.07 | 154 |
 | efficient_kan | 94.76 ± 0.06 | 88.92 ± 0.08 | 88.85 ± 0.09 | 88.85 ± 0.09 | 88.91 ± 0.09 | 183 |
 | gottlieb_kan | 90.66 ± 1.08 | 87.16 ± 0.24 | 87.07 ± 0.25 | 87.07 ± 0.25 | 87.13 ± 0.25 | 238 |
 | mlp | 93.56 ± 0.05 | 88.39 ± 0.06 | 88.36 ± 0.05 | 88.35 ± 0.05 | 88.41 ± 0.05 | **148** |
 
# References
* https://github.com/Blealtan/efficient-kan
* https://github.com/AthanasiosDelis/faster-kan
* https://github.com/ZiyaoLi/fast-kan/
* https://github.com/seydi1370/Basis_Functions
* https://github.com/KindXiaoming/pykan (the original KAN)

# Acknowledgements
We especially thank the contributions of https://github.com/Blealtan/efficient-kan, https://github.com/ZiyaoLi/fast-kan/, and https://github.com/seydi1370/Basis_Functions for their great work in KANs.

# Paper
```
@misc{ta2024bsrbfkan,
      title={BSRBF-KAN: A combination of B-splines and Radial Basic Functions in Kolmogorov-Arnold Networks}, 
      author={Hoang-Thang Ta},
      year={2024},
      eprint={2406.11173},
      archivePrefix={arXiv}
      }
}
```

# Contact
If you have any questions, please contact: tahoangthang@gmail.com. If you want to know more about me, please visit website: https://tahoangthang.com.
