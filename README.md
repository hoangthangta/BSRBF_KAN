**For a new KAN that is based on function combinations (also include BSRBF-KAN and better code), see: https://github.com/hoangthangta/FC_KAN.**

# BSRBF_KAN

In this repo, we use Efficient KAN (https://github.com/Blealtan/efficient-kan/ and FAST-KAN (https://github.com/ZiyaoLi/fast-kan/) to create BSRBF_KAN, which combines B-Splines (**BS**) and Radial Basis Functions (**RBF**) for Kolmogorov-Arnold Networks (KANs). 

Our paper's name is misspelled: "BSRBF-KAN: A combination of B-splines and Radial Basi~~c~~s (**s, not c**) Functions in Kolmogorov-Arnold Networks." Please cite our paper correctly; thank you!

Our paper is available at https://arxiv.org/abs/2406.11173 (**BSRBF-KAN: A combination of B-splines and Radial Basis Functions in Kolmogorov-Arnold Networks**) or https://www.researchgate.net/publication/381471539_BSRBF-KAN_A_combination_of_B-splines_and_Radial_Basis_Functions_in_Kolmogorov-Arnold_Networks.

# Requirements 
* numpy==1.26.4
* numpyencoder==0.3.0
* torch==2.3.0+cu118
* torchvision==0.18.0+cu118
* tqdm==4.66.4
  
# How to combine?
We start with layer normalization of the input and then merge three outputs: base_output, bs_output, and rbf_output. Although this method appears simple, finding an optimal combined KAN that is better than the available KANs in terms of something (accuracy, speed, convergence, etc) is time-consuming. We hope our research will lead to the development of various combined KANs using mathematical functions.

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
* *n_examples*: the number of examples in the training set used for training (default: -1, mean use all training data)

## Commands
```python run.py --mode "train" --ds_name "mnist" --model_name "bsrbf_kan" --epochs 25 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --grid_size 5 --spline_order 3```

```python run.py --mode "train" --ds_name "mnist" --model_name "efficient_kan" --epochs 25 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --grid_size 5 --spline_order 3```

```python run.py --mode "train" --ds_name "mnist" --model_name "fast_kan" --epochs 25 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --num_grids 8```

```python run.py --mode "train" --ds_name "mnist" --model_name "faster_kan" --epochs 25 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --num_grids 8```

```python run.py --mode "train" --ds_name "mnist" --model_name "gottlieb_kan" --epochs 25 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --spline_order 3```

```python run.py --mode "train" --ds_name "mnist" --model_name "mlp" --epochs 25 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10```

# Test on MNIST + Fashion MNIST + SL MNIST
We trained the models **on GeForce RTX 3060 Ti** (with other default parameters; see Commands). The results are in our updated paper, **we are working to update them**.

# References
* https://github.com/Blealtan/efficient-kan
* https://github.com/AthanasiosDelis/faster-kan
* https://github.com/ZiyaoLi/fast-kan/
* https://github.com/seydi1370/Basis_Functions
* https://github.com/KindXiaoming/pykan (the original KAN)

# Acknowledgements
We especially thank the contributions of https://github.com/Blealtan/efficient-kan, https://github.com/ZiyaoLi/fast-kan/, and https://github.com/seydi1370/Basis_Functions for their great work in KANs.

Also, give me a star if you like this repo. Thanks!

# Paper
```
@misc{ta2024bsrbfkan,
      title={BSRBF-KAN: A combination of B-splines and Radial Basis Functions in Kolmogorov-Arnold Networks}, 
      author={Hoang-Thang Ta},
      year={2024},
      eprint={2406.11173},
      archivePrefix={arXiv}
      }
}
```

# Contact
If you have any questions, please contact: tahoangthang@gmail.com. If you want to know more about me, please visit website: https://tahoangthang.com.
