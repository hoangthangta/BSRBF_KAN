# BSRBF_KAN

In this repo, we use Efficient KAN (https://github.com/Blealtan/efficient-kan/ and FAST-KAN (https://github.com/ZiyaoLi/fast-kan/) to create BSRBF_KAN, which combines B-Spline (**BS**) and Radial Basic Function (**RBF**) for Kolmogorov-Arnold Networks (KANs). We will publish our paper soon.

# Requirements 
* numpy==1.26.4
* numpyencoder==0.3.0
* torch==2.3.0+cu118
* torchvision==0.18.0+cu118
* tqdm==4.66.4
  
# How to combine
We do layer normalization for the input and then merge 3 outputs (base_output, spline_output, and rbf_output).

```
def forward(self, x):
        # layer normalization
        x = self.layernorm(x)
        
        # base
        #bias = torch.randn(self.output_dim)
        #base_output = F.linear(self.base_activation(x), self.base_weight, bias)
        base_output = F.linear(self.base_activation(x), self.base_weight)
        
        # b_splines
        spline_output = F.linear(self.b_splines(x).view(x.size(0), -1), self.spline_weight)
        
        # rbf
        rbf_output = self.rbf(x)
        rbf_output = torch.reshape(rbf_output, (rbf_output.shape[0], -1))
        rbf_output = F.linear(rbf_output, self.spline_weight)
        #rbf_output = self.linear(rbf_output)

        return base_output + rbf_output + spline_output
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
* *device*: use "cuda" or "CPU".

## Commands
```python run_mnist.py --mode "train" --model_name "bsrbf_kan" --epochs 10 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --grid_size 5 --spline_order 3```

```python run_mnist.py --mode "train" --model_name "efficient_kan" --epochs 10 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --grid_size 5 --spline_order 3```

```python run_mnist.py --mode "train" --model_name "fast_kan" --epochs 10 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --num_grids 8```

```python run_mnist.py --mode "train" --model_name "faster_kan" --epochs 10 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --num_grids 8```

# Test on MNIST
We trained the models in 15 epochs on GeForce RTX 3060 Ti (with other default parameters; see Commands), then stored our results in the folder **our_output.** BSRBF_KAN can converge better than other networks but requires more training time.

| Network  | Training Accuracy | Val Accuracy | Training time (seconds) |
| ------------- | ------------- |  ------------- | ------------- |
| BSRBF_KAN | - | - | - |
| Fast-KAN | - | - | - |
| EfficientKAN - | - | - |
| Faster-KAN - | - | - |

# References
* https://github.com/Blealtan/efficient-kan
* https://github.com/AthanasiosDelis/faster-kan
* https://github.com/ZiyaoLi/fast-kan/
* https://github.com/KindXiaoming/pykan (the original KAN)

# Acknowledgements
We especially thank the contributions of https://github.com/Blealtan/efficient-kan and https://github.com/ZiyaoLi/fast-kan/ for their great work in KANs.

# Contact
If you have any questions, please contact: tahoangthang@gmail.com. If you want to know more about me, please visit website: https://tahoangthang.com.
