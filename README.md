# BSRBF_KAN

In this repo, we use Efficient KAN (https://github.com/Blealtan/efficient-kan/ and FAST-KAN (https://github.com/ZiyaoLi/fast-kan/) to create BSRBF_KAN, which combines B-Spline (**BS**) and Radial Basic Function (**RBF**) for Kolmogorov-Arnold Networks (KANs).

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

# Requirements 
* numpy==1.26.4
* numpyencoder==0.3.0
* torch==2.3.0+cu118
* torchvision==0.18.0+cu118
* tqdm==4.66.4

# Test on MNIST
Our results on MNIST are stored in folder **our_output.**

| Network  | Training Accuracy | Val Accuracy | Training time (seconds) |
| ------------- | ------------- |  ------------- | ------------- |
| BSRBF_KAN | 0.9999 | **0.9776** | 119 |
| Fast-KAN | 0.9988 | 0.9762 | 87 |
| EfficientKAN | 0.9899 | 0.9731 | 91 |
| Faster-KAN | 0.9878 | 0.9717 | 92 |

# References
* https://github.com/Blealtan/efficient-kan
* https://github.com/AthanasiosDelis/faster-kan
* https://github.com/ZiyaoLi/fast-kan/
* https://github.com/KindXiaoming/pykan (the original KAN)

# Contact
If you have any questions, please contact: tahoangthang@gmail.com. If you want to know more about me, please visit website: https://tahoangthang.com.
