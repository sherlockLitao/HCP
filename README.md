# Hilbert Curve Projection Distance for Distribution Comparison

This repository is the official implementation of Hilbert Curve Projection Distance for Distribution Comparison [https://arxiv.org/abs/2205.15059].
This folder includes main experiments based on HCP, IPRHCP, and PRHCP.
Feel free to ask if any question.

If you use this toolbox in your research and find it useful, please cite:
```
@article{li2024hilbert,
  title={Hilbert curve projection distance for distribution comparison},
  author={Li, Tao and Meng, Cheng and Xu, Hongteng and Yu, Jun},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  publisher={IEEE}
}
```

# Platform:
We test these examples in a conda environment on Ubuntu 20.04.3, with cuda 10.1.
All experiments are implemented by a AMD 3600 CPU and
a RTX 1080Ti GPU.
(We also successfully test it on a Macbook with M1 chip.)

# Main Dependencies

## python
* argparse
* matplotlib
* numpy
* pickle
* pytorch
* sklearn
* pybind11
* hilbertcurve
* POT


## C++ library
* eigen3

## How to install eigen3:
sudo apt install libeigen3-dev




# What is included ?

* folder src: 
Both C++ based and python based implementations of HCP distance, as well as its invariants.
C++ uses recursive sort and python uses Hilbert indices.
* folder Autoencoder_big(DCGAN):
IPRHCP and PRHCP implementations of Autoencoder based on DCGAN.
* folder Autoencoder_small:
HCP implementations of Autoencoder on low dimensional latent space.
* folder 3D_point_classification:
Implementations of 3D point cloud classification.
* folder simu_flow:
Simulation of Approximation of Wasserstein flow.
* folder simu_high_dimensional_data:
  - prhcp_and_iprhcp_rate.ipynb: Empirical sample complexity and effectiveness for IPRHCP and PRHCP in high dimension compared to HCP and Wasserstein distance
  - prhcp_subspace.ipynb: Like PRW and SRW, we find PRHCP is also robust to noise but with low complexity.
  - three_gaussian.ipynb: Effectiveness and efficiency for PRHCP 
  
* folder some details:
  - inequality.ipynb: numerical proof of inequality about IPRHCP
  - k_order.ipynb: analysis of hyperparameter k (order of Hilbert curve $\widehat{H}_k$ )
  - motivation.ipynb: Figure 1 and some other findings which show SW has some disadvantages because it considers projected samples
  - time.ipynb: Time analysis






# Test our method:

## Before all, you need to:

Firstly, open a terminal and open file src/setup.py and change line 7,8.

* line 7 is path of Eigen (mine is /usr/include/eigen3/)
* line 8 is path of pybind11 (mine is /home/usrname/anaconda3/lib/.../site-packages/pybind11/include)

Secondly, you can open folder src and run: 

* python setup.py build_ext --inplace

**Then, you can use python to run all code.**


## For Autoencoder, you can:


As to Autoencoder_small, open folder Autoencoder_small and run
* python hcpae.py

As to Autoencoder_big(DCGAN), open folder Autoencoder_big(DCGAN) and run
* python test_prhcpae.py
* python test_iprhcpae.py 

Note: you need to download CelebA dataset first before running on CelebA because there is something wrong with torch.utils.data.DataLoader to download directly from website.


## For 3D point cloud classification:

You need to download the ModelNet10 dataset first.



# Some results

All *.ipynb files provide **figures** in our manuscript.

What's more, we provide **videos and figures** in folder simu_flow/saved_results_flows/

Color Transfer based on HCP and SW:

![image](https://github.com/sherlockLitao/HCP/blob/main/color_transfer/Results/color1.png)

We show the results of several iterations:

![image](https://github.com/sherlockLitao/HCP/blob/main/color_transfer/Results/color2.png)

![image](https://github.com/sherlockLitao/HCP/blob/main/color_transfer/Results/color3.png)



# Authors

This toolbox has been created and is maintained by

* [Tao Li](https://github.com/sherlockLitao)
* [Cheng Meng](https://github.com/ChengzijunAixiaoli)
* Jun Yu
* [Hongteng Xu](https://github.com/HongtengXu)


# Reference
[https://github.com/CGAL/cgal]

[https://github.com/eifuentes/swae-pytorch]

[https://github.com/HongtengXu/Relational-AutoEncoders]

[https://github.com/kimiandj/gsw]

[https://www.kaggle.com/code/balraj98/pointnet-for-3d-object-classification-pytorch]

[https://github.com/iamalexkorotin/Wasserstein2Barycenters]
