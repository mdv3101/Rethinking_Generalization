
# SMAI Project

## Understanding Deep Learning Requires Rethinking Generalization ([arxiv](https://arxiv.org/pdf/1611.03530.pdf))


### Aim:
To understand what differentiates neural networks that generalize well from those that do not

### Datasets:
CIFAR10, ImageNet

### Models:
MLP-512, Inception (tiny), Wide ResNet, AlexNet, Inception_v3

### Experiments done:
- Effect of explicit regularization like augmentation, weight decay, dropout
- Effect of implicit regularization like BatchNorm
- Input data corruption: Pixel shuffle, Gaussian pixels, Random pixels
- Label corruption with different corruption levels from 1 to 100 %

### Results
####  Data Corruption experiments
<img src="imgs/Effect of different inputs on convergence.png">

<img src="imgs/Effect of different inputs on training the Neural Network.png">

#### Label corruption experiments
<img src="imgs/Epochs to converge vs label corruption level-2.png">

<img src="imgs/Generalization error vs label corruption level-2.png">

#### Regularization experiments
<img src="imgs/Effect of Explicit Regularization on Generalization Performance.png">

<img src="imgs/Effect of Implicit Regularisation on Generalization Performance.png">

### Checkpoint files of Model Trained on ImageNet (Explicit Regularization):<br>
- w/o Augmentation, Learning Rate Scheduler, Dropout: [checkpoint](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/madhav_agarwal_research_iiit_ac_in/EZWUXUvNGQBCgcJcNxcFg9wB0aYKnly-6dGN8XQWMGTwMA?e=2jlPMu) <br>
- w/o Augmentation, w/o Learning Rate Scheduler, Dropout : [checkpoint](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/madhav_agarwal_research_iiit_ac_in/EYu6wOz7pHxKlo-njCcCRNcBBQxZNhVA_KZnajLpfY9x-Q?e=1KAW6n) <br>
- with Augmentation, Learning Rate Scheduler, Dropout : [checkpoint](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/madhav_agarwal_research_iiit_ac_in/EfM9FrpDFOxLvDBTgfwr9yABsQev1rBbHYfUQDg4mj_hfQ?e=hHIZG8)

### Team Members:
- Siddhant Bansal ([@Sid2697](https://github.com/Sid2697))
- Piyush Singh ([@piyush-kgp](http://github.com/piyush-kgp))
- Madhav Agarwal ([@mdv3101](https://github.com/mdv3101))
