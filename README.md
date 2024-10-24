# README

## 项目介绍

本项目是一个基于 Pytorch 的 PINN 框架，用于求解常微分方程的初值问题和常微分方程的系数估计问题。

本项目采用的神经网络为基础的 MLP 网络，输入为时间坐标，输出为系统的位移响应。


本项目参考了项目:

- [PINN-iPINN](https://github.com/jmorrow1000/PINN-iPINN)
- [PINN-with-forcing-function](https://github.com/jmorrow1000/PINN-with-forcing-function)

## 仿真实验结果

- 对于常微分方程系统来说，求解之后的位移和速度之间的数量级差异不能太大
- batch_size 对于训练的效果至关重要，这里 32 是一个最合适的取值。
- 正问题比反问题好解决，反问题的参数基本上没有辨识出来。这个和系统的初值问题以及 setting time 有关。
