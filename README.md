<p align="center"><img width="40%" src="https://github.com/Walhalla-Summary/Pytorch_Learning/blob/master/logo/pytorch_logo.png" /></p>

--------------------------------------------------------------------------------

Pytorch知识点总结、强化学习(Reinforcement Learning)、机器学习(Machine Learning)、神经网络(Neural Network)


PyTorch 是一个 Python 包，它提供了两个高级特性：

-   具有强大 GPU 加速功能的张量计算（如 NumPy）
-   建立在基于磁带的 autograd 系统上的深度神经网络


在粒度级别上，PyTorch 是一个由以下组件组成的库： 

|                          Component                           |                         Description                          |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|   [**torch**](https://pytorch.org/docs/stable/torch.html)    |          像 NumPy 这样的张量库，具有强大的 GPU 支持          |
| [**torch.autograd**](https://pytorch.org/docs/stable/autograd.html) |  一个基于tape的自动微分库，支持torch中所有可微分的张量运算   |
|  [**torch.jit**](https://pytorch.org/docs/stable/jit.html)   | 用于从 PyTorch 代码创建可序列化和可优化模型的编译堆栈 (TorchScript) |
|   [**torch.nn**](https://pytorch.org/docs/stable/nn.html)    |    与 autograd 深度集成的神经网络库，旨在实现最大的灵活性    |
| [**torch.multiprocessing**](https://pytorch.org/docs/stable/multiprocessing.html) | Python 多处理，但具有跨进程的火炬张量的神奇内存共享。对数据加载和 Hogwild 训练很有用 |
| [**torch.utils**](https://pytorch.org/docs/stable/data.html) |           DataLoader和其他实用程序功能，以方便使用           |

通常，PyTorch 用作：

-   替代 NumPy 以使用 GPU 的强大功能。
-   一个深度学习研究平台，可提供最大的灵活性和速度。
