# 第三章 Pytorch张量

## 张量

 张量是 PyTorch 中的核心数据抽象，torch.tensor包含单一数据元素的多维矩阵，表示由数值组成的数组，创建的数组可能有多个维度。具有⼀个轴的张量对应数学上的向量(vector)，具有两个轴的张量对应数学上的矩阵(matrix)，具有两个轴以上的张量没有特殊的数学名称

## 数据类型

Torch定义了10几种具有CPU和GPU变体的张量类型

|                           数据类型                           |                 类型                  |        CPU张量         |           GPU张量           |
| :----------------------------------------------------------: | :-----------------------------------: | :--------------------: | :-------------------------: |
|                          32 位浮点                           |   `torch.float32`或者`torch.float`    |  `torch.FloatTensor`   |  `torch.cuda.FloatTensor`   |
|                          64 位浮点                           |   `torch.float64`或者`torch.double`   |  `torch.DoubleTensor`  |  `torch.cuda.DoubleTensor`  |
| 16 位浮点数[1](https://pytorch.org/docs/stable/tensors.html#id4) |    `torch.float16`或者`torch.half`    |   `torch.HalfTensor`   |   `torch.cuda.HalfTensor`   |
| 16 位浮点数[2](https://pytorch.org/docs/stable/tensors.html#id5) |           `torch.bfloat16`            | `torch.BFloat16Tensor` | `torch.cuda.BFloat16Tensor` |
|                          32 位复数                           |  `torch.complex32`或者`torch.chalf`   |                        |                             |
|                          64 位复数                           |  `torch.complex64`或者`torch.cfloat`  |                        |                             |
|                          128 位复数                          | `torch.complex128`或者`torch.cdouble` |                        |                             |
|                      8 位整数（无符号）                      |             `torch.uint8`             |   `torch.ByteTensor`   |   `torch.cuda.ByteTensor`   |
|                      8 位整数（有符号）                      |             `torch.int8`              |   `torch.CharTensor`   |   `torch.cuda.CharTensor`   |
|                     16 位整数（有符号）                      |    `torch.int16`或者`torch.short`     |  `torch.ShortTensor`   |  `torch.cuda.ShortTensor`   |
|                     32 位整数（有符号）                      |     `torch.int32`或者`torch.int`      |   `torch.IntTensor`    |   `torch.cuda.IntTensor`    |
|                     64 位整数（有符号）                      |     `torch.int64`或者`torch.long`     |   `torch.LongTensor`   |   `torch.cuda.LongTensor`   |
|                            布尔值                            |             `torch.bool`              |   `torch.BoolTensor`   |   `torch.cuda.BoolTensor`   |
|                  量化的 8 位整数（无符号）                   |            `torch.quint8`             |   `torch.ByteTensor`   |              /              |
|                  量化的 8 位整数（有符号）                   |             `torch.qint8`             |   `torch.CharTensor`   |              /              |
|                  量化的 32 位整数（有符号）                  |            `torch.qint32`             |   `torch.IntTensor`    |              /              |
| 量化的 4 位整数（无符号）[3](https://pytorch.org/docs/stable/tensors.html#id6) |           `torch.quint4x2`            |   `torch.ByteTensor`   |              /              |

1.binary16: 使用1个符号、5个指数和10位有效位

2.BrainFloating: 使用1个符号、8个指数和7位有效位。具有与float32性质

3.量化的4位整数存储为8位有符号整数，目前仅仅在EmbeddingBag运算符中支持。

4.torch.tensor默认张量类型为torch.FloatTensor

## 初始化和基本操作

可使用构造函数从Python list 或序列构造张量 torch.tensor()

```python
torch.tensor([1., -1.], [1., -1.])
```

```python
tensor([[ 1.0000, -1.0000],
        [ 1.0000, -1.0000]])
```

通过numpy构造

```python
torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]))
```

```python
tensor([[ 1,  2,  3],
        [ 4,  5,  6]])
```

torch.tensor()复制data，如果存在张量data并且指向改其requires_grad标志，需要使用requires_grad_()或detach()避免使用副本。如果为numpy数组并且想避免复制，需使用torch.as\_tensor()



通过将torch.dtype或torch.device传递给构造函数或张量创建操作来构造特定数据类型的张量

```python
>>> torch.zeros([2, 4], dtype=torch.int32)
tensor([[ 0,  0,  0,  0],
        [ 0,  0,  0,  0]], dtype=torch.int32)
>>> cuda0 = torch.device('cuda:0')
>>> torch.ones([2, 4], dtype=torch.float64, device=cuda0)
tensor([[ 1.0000,  1.0000,  1.0000,  1.0000],
        [ 1.0000,  1.0000,  1.0000,  1.0000]], dtype=torch.float64, device='cuda:0')
```

可使用Python的索引和切片访问和修改张量的内容

```python
>>> x = torch.tensor([[1, 2, 3], [4, 5, 6]])
>>> print(x[1][2])
tensor(6)
>>> x[0][1] = 8
>>> print(x)
tensor([[ 1,  8,  3],
        [ 4,  5,  6]])
```

使用torch.Tensor.item()从包含单个值的张量中获取Python数字

```python
>>> x = torch.tensor([[1]])
>>> x
tensor([[ 1]])
>>> x.item()
1
>>> x = torch.tensor(2.5)
>>> x
tensor(2.5000)
>>> x.item()
2.5
```

记录操作，并进行自动微分，设置requires_grad = True

```python
>>> x = torch.tensor([[1., -1.], [1., 1.]], requires_grad=True)
>>> out = x.pow(2).sum()
>>> out.backward()
>>> x.grad
tensor([[ 2.0000, -2.0000],
        [ 2.0000,  2.0000]])
```

每个张量都存在一个关联的torch.Storage，主要用于保存其数据，张量类同时引入提供了存储的多维、跨步试图，并在其上定义了数字操作。

## 

改变张量的方法用下划线后缀标记，如：

torch.FloatTensor.abs_()，它会就地计算绝对值并返回修改后的张量，同时torch.FloatTensor.abs()在新张量中计算结果。

注意：当前的实现torch.tensor引入了内存开销，因此在具有许多微笑张量的应用程序中可能会导致意外的高内存使用，使用时，需考虑是否使用大型结构。

## 创建张量

```python
# 导包
import torch
import math

x = torch.empty(3, 4)
print(type(x))
print(x)
```

输出：

```python
<class 'torch.Tensor'>
tensor([[ 6.8382e-18,  3.0801e-41, -1.3010e+20,  4.5804e-41],
        [-1.3122e+20,  4.5804e-41,  6.8386e-18,  3.0801e-41],
        [-1.3010e+20,  4.5804e-41, -1.3122e+20,  4.5804e-41]])
```

解释：

-   使用`torch.empty()方法`创建了一个张量。
-   张量本身是二维的，有 3 行 4 列。
-   返回对象类型是;`torch.Tensor`的别名`torch.FloatTensor`。默认情况下，PyTorch 张量填充有 32 位浮点数。
-   在打印张量时，可能会看到一些看起来很随机的值。该`torch.empty()`调用为张量分配内存，但没有使用任何值对其进行初始化 - 所以看到的是分配时内存中的任何内容。

1.使用预先存在的数据创建张量，totch.tensor()

2.创建具有特定大小的张量，torch.\*

3.创建与另一个张量具有相同大小(和相似类型)的张量，torch.\*\_like

4.创建与另一个张量具有相似类型，但大小不同的张量，torch.new\_\*



2.1 随机初始化矩阵，通过torch.randn()的方法，构造一个随机初始化的矩阵

```python
import torch

# 初始化矩阵
x = torch.randn(4, 3)
print(x)

# randn() 返回一个张量，其中填充了来自均值“0”和方差“1”（也称为标准正态分布）的正态分布的随机数
```

```python
tensor([[ 0.5972, -0.6283, -0.0439],
        [-0.6223,  0.3107,  0.4680],
        [-1.5478,  1.6934,  0.3533],
        [-0.8553,  2.2113,  1.6057]])
```

```python
>>> torch.randn(4)
tensor([-2.1436, 0.9966, 2.3426, -0.6366])

>>> torch.randn(2, 3)
tensor([[1.5954, 2.8929, -1.0923],
        [1.1719, -0.4709, -0.1996]])
```

2.2 全0矩阵构建，通过torch.zeros()构造一个矩阵全为0，并且通过dtype设置数据类型为long。还可以通过torch.zero_()和torch.zero_like()将现有的矩阵转换为全0的矩阵。

```python
import torch

# 创建全0矩阵（4 x 3）
x_1 = torch.zeros(4, 3, dtype = torch.long)
print(x_1)
```

```python
tensor([[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]])
```

```python
>>> torch.zeros(2, 3)
tensor([[0., 0., 0.],
        [0., 0., 0.]])

>>> torch.zeros(5)
tensor([0., 0., 0., 0., 0.])
```

2.3 通过torch.tensor()直接创建并使用数据，构造一个张量

```python
import torch

x_2 = torch.tensor([1, 2, 3])
print(x_2)
```

```python
tensor([1, 2, 3])
```

```python
>>> torch.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])
tensor([[0.1000, 1.2000],
        [2.2000, 3.1000],
        [4.9000, 5.2000]])

>>> torch.tensor([0, 1])  # 数据类型推断
tensor([0, 1])

>>> torch.tensor([[0.11111, 0.222222, 0.3333333]],
                  ...
dtype = torch.float64,
        ...
device = torch.device('cuda:0'))  # 在 CUDA 设备上创建双张量
tensor([[0.1111, 0.2222, 0.3333]], dtype=torch.float64, device='cuda:0')

>>> torch.tensor(3.14159)  # 创建一个零维（标量）张量
tensor(3.1416)

>>> torch.tensor([])  # 创建一个空张量（大小为 (0,)）
tensor([])
```

2.4 通过numpy array数组进行创建

```python
import torch
import numpy as np

x_3 = np.array((1, 2, 3))
x_tensor_3 = torch.from_numpy(x_3) # 将np数组转换为张量
print(x_tensor_3)
```

```python
tensor([1, 2, 3], dtype=torch.int32)
```

2.5 基于已经存在的tensor，创建一个tensor

```python
import torch

x_4 = x.new_ones(4, 3, dtype=torch.double)
# 创建一个新的全1矩阵tensor，返回的tensor默认具有相同的torch.dtype和torch.device
# 也可以像之前的写法 x = torch.ones(4, 3, dtype=torch.double)

print(x)

x_5 = torch.randn_like(x, dtype=torch.float)
# 重置数据类型
print(x)
# 结果会有一样的size
# 获取它的维度信息
print(x_5.size())
print(x_5.shape)
```

```python
tensor([1, 2, 3], dtype=torch.int32)
tensor([[-2.2879, -0.0130,  0.2605],
        [ 0.3561,  0.7925,  0.7546],
        [-0.4286, -0.1336,  1.3082],
        [ 0.0418, -0.0130, -1.0453]])
tensor([[-2.2879, -0.0130,  0.2605],
        [ 0.3561,  0.7925,  0.7546],
        [-0.4286, -0.1336,  1.3082],
        [ 0.0418, -0.0130, -1.0453]])
torch.Size([4, 3])
torch.Size([4, 3])
```

返回的torch.Size其实是一个tuple，⽀持所有tuple的操作。可以使用索引操作取得张量的长、宽等数据维度。 

2.6 使用eye()方法，构造对角为1，其余为0的tensor

```python
import torch

x_6 = torch.eye(5, dtype=torch.float32)
print(x_6)
```

```python
tensor([[1., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0.],
        [0., 0., 1., 0., 0.],
        [0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 1.]])
```

```python
>> torch.eye(3)
tensor([[ 1.,  0.,  0.],
        [ 0.,  1.,  0.],
        [ 0.,  0.,  1.]])
```

2.7 使用arange(s, e, step)构建从s到e，步长为step的tensor

```python
import torch

x_7 = torch.arange(1, 10, 2, dtype=torch.float32)
print(x_7)
```

```python
tensor([1., 3., 5., 7., 9.])
```

```python
>>> torch.arange(5)
tensor([ 0,  1,  2,  3,  4])

>>> torch.arange(1, 4)
tensor([ 1,  2,  3])

>>> torch.arange(1, 2.5, 0.5)
tensor([ 1.0000,  1.5000,  2.0000])
```

2.8 使用linspace(s, e, steps)，构建从s到e，均匀分布成step份

```python
import torch

x_8 = torch.linspace(1, 10, 5, dtype=torch.float32)
print(x_8)
```

```python
tensor([ 1.0000,  3.2500,  5.5000,  7.7500, 10.0000])
```

```python
>>> torch.linspace(3, 10, steps=5)
tensor([  3.0000,   4.7500,   6.5000,   8.2500,  10.0000])

>>> torch.linspace(-10, 10, steps=5)
tensor([-10.,  -5.,   0.,   5.,  10.])

>>> torch.linspace(start=-10, end=10, steps=5)
tensor([-10.,  -5.,   0.,   5.,  10.])

>>> torch.linspace(start=-10, end=10, steps=1)
tensor([-10.])
```

2.9 rand 构建张量，返回一个由区间上均匀分布的随机数填充的张量：math:`[0, 1)` ，张量的形状由变量参数 :attr:`size` 定义

```python
import torch

x_9 = torch.rand(8)
print(x_9)
```

```python
>>> torch.rand(4)
tensor([ 0.5204,  0.2503,  0.3525,  0.5673])

 >>> torch.rand(2, 3)
 tensor([[ 0.8237,  0.5781,  0.6879],
         [ 0.3816,  0.7249,  0.0998]])
```

2.10 normal(mean, std, , generator=None, out=None) -> Tensor 返回从给出均值和标准差的独立正态分布中抽取的随机数张量。 

:attr:`mean` 是每个输出元素正态分布均值的张量 

:attr:`std` 是每个输出元素正态分布标准差的张量 

:attr:`mean` 和 : attr:`std` 不需要匹配，但每个张量中的元素总数需要相同

```python
>>> torch.normal(mean=torch.arange(1., 11.), std=torch.arange(1, 0, -0.1))
tensor([  1.0425,   3.5672,   2.7969,   4.2925,   4.7229,   6.2134,
        8.0505,   8.1408,   9.0563,  10.0566])

>>> torch.normal(mean=0.5, std=torch.arange(1., 6.))
tensor([-1.2793, -1.0732, -2.0687,  5.1177, -1.2303])

>>> torch.normal(mean=torch.arange(1., 6.))
tensor([ 1.1552,  2.6148,  2.6535,  5.8318,  4.2361])

>>> torch.normal(2, 3, size=(1, 4))
tensor([[-1.3987, -1.9544,  3.6048,  0.7909]])
```

2.11 randperm  从 0 返回整数的随机排列``到``n - 1

````python
>>> torch.randperm(4)
tensor([2, 1, 0, 3])
````

常见的构造Tensor的方法：

|        函数         |                       功能                        |
| :-----------------: | :-----------------------------------------------: |
|    Tensor(sizes)    |                   基础构造函数                    |
|    tensor(data)     |                  类似于np.array                   |
|     ones(sizes)     |                        全1                        |
|    zeros(sizes)     |                        全0                        |
|     eye(sizes)      |                 对角为1，其余为0                  |
|  arange(s,e,step)   |                从s到e，步长为step                 |
| linspace(s,e,steps) |              从s到e，均匀分成step份               |
|  rand/randn(sizes)  | rand是[0,1)均匀分布；randn是服从N(0，1)的正态分布 |
|  normal(mean,std)   |         正态分布(均值为mean，标准差是std)         |
|     randperm(m)     |                     随机排列                      |

1.数学运行

```python
import torch
x_1 = torch.tensor((1, 2, 3))
x_2 = torch.tensor((4, 5, 6))

print(x_1 + x_2) # 加
print(x_1 - x_2) # 减
print(x_1 * x_2) # 乘
print(x_1 / x_2) # 除
print(x_1 % x_2) # 余
print(x_2 % x_1)
```

```python
tensor([5, 7, 9])
tensor([-3, -3, -3])
tensor([ 4, 10, 18])
tensor([0.2500, 0.4000, 0.5000])
tensor([1, 2, 3])
tensor([0, 1, 0]
```

```python
#加减乘除
"""
a + b = torch.add(a, b)
a - b = torch.sub(a, b)
a * b = torch.mul(a, b)
a / b = torch.div(a, b)
"""


import torch

a = torch.rand(3, 4)
b = torch.rand(4)
print(a)
# 输出：
    tensor([[0.6232, 0.5066, 0.8479, 0.6049],
            [0.3548, 0.4675, 0.7123, 0.5700],
            [0.8737, 0.5115, 0.2106, 0.5849]])

print(b)
# 输出：
    tensor([0.3309, 0.3712, 0.0982, 0.2331])
    
# 相加
# b会被广播
print(a + b)
# 输出：
    tensor([[0.9541, 0.8778, 0.9461, 0.8380],
            [0.6857, 0.8387, 0.8105, 0.8030],
            [1.2046, 0.8827, 0.3088, 0.8179]])   
# 等价于上面相加
print(torch.add(a, b))
# 输出：
    tensor([[0.9541, 0.8778, 0.9461, 0.8380],
            [0.6857, 0.8387, 0.8105, 0.8030],
            [1.2046, 0.8827, 0.3088, 0.8179]])  

# 比较两个是否相等
print(torch.all(torch.eq(a + b, torch.add(a, b))))
# 输出：
    tensor(True)    
```



2.矩阵相乘

```python
import torch

x_3 = torch.tensor((1, 2, 3, 4, 5))
x_4 = torch.tensor((9, 8, 7, 6, 5))

x_5 = x_3.dot(x_4) 
print(x_5) # 1x9 + 2x8 + 3x7 + 4x6 + 5x5 = tensor(95)

x_6 =x_3.multiply(x_4) 
print(x_6) # [1x9 , 2x8 , 3x7 , 4x6 , 5x5] = tensor([ 9, 16, 21, 24, 25])

x_7 = x_3.mul(x_4)
print(x_7) # [1x9 , 2x8 , 3x7 , 4x6 , 5x5] = tensor([ 9, 16, 21, 24, 25])

x_8 = torch.exp(x_3) # e = 2.718
print(x_8) # tensor([  2.7183,   7.3891,  20.0855,  54.5981, 148.4132])
```

```python
"""
torch.mm(a, b) # 此方法只适用于2维
torch.matmul(a, b)
a @ b = torch.matmul(a, b) # 推荐使用此方法
"""

a = torch.full((2, 2), 3)
print(a)
# 输出
    tensor([[3., 3.],
            [3., 3.]])

b = torch.ones(2, 2)
print(b)
# 输出
    tensor([[1., 1.],
            [1., 1.]])
    
print(torch.mm(a, b))
# 输出
    tensor([[6., 6.],
            [6., 6.]])

print(torch.matmul(a, b))
# 输出
    tensor([[6., 6.],
            [6., 6.]])
    
print(a @ b)
# 输出
    tensor([[6., 6.],
            [6., 6.]])    

```

3.幂计算

```python
"""
pow, sqrt, rsqrt
"""
a = torch.full([2, 2], 3)
print(a)
# 输出
    tensor([[3., 3.],
            [3., 3.]])
    
print(a.pow(2))
# 输出
    tensor([[9., 9.],
            [9., 9.]])    
    
aa = a ** 2
print(aa)
# 输出
    tensor([[9., 9.],
            [9., 9.]]) 
    
# 平方根
print(aa.sqrt())
# 输出
    tensor([[3., 3.],
            [3., 3.]])
# 平方根    
print(aa ** (0.5))
# 输出
    tensor([[3., 3.],
            [3., 3.]])    
# 平方根    
print(aa.pow(0.5))
# 输出
    tensor([[3., 3.],
            [3., 3.]])    
    
# 平方根的倒数
print(aa.rsqrt())
# 输出
    tensor([[0.3333, 0.3333],
            [0.3333, 0.3333]])        
tensor([[3., 3.],
        [3., 3.]])
```

4.限幅

```python
a.max() # 最大值
a.min() # 最小值
a.median() # 中位数
a.clamp(10) # 将最小值限定为10
a.clamp(0, 10) # 将数据限定在[0, 10]，两边都是闭区间
```

5.近似值

```python
a.floor() # 向下取整：floor，地板
a.ceil() # 向上取整：ceil，天花板
a.trunc() # 保留整数部分：truncate，截断
a.frac() # 保留小数部分：fraction，小数
a.round() # 四舍五入：round，大约
```

## 张量属性

每个torch.Tensor都有torch.dtype, torch.device, torch.layout

### dtype

torch.dtype表示数据类型的对象torch.Tensor，Pytorch有十二种不同的数据类型：

|                           数据类型                           |                 类型                  |         构造函数         |
| :----------------------------------------------------------: | :-----------------------------------: | :----------------------: |
|                          32 位浮点                           |   `torch.float32`或者`torch.float`    |  `torch.*.FloatTensor`   |
|                          64 位浮点                           |   `torch.float64`或者`torch.double`   |  `torch.*.DoubleTensor`  |
|                          64 位复数                           |  `torch.complex64`或者`torch.cfloat`  |                          |
|                          128 位复数                          | `torch.complex128`或者`torch.cdouble` |                          |
| 16 位浮点数[1](https://pytorch.org/docs/stable/tensor_attributes.html#id3) |    `torch.float16`或者`torch.half`    |   `torch.*.HalfTensor`   |
| 16 位浮点数[2](https://pytorch.org/docs/stable/tensor_attributes.html#id4) |           `torch.bfloat16`            | `torch.*.BFloat16Tensor` |
|                      8 位整数（无符号）                      |             `torch.uint8`             |   `torch.*.ByteTensor`   |
|                      8 位整数（有符号）                      |             `torch.int8`              |   `torch.*.CharTensor`   |
|                     16 位整数（有符号）                      |    `torch.int16`或者`torch.short`     |  `torch.*.ShortTensor`   |
|                     32 位整数（有符号）                      |     `torch.int32`或者`torch.int`      |   `torch.*.IntTensor`    |
|                     64 位整数（有符号）                      |     `torch.int64`或者`torch.long`     |   `torch.*.LongTensor`   |
|                            布尔值                            |             `torch.bool`              |   `torch.*.BoolTensor`   |

1.确定torch.dtype是否为浮点数据类型，使用 is_floating_point 进行判断

2.确定torch.dtype是否为复杂数据类型，使用 is_complex 进行判断

3.当算术运算(add, sub, div, mul)的输入的dtype不同时，可通过满足以下规则的最小dtype提升

-   如果标量操作数的类型比张量操作数的类型更高(复数 > 浮点数 > 整数 > 布尔值)，可将提升为具有足够大小以容纳该类别的所有标量操作数的类型。
-   如果零维张量操作数的类别高于维操作数，可将提升为足够大小和类别的类型，以容纳该类别的所有零维张量操作数。
-   如果不存在更高类别的零维度操作数，可将提升为具有足够带线啊哦和类别的类型，以容纳所有维度的操作数



浮点标量操作数具有dtype，torch.get_default_dtype()，并且整数非布尔操作数具有dtype,torch.int64，与numpy存在不同，在却低估操作数的最小dtype时，不检查数值，尚不支持量化和复杂类型。

```python
>>> float_tensor = torch.ones(1, dtype=torch.float)
>>> double_tensor = torch.ones(1, dtype=torch.double)
>>> complex_float_tensor = torch.ones(1, dtype=torch.complex64)
>>> complex_double_tensor = torch.ones(1, dtype=torch.complex128)
>>> int_tensor = torch.ones(1, dtype=torch.int)
>>> long_tensor = torch.ones(1, dtype=torch.long)
>>> uint_tensor = torch.ones(1, dtype=torch.uint8)
>>> double_tensor = torch.ones(1, dtype=torch.double)
>>> bool_tensor = torch.ones(1, dtype=torch.bool)

# 零维张量
>>> long_zerodim = torch.tensor(1, dtype=torch.long)
>>> int_zerodim = torch.tensor(1, dtype=torch.int)

>>> torch.add(5, 5).dtype
torch.int64

# 5是一个int64，但没有比int_tensor更高的类别，因此不考虑。
>>> (int_tensor + 5).dtype
torch.int32
>>> (int_tensor + long_zerodim).dtype
torch.int32
>>> (long_tensor + int_tensor).dtype
torch.int64
>>> (bool_tensor + long_tensor).dtype
torch.int64
>>> (bool_tensor + uint_tensor).dtype
torch.uint8
>>> (float_tensor + double_tensor).dtype
torch.float64
>>> (complex_float_tensor + complex_double_tensor).dtype
torch.complex128
>>> (bool_tensor + int_tensor).dtype
torch.int32

# 由于long与float不同，所以result dtype只需要足够大
# 以保持浮点数
>>> torch.add(long_tensor, float_tensor).dtype
torch.float32
```

当指定算术运算的输出张量时，运行转换为它的dtype，以下情况除外：

-   积分输出张量不能接受浮点张量
-   布尔输出张量不饿能接受非布尔张量
-   非复数输出张量不能接受复数张量

1、允许情况

```python
>>> float_tensor *= float_tensor
>>> float_tensor *= int_tensor
>>> float_tensor *= uint_tensor
>>> float_tensor *= bool_tensor
>>> float_tensor *= double_tensor
>>> int_tensor *= long_tensor
>>> int_tensor *= uint_tensor
>>> uint_tensor *= int_tensor
```

2、不允许情况

```python
>>> int_tensor *= float_tensor
>>> bool_tensor *= int_tensor
>>> bool_tensor *= uint_tensor
>>> float_tensor *= complex_float_tensor
```



### deivice

torch.device是一个对象，表示torch.Tensor长在或将要分配相关设备属性。torch.device包含设备类型('cpu')或设备类型('cuda')可选的序号，如果设备序号不存在，此对象将始终代表设备类型的当前设备，即使在torch.cuda.set_device()被调用后也一样。

例如：torch.Tensor用device构造的'cuda'等价于'cuda: x'，x是torch.cuda.current_device()设备结果。

torch.device可通过字符串或自同构字符串和设备序号构造：

```python
# 1.通过字符串
>>> torch.device('cuda: 0')
device(type = 'cuda', index = 0)

>>>torch.device('cpu')
device(type = 'cpu')

>>>torch.device('cuda') # 缓存cuda设备
device(type = 'cuda')

# 2.通过字符串和设备序号
>>>torch.device('cuda', 0)
device(type = 'cuda', index = 0)

>>>torch.device('cpu', 0)
device(type = 'cpu', index = 0)
```

函数中的torch.device参数通常可以用字符串替换：

```python
# 接收torch.device的函数示例
>>> cuda1 = torch.device('cuda: 1')
>>> torch.randn((2, 3), device = cuda1)
```

由于遗留原因，可以通过当设备序号构建设备，该序号被视为cuda设备，这匹配Tensor。get_device()，它返回cuda张量的序号，并且不支持cpu数量。

```python
>>>torch.device(1)
device(type = 'cuda', index = 1)
```

设置设备的方法通常会接受格式正确的字符串或旧版整数设备序号，即以下都为等价：

```python
>>> torch.randn((2,3), device=torch.device('cuda:1'))
>>> torch.randn((2,3), device='cuda:1')
>>> torch.randn((2,3), device=1)  # legacy 遗留原因
```

### layout

torch.layout表示内存布局的对象，目前支持torch.strided(密度张量)并未torch.sparse_coo(稀疏coo张量)提供测试(beta)支持。

torch.strided表示密集张量，是做长远的内存布局。每个跨步张量都存在一个管理的torch.Storage，用于保存其数据，张量提供了存储的多维、跨步试图。

步幅为一个整数列表，第k个步幅表示在张量的第k维中，从1个元素到下一个元素所需要的内存跳跃，这概念使得搞笑地支持许多张量操作成为可能。

```python
>>> x = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
>>> x.stride()
(5, 1)

>>> x.t().stride()
(1, 5)
```

### memory_format

torch.memory_format是一个对象，表示torch.Tensor分配或将分配的内存格式：

1.torch.contiguous_format：张量张总或将被分配在密集的非重叠内存中，值按降序表示的步幅

2.torch.channels_last：张量张总或将被分配在密集的非重叠。由aka NUWC顺序中的值表示步幅，strides[0] > strides[2] > strides[3] > strides[1] == 1

3.torch.preserve_format：用于克隆函数，以保留输入张量的内存格式。如果输入张量分配在密集的非重叠内存中，则输出张量步幅将从输入中进行复制，否则输出步幅将随torch,contiguous_format

## 张量视图

PyTorch允许张量成为iew现有张量的以恶。视图张量与其基张量共享相同的基础数据，支持biew避免了显示数据复制，从而能够快速且内存有效的整型、切片和元素操作。

例如：查看张量t，可调用r.view(...)

```python
>>> t = torch.rand(4, 4)
>>> b = t.view(2, 8)

# `t’和‘b’共享相同的基础数据。
>>> t.storage().data_ptr() == b.storage().data_ptr()  
True

#修改视图张量也会更改基础张量。
>>> b[0][0] = 3.14
>>> t[0][0]
tensor(3.14)
```

由于试图与其张量共享基础数据，因此在试图中编辑数据，它也会反映在基张量中，通常，Pytorch操作会返回一个新张量作为输出，例如add().但是在试图操作的情况下，输出是输入张量的视图，以避免不必要的数据复制。创建视图时不会发生数据移动，视图张量知识改变了它解释相同数据的方式。



考虑连续张量可能会查收非连续张量，需注意，因为连续可能会对性能隐含影响，如transpose()

```python
>>> base = torch.tensor([[0, 1],[2, 3]])
>>> base.is_contiguous()
True

# 't'是'base'的视图。此处未发生数据移动。
>>> t = base.transpose(0, 1)  

# 视图张量可能是不连续的。
>>> t.is_contiguous()
False

# 要获得连续张量，请调用“”。continuous（）`
# 当“t”不连续时复制数据。
>>> c = t.contiguous()
```

-   [`reshape()`](https://pytorch.org/docs/stable/generated/torch.Tensor.reshape.html#torch.Tensor.reshape)，[`reshape_as()`](https://pytorch.org/docs/stable/generated/torch.Tensor.reshape_as.html#torch.Tensor.reshape_as)并且[`flatten()`](https://pytorch.org/docs/stable/generated/torch.Tensor.flatten.html#torch.Tensor.flatten)可以返回视图或新张量，用户代码不应依赖于它是否是视图。
-   [`contiguous()`](https://pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html#torch.Tensor.contiguous)如果输入张量已经是连续的，则返回**自身**，否则通过复制数据返回一个新的连续张量。



## 初始化张量

常见的情况是全零、全一或随机值， `torch`模块为所有这些提供工厂方法： 

```python
zeros = torch.zeros(2, 3) # 全0
print(zeros)

ones = torch.ones(2, 3) # 全1
print(ones)

torch.manual_seed(1729)
random = torch.rand(2, 3)
print(random)
```

输出：

```python
tensor([[0., 0., 0.],
        [0., 0., 0.]])
tensor([[1., 1., 1.],
        [1., 1., 1.]])
tensor([[0.3126, 0.3791, 0.3087],
        [0.0736, 0.4216, 0.0691]])
```

## 随机张量

 `torch.manual_seed()` 手动设置随机数生成器的种子生成随机张量。

```python
torch.manual_seed(1729)
random1 = torch.rand(2, 3)
print(random1)

random2 = torch.rand(2, 3)
print(random2)

torch.manual_seed(1729)
random3 = torch.rand(2, 3)
print(random3)

random4 = torch.rand(2, 3)
print(random4)
```

输出：

```python
tensor([[0.3126, 0.3791, 0.3087],
        [0.0736, 0.4216, 0.0691]])
tensor([[0.2332, 0.4047, 0.2162],
        [0.9927, 0.4128, 0.5938]])
tensor([[0.3126, 0.3791, 0.3087],
        [0.0736, 0.4216, 0.0691]])
tensor([[0.2332, 0.4047, 0.2162],
        [0.9927, 0.4128, 0.5938]])
```

`random1`和`random3`带有相同的值，就像`random2`和一样`random4`。手动设置 RNG 的种子会重置它，因此在大多数设置中，取决于随机数的相同计算应该提供相同的结果.

## 张量形状

通常对两个或多个张量执行操作时，它们需要具有相同的 *形状*  ，有相同数量的维度和每个维度中相同数量的单元格。使用 `torch.*_like()`方法 .

```python
x = torch.empty(2, 2, 3)
print(x.shape)
print(x)

empty_like_x = torch.empty_like(x)
print(empty_like_x.shape)
print(empty_like_x)

zeros_like_x = torch.zeros_like(x)
print(zeros_like_x.shape)
print(zeros_like_x)

ones_like_x = torch.ones_like(x)
print(ones_like_x.shape)
print(ones_like_x)

rand_like_x = torch.rand_like(x)
print(rand_like_x.shape)
print(rand_like_x)
```

输出：

```python
torch.Size([2, 2, 3])
tensor([[[0., 0., 0.],
         [0., 0., 0.]],

        [[0., 0., 0.],
         [0., 0., 0.]]])
torch.Size([2, 2, 3])
tensor([[[ 1.4759e+14,  4.5806e-41,  6.5599e+06],
         [ 3.0801e-41, -6.1473e+01,  4.5804e-41]],

        [[ 9.3640e+13,  4.5806e-41, -6.9497e+20],
         [ 4.5804e-41,  3.7715e+14,  4.5806e-41]]])
torch.Size([2, 2, 3])
tensor([[[0., 0., 0.],
         [0., 0., 0.]],

        [[0., 0., 0.],
         [0., 0., 0.]]])
torch.Size([2, 2, 3])
tensor([[[1., 1., 1.],
         [1., 1., 1.]],

        [[1., 1., 1.],
         [1., 1., 1.]]])
torch.Size([2, 2, 3])
tensor([[[0.6128, 0.1519, 0.0453],
         [0.5035, 0.9978, 0.3884]],

        [[0.6929, 0.1703, 0.1384],
         [0.4759, 0.7481, 0.0361]]])
```

在第一个输出中使用了`.shape` 来对张量的属性进行了查看，这个属性包含一个张量每个维度的范围列表

如： 例子中，`x`是一个形状为 2 x 2 x 3 的三维张量 

使用该`.shape` 属性，可以验证这些方法中的每一个都返回相同维度和范围的张量

```python
.empty_like()
.zeros_like()
.ones_like()
.rand_like()
```

 创建将覆盖的张量的最后一种方法是直接从 PyTorch 集合中指定其数据： 

```python
some_constants = torch.tensor([[3.1415926, 2.71828], [1.61803, 0.0072897]])
print(some_constants)

some_integers = torch.tensor((2, 3, 5, 7, 11, 13, 17, 19))
print(some_integers)

more_integers = torch.tensor(((2, 4, 6), [3, 6, 9]))
print(more_integers)
```

输出：

```python
tensor([[3.1416, 2.7183],
        [1.6180, 0.0073]])
tensor([ 2,  3,  5,  7, 11, 13, 17, 19])
tensor([[2, 4, 6],
        [3, 6, 9]])
```

如果python中元组或列表已有数据，可用直接使用``torch.tensor()``方法创建张量最直接, `torch.tensor()`创建数据的副本 

## 数据类型

通过以下几种方式设置张量的数据类型 

```python
a = torch.ones((2, 3), dtype=torch.int16)
print(a)

b = torch.rand((2, 3), dtype=torch.float64) * 20.
print(b)

c = b.to(torch.int32)
print(c)
```

输出：

```python
tensor([[1, 1, 1],
        [1, 1, 1]], dtype=torch.int16)
tensor([[ 0.9956,  1.4148,  5.8364],
        [11.2406, 11.2083, 11.6692]], dtype=torch.float64)
tensor([[ 0,  1,  5],
        [11, 11, 11]], dtype=torch.int32)
```

1、设置张量的基础数据类型的最简单方法是在创建时使用可选参数。 

在上面单元格的第一行中，设置`dtype=torch.int16`了张量`a`。当 print 时`a`，可以看到它充满了`1`而不是`1.`- Python 的微妙提示，即这是一个整数类型而不是浮点数。 

注意的另一件事`a`与将其保留`dtype`为默认值（32 位浮点）不同，打印张量还指定其`dtype`. 

2、设置数据类型的另一种方法是使用`.to()`方法。 

在上面的单元格中，以通常的方式创建一个随机浮点张量`b`之后，通过使用该方法`c`转换`b`为 32 位整数来创建。`.to()`请注意，它`c`包含与 相同的所有值`b`，但被截断为整数。 

3、可用的数据类型包括： 

-   `torch.bool`
-   `torch.int8`
-   `torch.uint8`
-   `torch.int16`
-   `torch.int32`
-   `torch.int64`
-   `torch.half`
-   `torch.float`
-   `torch.double`
-   `torch.bfloat`

## 数学&逻辑

1、张量与标量进行计算

```python
ones = torch.zeros(2, 2) + 1
twos = torch.ones(2, 2) * 2
threes = (torch.ones(2, 2) * 7 - 1) / 2
fours = twos ** 2
sqrt2s = twos ** 0.5

print(ones)
print(twos)
print(threes)
print(fours)
print(sqrt2s)
```

输出：

```python
tensor([[1., 1.],
        [1., 1.]])
tensor([[2., 2.],
        [2., 2.]])
tensor([[3., 3.],
        [3., 3.]])
tensor([[4., 4.],
        [4., 4.]])
tensor([[1.4142, 1.4142],
        [1.4142, 1.4142]])
```

张量和标量之间的算术运算，例如加法、减法、乘法、除法和求幂，分布在张量的每个元素上。  进行计算时，张量要具有相同数量的元素。 

 两个张量之间的类似操作:

```python
powers2 = twos ** torch.tensor([[1, 2], [3, 4]])
print(powers2)

fives = ones + fours
print(fives)

dozens = threes * fours
print(dozens)
```

输出：

```python
tensor([[ 2.,  4.],
        [ 8., 16.]])
tensor([[5., 5.],
        [5., 5.]])
tensor([[12., 12.],
        [12., 12.]])
```

## 张量广播

相同形状规则的例外是 *张量广播。* 广播是一种在形状相似的张量之间执行操作的方法. 

举例：

```python
rand = torch.rand(2, 4)
doubled = rand * (torch.ones(1, 4) * 2)

print(rand)
print(doubled)
```

输出：

```python
tensor([[0.6146, 0.5999, 0.5013, 0.9397],
        [0.8656, 0.5207, 0.6865, 0.3614]])
tensor([[1.2291, 1.1998, 1.0026, 1.8793],
        [1.7312, 1.0413, 1.3730, 0.7228]])
```

这里有什么诀窍？如何将 2x4 张量乘以 1x4 张量？

广播是一种在形状相似的张量之间执行操作的方法。在上面的示例中，单行四列张量乘以 *两行* 四列张量的两行。

这是深度学习中的一个重要操作。常见的例子是将学习权重的张量乘以*一批*输入张量，将操作分别应用于批次中的每个实例，并返回一个相同形状的张量——就像 (2, 4) * (1, 4)上面的示例返回了一个形状为 (2, 4) 的张量。

广播规则如下：

-   每个张量必须至少有一个维度: 没有空张量。
-   比较两个张量的维度大小，*从最后一个到第一个：*
    -   每个维度必须相等
    -   其中一个维度的大小必须为 1
    -   张量之一中不存在维度

相同形状的张量是微不足道的“可广播的”

 以下是一些遵守上述规则并允许广播的情况示例 

```python
a =     torch.ones(4, 3, 2)

b = a * torch.rand(   3, 2) # 第三和第二个尺寸与a相同，尺寸1不存在
print(b)

c = a * torch.rand(   3, 1) # 第三个尺寸=1，第二个尺寸等于
print(c)

d = a * torch.rand(   1, 2) # 第三个尺寸与a相同，第二个尺寸=1
print(d)
```

输出：

```python
tensor([[[0.6493, 0.2633],
         [0.4762, 0.0548],
         [0.2024, 0.5731]],

        [[0.6493, 0.2633],
         [0.4762, 0.0548],
         [0.2024, 0.5731]],

        [[0.6493, 0.2633],
         [0.4762, 0.0548],
         [0.2024, 0.5731]],

        [[0.6493, 0.2633],
         [0.4762, 0.0548],
         [0.2024, 0.5731]]])
tensor([[[0.7191, 0.7191],
         [0.4067, 0.4067],
         [0.7301, 0.7301]],

        [[0.7191, 0.7191],
         [0.4067, 0.4067],
         [0.7301, 0.7301]],

        [[0.7191, 0.7191],
         [0.4067, 0.4067],
         [0.7301, 0.7301]],

        [[0.7191, 0.7191],
         [0.4067, 0.4067],
         [0.7301, 0.7301]]])
tensor([[[0.6276, 0.7357],
         [0.6276, 0.7357],
         [0.6276, 0.7357]],

        [[0.6276, 0.7357],
         [0.6276, 0.7357],
         [0.6276, 0.7357]],

        [[0.6276, 0.7357],
         [0.6276, 0.7357],
         [0.6276, 0.7357]],

        [[0.6276, 0.7357],
         [0.6276, 0.7357],
         [0.6276, 0.7357]]])
```

观察每个张量的值：

-   创建的乘法运算`b`在`a`.
-   对于`c`，该操作在所有层和行上广播 `a`，每个 3 元素列都是相同的。
-   对于`d`，改变了它，现在每一 *行* 都是相同的，跨层和列。

## 张量数学

PyTorch 张量有超过三百个可以在它们上执行的操作 

以下是一些主要操作类别的小样本： 

```python
# 通用函数
a = torch.rand(2, 4) * 2 - 1
print('Common functions:')
print(torch.abs(a))
print(torch.ceil(a))
print(torch.floor(a))
print(torch.clamp(a, -0.5, 0.5))

# 三角函数及其逆函数
angles = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
sines = torch.sin(angles)
inverses = torch.asin(sines)
print('\nSine and arcsine:')
print(angles)
print(sines)
print(inverses)

# 按位运算
print('\nBitwise XOR:')
b = torch.tensor([1, 5, 11])
c = torch.tensor([2, 7, 10])
print(torch.bitwise_xor(b, c))

# 比较
print('\nBroadcasted, element-wise equality comparison:')
d = torch.tensor([[1., 2.], [3., 4.]])
e = torch.ones(1, 2)  # 许多比较操作支持广播！
print(torch.eq(d, e)) # 返回bool类型的张量

# 减少
print('\nReduction ops:')
print(torch.max(d))        # 返回单个元素张量
print(torch.max(d).item()) # 从返回的张量中提取值
print(torch.mean(d))       # 平均的
print(torch.std(d))        # 标准偏差
print(torch.prod(d))       # 所有数字的乘积
print(torch.unique(torch.tensor([1, 2, 1, 2, 1, 2]))) # 筛选唯一元素

# 向量和线性代数运算
v1 = torch.tensor([1., 0., 0.])         # x单位矢量
v2 = torch.tensor([0., 1., 0.])         # y单位矢量
m1 = torch.rand(2, 2)                   # 随机矩阵
m2 = torch.tensor([[3., 0.], [0., 3.]]) # 三次单位矩阵

print('\nVectors & Matrices:')
print(torch.cross(v2, v1)) # z单位矢量的负值（v1 x v2 ==- v2 x v1）
print(m1)
m3 = torch.matmul(m1, m2)
print(m3)                  # 三次m1
print(torch.svd(m3))       # 奇异值分解
```

输出：

```python
Common functions:
tensor([[0.9238, 0.5724, 0.0791, 0.2629],
        [0.1986, 0.4439, 0.6434, 0.4776]])
tensor([[-0., -0., 1., -0.],
        [-0., 1., 1., -0.]])
tensor([[-1., -1.,  0., -1.],
        [-1.,  0.,  0., -1.]])
tensor([[-0.5000, -0.5000,  0.0791, -0.2629],
        [-0.1986,  0.4439,  0.5000, -0.4776]])

Sine and arcsine:
tensor([0.0000, 0.7854, 1.5708, 2.3562])
tensor([0.0000, 0.7071, 1.0000, 0.7071])
tensor([0.0000, 0.7854, 1.5708, 0.7854])

Bitwise XOR:
tensor([3, 2, 1])

Broadcasted, element-wise equality comparison:
tensor([[ True, False],
        [False, False]])

Reduction ops:
tensor(4.)
4.0
tensor(2.5000)
tensor(1.2910)
tensor(24.)
tensor([1, 2])

Vectors & Matrices:
tensor([ 0.,  0., -1.])
tensor([[0.7375, 0.8328],
        [0.8444, 0.2941]])
tensor([[2.2125, 2.4985],
        [2.5332, 0.8822]])
torch.return_types.svd(
U=tensor([[-0.7889, -0.6145],
        [-0.6145,  0.7889]]),
S=tensor([4.1498, 1.0548]),
V=tensor([[-0.7957,  0.6056],
        [-0.6056, -0.7957]]))
```

## 修改张量

张量上的大多数二元运算将返回第三个新张量。当说（哪里和是张量）时，新张量将占据与其他张量不同的内存区域。`c = a * b` 

 1、就地更改张量 

 如果正在进行元素计算，可以丢弃中间值，这时需要一个附加下划线 ( `_`) 的版本，它将改变一个张量.

举例：

```python
a = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
print('a:')
print(a)
print(torch.sin(a))   # 这个操作在内存中创建了一个新的张量
print(a)              # a未更改

b = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
print('\nb:')
print(b)
print(torch.sin_(b))  # 注意下划线
print(b)              # b已更改
```

输出：

```python
a:
tensor([0.0000, 0.7854, 1.5708, 2.3562])
tensor([0.0000, 0.7071, 1.0000, 0.7071])
tensor([0.0000, 0.7854, 1.5708, 2.3562])

b:
tensor([0.0000, 0.7854, 1.5708, 2.3562])
tensor([0.0000, 0.7071, 1.0000, 0.7071])
tensor([0.0000, 0.7071, 1.0000, 0.7071])
```

对于算术运算，有些函数的行为类似： 

```python
a = torch.ones(2, 2)
b = torch.rand(2, 2)

print('Before:')
print(a)
print(b)
print('\nAfter adding:')
print(a.add_(b))
print(a)
print(b)
print('\nAfter multiplying')
print(b.mul_(b))
print(b)
```

输出：

```python
Before:
tensor([[1., 1.],
        [1., 1.]])
tensor([[0.3788, 0.4567],
        [0.0649, 0.6677]])

After adding:
tensor([[1.3788, 1.4567],
        [1.0649, 1.6677]])
tensor([[1.3788, 1.4567],
        [1.0649, 1.6677]])
tensor([[0.3788, 0.4567],
        [0.0649, 0.6677]])

After multiplying
tensor([[0.1435, 0.2086],
        [0.0042, 0.4459]])
tensor([[0.1435, 0.2086],
        [0.0042, 0.4459]])
```

就地算术函数是对象上的方法 ，不像许多其他函数那样`torch.Tensor`附加到模块. *调用张量是原地改变的张量。*  

2、 可以将计算结果放在现有的已分配张量中。 

有一个`out`参数可以指定一个张量来接收输出。如果`out`张量是正确的形状 和`dtype`，则无需新的内存分配就可以发生这种情况： 

```python
a = torch.rand(2, 2)
b = torch.rand(2, 2)
c = torch.zeros(2, 2)
old_id = id(c)

print(c)
d = torch.matmul(a, b, out=c)
print(c)                # c的内容已更改

assert c is d           # 测试c&d是同一个对象，不仅仅包含相等的值
assert id(c), old_id    # 确保我们的新c与旧c是同一个对象

torch.rand(2, 2, out=c) # 也适用于创造！
print(c)                # c又变了
assert id(c), old_id    # 还是同一个！
```

输出：

```python
tensor([[0., 0.],
        [0., 0.]])
tensor([[0.3653, 0.8699],
        [0.2364, 0.3604]])
tensor([[0.0776, 0.4004],
        [0.9877, 0.0352]])
```

## 复制张量

1、与 Python 中的任何对象一样，将张量分配给变量会使变量成为张量的*标签*，而不是复制它.例如： 

```python
a = torch.ones(2, 2)
b = a

a[0][1] = 561  # 改变了
print(b)       # 而且b也改变了
```

输出：

```python
tensor([[  1., 561.],
        [  1.,   1.]])
```

2、 如果想要一个单独的数据副本来处理，应该使用 `clone()`方法比较合适.

```python
a = torch.ones(2, 2)
b = a.clone()

assert b is not a      # 内存中的不同对象
print(torch.eq(a, b))  # 内容仍然相同

a[0][1] = 561          # a 改变了
print(b)               # b 没有一个改变
```

输出：

```python
tensor([[True, True],
        [True, True]])
tensor([[1., 1.],
        [1., 1.]])
```

 **使用 ``clone()`` 时需要注意一件重要的事情。** 如果原张量已启用 autograd，则克隆也将启用。 

在许多情况下，如果模型在其 `forward()` 方法中有多个计算路径，并且 *原始* 张量及其克隆都有助于模型的输出，那么要启用模型学习，需要为两个张量都打开 autograd。如果原张量启用了自动分级（如果它是一组学习权重或从涉及权重的计算中派生的，通常会启用），那么将获得想要的结果。 



3、 该 *`detach()`方法* 将张量与其计算历史分离

假设模型的`forward()`函数中执行计算，默认情况下所有内容都打开了渐变，但想在中途提取一些值以生成一些指标。在这种情况下， *不* 希望源张量的克隆副本跟踪梯度——关闭 autograd 的历史跟踪可以提高性能。为此，可以`.detach()`在原张量上使用该方法： 

```python
a = torch.rand(2, 2, requires_grad=True) # 开启梯度记录
print(a)

b = a.clone()
print(b)

c = a.detach().clone()
print(c)

print(a)
```

输出：

```python
tensor([[0.0905, 0.4485],
        [0.8740, 0.2526]], requires_grad=True)
tensor([[0.0905, 0.4485],
        [0.8740, 0.2526]], grad_fn=<CloneBackward0>)
tensor([[0.0905, 0.4485],
        [0.8740, 0.2526]])
tensor([[0.0905, 0.4485],
        [0.8740, 0.2526]], requires_grad=True)
```

解释：

-   `a`在`requires_grad=True`打开状态下创建。
-   当 print`a`时，它会通知该属性 `requires_grad=True`- 这意味着 autograd 和计算历史跟踪已打开。
-   克隆`a`并标记它`b`。当 print 时`b`，可以看到它正在跟踪它的计算历史——它继承了 `a`的 autograd 设置，并添加到了计算历史中。
-   克隆`a`到`c`，但 `detach()`先调用。
-   打印`c`，看不到计算历史，也看不到 `requires_grad=True`.



该`detach()`方法*将张量与其计算历史分离。*它说，“做接下来的任何事情，就好像 autograd 已经关闭一样。” 它在*不*改变的情况下做到这一点`a`——可以看到，当 `a`最后再次打印时，它保留了它的`requires_grad=True`属性。 

## CUDA计算

PyTorch 的主要优势之一是它在兼容 CUDA 的 Nvidia GPU 上的强大加速。(“CUDA”代表*Compute Unified Device Architecture*，这是 Nvidia 的并行计算平台)

1、使用方法检查GPU是否可用 `is_available()` 

```python
if torch.cuda.is_available():
    print('We have a GPU!')
else:
    print('Sorry, CPU only.')
    
# We have a GPU!
    
# 创建环境时
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

2、将数据传输到目标设备

**确定有一个或多个 GPU 可用，将需要计算的数据指定到目标设备上**

1.CPU 对计算机 RAM 中的数据进行计算。

2.GPU 连接了专用内存。在设备上执行计算时，必须将该计算所需的所有数据移动到该设备可访问的内存中.(通俗地说，“将数据移动到 GPU 可访问的内存”简称为“将数据移动到 GPU”)

```python
if torch.cuda.is_available():
    gpu_rand = torch.rand(2, 2, device='cuda') # 设置cuda
    print(gpu_rand)
else:
    print('Sorry, CPU only.')
```

输出：

```python
tensor([[0.3344, 0.2640],
        [0.2119, 0.0582]], device='cuda:0')
```

默认情况下，在 CPU 上创建新张量，必须使用可选 `device`参数指定何时在 GPU 上创建张量。可以看到，打印新张量时，PyTorch 会通知它在哪个设备上(如果它不在 CPU 上) 

查询 GPU 的数量`torch.cuda.device_count()`。如果有多个 GPU，则可以通过 index 指定 `device='cuda:0'`, `device='cuda:1'` 

通过创建可以传递给张量而不是字符串的设备句柄来做到这一点:

```python
if torch.cuda.is_available():
    my_device = torch.device('cuda')
else:
    my_device = torch.device('cpu')
print('Device: {}'.format(my_device))

x = torch.rand(2, 2, device=my_device)
print(x)
```

输出：

```python
Device: cuda
tensor([[0.0024, 0.6778],
        [0.2441, 0.6812]], device='cuda:0')
```

如果有一个现有的张量存在于一个设备上，可以使用该方法将其移动到另一个设备上`to()`。以下代码行在 CPU 上创建一个张量，并将其移动到您在前一个单元格中获取的任何设备句柄：

```python
y = torch.rand(2, 2)
y = y.to(my_device)
```

两个或多个张量的计算，所有张量必须在同一个设备上。无论是否有可用的 GPU 设备，以下代码都会引发运行时错误： 

```python
x = torch.rand(2, 2)
y = torch.rand(2, 2, device='gpu')
z = x + y  # 将引发异常
```

## 操纵形状

1、修改维数

更改维数的一种情况是将单个输入实例传递给模型。PyTorch 模型通常需要批量输入。 

举例： 例如，假设有一个适用于 3 x 226 x 226 图像的模型，一个具有 3 个颜色通道的 226 像素正方形。当加载和转换它时，会得到一个张量 shape 。但是，模型需要输入 shape ，其中是批次中的图像数量。那么如何制作一批呢？ `(3, 226, 226)` `(N, 3, 226, 226)` 

```python
a = torch.rand(3, 226, 226)
b = a.unsqueeze(0)

print(a.shape)
print(b.shape)
```

输出：

```python
torch.Size([3, 226, 226])
torch.Size([1, 3, 226, 226])
```

该`unsqueeze()`方法添加范围为 1 的维度。 `unsqueeze(0)`将其添加为新的第零维度 

2、挤压是什么意思？

即范围 1 的任何维度都不会改变张量中元素的数量

```python
c = torch.rand(1, 1, 1, 1, 1)
print(c)
```

假设模型的输出是每个输入的 20 元素向量。然后期望输出具有 shape ，其中是输入批次中的实例数。这意味着对于单输入批次，将获得 shape 的输出。`(N, 20)` `N` `(1, 20)` 

如果想对该输出进行一些*非批量计算——只需要一个 20 元素向量，该怎么办？* 

```python
a = torch.rand(1, 20)
print(a.shape)
print(a)

b = a.squeeze(0)
print(b.shape)
print(b)

c = torch.rand(2, 2)
print(c.shape)

d = c.squeeze(0)
print(d.shape)
```

输出：

```python
torch.Size([1, 20])
tensor([[0.1899, 0.4067, 0.1519, 0.1506, 0.9585, 0.7756, 0.8973, 0.4929, 0.2367,
         0.8194, 0.4509, 0.2690, 0.8381, 0.8207, 0.6818, 0.5057, 0.9335, 0.9769,
         0.2792, 0.3277]])
torch.Size([20])
tensor([0.1899, 0.4067, 0.1519, 0.1506, 0.9585, 0.7756, 0.8973, 0.4929, 0.2367,
        0.8194, 0.4509, 0.2690, 0.8381, 0.8207, 0.6818, 0.5057, 0.9335, 0.9769,
        0.2792, 0.3277])
torch.Size([2, 2])
torch.Size([2, 2])
```

从形状中看到二维张量现在是一维的，如果仔细观察上面单元格的输出，会发现打印`a`显示了一组“额外”的方括号 `[]`，因为有一个额外的方面. 可能只有`squeeze()`范围 1 的尺寸。

尝试将尺寸为 2 的尺寸压缩到 中`c`，并返回与开始时相同的形状。调用`squeeze()`并且`unsqueeze()`只能作用于范围 1 的维度，因为否则会改变张量中元素的数量 

```python
a =     torch.ones(4, 3, 2)

c = a * torch.rand(   3, 1) # 3rd dim = 1, 2nd dim identical to a
print(c)
```

其最终效果是在维度 0 和 2 上广播操作，导致随机的 3 x 1 张量逐元素乘以 中的每个 3 元素列`a` 

3、如果随机向量只是三元素向量怎么办？

将失去进行广播的能力，因为最终尺寸不会根据广播规则匹配`unsqueeze()`来救援： 

```python
a = torch.ones(4, 3, 2)
b = torch.rand(   3)     # 尝试乘以a*b将产生运行时错误
c = b.unsqueeze(1)       # 更改为二维张量，在末尾添加新尺寸
print(c.shape)
print(a * c)             # 广播又起作用
```

输出：

```python
torch.Size([3, 1])
tensor([[[0.1891, 0.1891],
         [0.3952, 0.3952],
         [0.9176, 0.9176]],

        [[0.1891, 0.1891],
         [0.3952, 0.3952],
         [0.9176, 0.9176]],

        [[0.1891, 0.1891],
         [0.3952, 0.3952],
         [0.9176, 0.9176]],

        [[0.1891, 0.1891],
         [0.3952, 0.3952],
         [0.9176, 0.9176]]])
```

`squeeze()`：`unsqueeze()` `squeeze_()` `unsqueeze_()` 

```python
batch_me = torch.rand(3, 226, 226)
print(batch_me.shape)
batch_me.unsqueeze_(0)
print(batch_me.shape)
```

输出：

```python
torch.Size([3, 226, 226])
torch.Size([1, 3, 226, 226])
```

4、 改变张量的形状，同时仍然保留元素的数量及其内容 

这种情况的一种情况是在模型的卷积层和模型的线性层之间的接口处——这在图像分类模型中很常见。卷积核将产生一个形状 *特征 x 宽 x 高的输出张量，* 但接下来的线性层需要一个一维输入。

`reshape()`将执行此操作，前提是请求的维度产生与输入张量相同数量的元素 

```python
output3d = torch.rand(6, 20, 20)
print(output3d.shape)

input1d = output3d.reshape(6 * 20 * 20)
print(input1d.shape)

# 也可以将其作为torch模块上的方法调用
print(torch.reshape(output3d, (6 * 20 * 20,)).shape)
```

输出：

```python
torch.Size([6, 20, 20])
torch.Size([2400])
torch.Size([2400])
```

`reshape()`将返回要更改的张量的 *视图* ,查看相同底层内存区域的单独张量对象。*这很重要：*这意味着对原张量所做的任何更改都将反映在该张量的视图中，除非`clone()`
