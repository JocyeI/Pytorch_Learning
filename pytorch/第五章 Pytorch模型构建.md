# 第五章 Pytorch模型构建

## nn.Containers

Pytorch nn 模块提供了创建和训练神经网络的各种工具，其专门为深度学习设计，核心的数据结构是Module。Module是一个抽象的概念，既可以表示神经网络中的某个层，也可以表示一个包含很多层的神经网络。 

**1、 torch.nn.parameter.Parameter(*data=None*, *requires_grad=True*)**

**一种被视为模块参数的张量。**

参数是[`Tensor`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)子类，在与 s 一起使用时具有非常特殊的属性`Module`- 当它们被分配为模块属性时，它们会自动添加到其参数列表中，并将出现在例如`parameters()`迭代器中。

分配张量没有这样的效果。这是因为人们可能想要在模型中缓存一些临时状态，例如 RNN 的最后一个隐藏状态。如果没有这样的类[`Parameter`](https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html#torch.nn.parameter.Parameter)，这些临时人员也会被注册。

参数：

-   **data** ( [*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) ) -- 参数张量。
-   **requires_grad** ( [*bool*](https://docs.python.org/3/library/functions.html#bool) *,* *optional* ) -- 如果参数需要渐变。默认值：真



**2、 torch.nn.parameter.UninitializedParameter(*requires_grad=True*, *device=None*, *dtype=None*)**

 **未初始化的参数。**

`torch.nn.Parameter` 统一化参数是数据形状仍然未知的一种特殊情况。

与不同`torch.nn.Parameter`，未初始化的参数不包含任何数据，并且尝试访问某些属性（例如它们的形状）将引发运行时错误。可以对未初始化参数执行的唯一操作是更改其数据类型，将其移动到不同的设备并将其转换为常规`torch.nn.Parameter`.

参数实现时使用的默认设备或 dtype 可以在构造过程中使用例如设置`device='cuda'`。

cls_to_become:   [`torch.nn.parameter.Parameter`](https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html#torch.nn.parameter.Parameter) 



**3、 torch.nn.parameter.UninitializedBuffer(*requires_grad=False*, *device=None*, *dtype=None*)**

**未初始化的缓冲区。**

统一缓冲区是[`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) 数据形状仍然未知的一种特殊情况。

与 a 不同[`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)，未初始化的参数不包含任何数据，并且尝试访问某些属性（例如它们的形状）将引发运行时错误。可以对未初始化参数执行的唯一操作是更改其数据类型，将其移动到不同的设备并将其转换为常规[`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor).

可以在构造期间使用例如设置缓冲区实现时使用的默认设备或 dtype `device='cuda'`。

### Module

所有神经网络模块的基类。自定义模型也应该继承这个类。模块还可以包含其他模块，允许将它们嵌套在树结构中。可以将子模块分配为常规属性：

```python
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
```

以这种方式分配的子模块将被注册，并且在您调用时也会转换其参数[`to()`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.to)。 \_\_init\_\_()必须在子类进行分配前调用父类。

1、training(bool)

布尔值，表示次模块是处于训练模式还是评估模式



2、add_module(name, module)

将子模块添加到当前模块，可以使用给定名称作为属性对模块进行访问

参数:

-   name(string)：子模块名称，可以使用给定的名称从模块中访问子模块
-   module(Module)：要添加到模块的子模块



3、apply(fn)

递归地应用fn每个子模块(由返回.children())以及self.典型用途，包括初始化模型的参数

参数：

-   fn(Module -> None)：应用于每个子模块的函数

return:

-   self

Return type：

-   Moudle

```python
@torch.no_grad()
def init_weights(m):
    print(m)
    if type(m) == nn.Linear:
        m.weight.fill_(1.0)
        print(m.weight)
net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
net.apply(init_weights)
```

```python
Linear(in_features=2, out_features=2, bias=True)
Parameter containing:
tensor([[ 1.,  1.],
        [ 1.,  1.]])
Linear(in_features=2, out_features=2, bias=True)
Parameter containing:
tensor([[ 1.,  1.],
        [ 1.,  1.]])
Sequential(
  (0): Linear(in_features=2, out_features=2, bias=True)
  (1): Linear(in_features=2, out_features=2, bias=True)
)
Sequential(
  (0): Linear(in_features=2, out_features=2, bias=True)
  (1): Linear(in_features=2, out_features=2, bias=True)
)
```

4、bfloat16()

将所有浮点参数和缓冲区转换为bfloat16数据类型，此方法就地修改模块



5、buffers(recurse=True)

返回模块缓冲区的迭代器

参数：

-   recurse(bool)：如果为True，则生成此模块和所有子模块的缓冲区。否则，只产生作为该模块直接成员的缓冲区。

yields：torch.Tensor -> moduel buffer

```python
for buf in model.buffers():
	print(type(buf), buf.size())
```

```python
<class 'torch.Tensor'> (20L,)
<class 'torch.Tensor'> (20L, 1L, 5L, 5L)
```

### Sequential

1、torch.nn.Sequential(\*args)

一个顺序容器。模块将按照它们在构造函数中传递的顺序添加到其中。或者，OrderedDict可传入模块，该forward()方法Sequential接受任何输入并将其转发到它包含的第一个模块。然后将美俄后续模块的输出顺序“链接”到输入，最后返回一个模块的输出。

1.Sequential提供的手动调用一系列模块的价值，是它运行将整个容器视为单个模块，以便对它执行转换。

2.Sequential应用于它存储的每个模块(每个模块都注册为Sequential子模块)

3.Sequential与ModuleList区别： `ModuleList`正是它听起来的样子——一个存储`Module`s 的列表！另一方面，中的层以`Sequential`级联方式连接。 

```python
model_1 = nn.Sequential(
	nn.Conv2d(1, 20, 5),
    nn.Relu(),
    nn.Conv2d(20, 64, 6),
    nn.Relu()
)

model_2 = nn.Sequential(OrderedDict(
	[
        ('conv1', nn.Conv2d(1, 20, 5)),
        ('relu1', nn.Relu()),
        ('conv2',nn.Conv2d(20, 60, 5)),
        ('relu2', nn.Relu())
    ]
))
```

### ModuleList

`toch.nn.ModuleList` 在列表中保存子模块。

`ModelList` 常规 Python 列表一样被索引，但它包含的模块已正确注册，并且对所有 [`Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)方法都是可见的。 

参数

-   modules ( *iterable* *,* *optional* ) -- 要添加的可迭代模块

举例：

```python
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
        return x
```

①：append(moduel)

将给定模块附加到列表的末尾。

参数：

-   module( [*nn.Module*](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) ) – 要附加的模块



②：extend(moduel)

将Python可迭代的模块附加到列表的末尾

参数：

-   modules( *iterable* ) -- 要附加的模块的可迭代



③：insert(index, moduel)

在列表中给定索引之前插入给定模块

参数：

-   **index** ( [*int*](https://docs.python.org/3/library/functions.html#int) ) -- 要插入的索引
-   **module** ( [*nn.Module*](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) ) – 要插入的模块



### ModuleDict

`torch.nn.ModuelDict`在字典中保存子模块

`ModuleDict`可以像普通的 Python 字典一样被索引，但它包含的模块已正确注册，并且对所有 Module方法都是可见的。

`ModuleDict`是一个**有序**的字典

-   插入顺序
-   in `update()`，合并的 `OrderedDict`, `dict`（从 Python 3.6 开始）或另一个 `ModuleDict`的参数 `update()`的顺序



请注意，`update()`对于其他无序映射类型（例如，Python `dict`3.6 版之前的 Python 普通映射）不会保留合并映射的顺序。

参数：

-    modules ( *iterable* *,* *optional* ) – (string: module) 的映射 (字典) 或类型 (string, module) 的键值对的可迭代 

举例：

```python
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.choices = nn.ModuleDict({
                'conv': nn.Conv2d(10, 10, 3),
                'pool': nn.MaxPool2d(3)
        })
        self.activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['prelu', nn.PReLU()]
        ])

    def forward(self, x, choice, act):
        x = self.choices[choice](x)
        x = self.activations[act](x)
        return x
```



1、clear()：从 ModuleDict 中删除所有项目 

2、items()： 返回 ModuleDict 键/值对的可迭代对象 

3、keys()： 返回 ModuleDict 键的可迭代对象 

4、pop(key)： 从 ModuleDict 中删除键并返回其模块 

参数：

-   key ( *string* ) – 从 ModuleDict 中弹出的键

5、update(module)

`ModuleList` 使用映射或可迭代的键值对更新，覆盖现有键.( 如果[`modules`](https://pytorch.org/docs/stable/nn.html#module-torch.nn.modules)是一个`OrderedDict`、一个[`ModuleDict`](https://pytorch.org/docs/stable/generated/torch.nn.ModuleDict.html#torch.nn.ModuleDict)或一个可迭代的键值对，则保留其中新元素的顺序。 )

参数：

-   modules ( *iterable* ) – 从 string 到 的映射（字典） ，或类型为 (string, )[`Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)的键值对的可迭代[`Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

6、values()：返回ModuleDict值的可迭代对象



### ParameterList()

`torch.nn.ParamterList(value = None)` 

 将参数保存在列表中 

[`ParameterList`](https://pytorch.org/docs/stable/generated/torch.nn.ParameterList.html#torch.nn.ParameterList)可以像常规 Python 列表一样使用，但张量`Parameter`已正确注册，并且对所有[`Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)方法都可见。

注意，构造函数，分配列表的元素， `append()`方法和`extend()` 方法将任何转换[`Tensor`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)为`Parameter` .

参数：

-   parameters ( *iterable* *,* *optional* ) -- 要添加到列表中的元素的可迭代。

举例：

```python
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.params = nn.ParameterList([nn.Parameter(torch.randn(10, 10)) for i in range(10)])

    def forward(self, x):
        # ParameterList可以作为可迭代的，也可以使用int进行索引
        for i, p in enumerate(self.params):
            x = self.params[i // 2].mm(x) + p.mm(x)
        return x
```

### ParameterDict()

 `torch.nn.``ParameterDict`(*parameters=None*)，在字典中保存参数。

ParameterDict 可以像普通的 Python 字典一样被索引，但它包含的参数已正确注册，并且对所有 Module 方法都是可见的。其他对象的处理方式与常规 Python 字典一样

[`ParameterDict`](https://pytorch.org/docs/stable/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict)是一个**有序**字典。 [`update()`](https://pytorch.org/docs/stable/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.update)与其他无序映射类型（例如，Python 的 plain `dict`）不保留合并映射的顺序。另一方面，`OrderedDict`或另一个[`ParameterDict`](https://pytorch.org/docs/stable/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict) 将保留他们的顺序。

注意，构造函数，分配字典的元素和 [`update()`](https://pytorch.org/docs/stable/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.update)方法会将任何转换[`Tensor`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)为 `Parameter` .

参数：

-   values( *iterable* *,* *optional* ) – (string : Any) 的映射 (字典) 或类型 (string, Any) 的键值对的可迭代

举例：

```python
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.params = nn.ParameterDict({
                'left': nn.Parameter(torch.randn(5, 10)),
                'right': nn.Parameter(torch.randn(5, 10))
        })

    def forward(self, x, choice):
        x = self.params[choice].mm(x)
        return x
```

`torch.nn.Module` 是 PyTorch 基类，旨在封装特定于 PyTorch 模型及其组件的行为的一个重要行为是注册参数。如果特定子类具有学习权重，则这些权重表示为 的实例。该类是 的子类，具有特殊行为，当它们被分配为 a 的属性时，它们被添加到该模块参数的列表中。可以通过类上的方法访问这些参数。`torch.nn.Module`  `torch.nn.Parameter`  `torch.Tensor` 

举例：定义一个非常简单的模型，它有两个线性层和一个激活函数。将创建它的一个实例，并要求它报告其参数： 

```python
import torch

class TinyModel(torch.nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()

        self.linear1 = torch.nn.Linear(100, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 10)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

tinymodel = TinyModel()

print('The model:')
print(tinymodel)

print('\n\nJust one layer:')
print(tinymodel.linear2)

print('\n\nModel params:')
for param in tinymodel.parameters():
    print(param)

print('\n\nLayer params:')
for param in tinymodel.linear2.parameters():
    print(param)
```

输出：

```python
The model:
TinyModel(
  (linear1): Linear(in_features=100, out_features=200, bias=True)
  (activation): ReLU()
  (linear2): Linear(in_features=200, out_features=10, bias=True)
  (softmax): Softmax(dim=None)
)


Just one layer:
Linear(in_features=200, out_features=10, bias=True)


Model params:
Parameter containing:
tensor([[-0.0514, -0.0764, -0.0733,  ...,  0.0595,  0.0263, -0.0798],
        [ 0.0035,  0.0956,  0.0543,  ..., -0.0105,  0.0529,  0.0117],
        [-0.0856, -0.0187,  0.0400,  ...,  0.0135,  0.0889, -0.0475],
        ...,
        [-0.0485, -0.0758, -0.0999,  ..., -0.0008,  0.0927,  0.0511],
        [-0.0820,  0.0185,  0.0672,  ..., -0.0304, -0.0375,  0.0816],
        [ 0.0204, -0.0969,  0.0963,  ...,  0.0097, -0.0834,  0.0518]],
       requires_grad=True)
Parameter containing:
tensor([ 0.0188, -0.0167,  0.0998, -0.0606, -0.0622, -0.0551, -0.0882, -0.0570,
         0.0557,  0.0108,  0.0719,  0.0645, -0.0900, -0.0401, -0.0761,  0.0741,
        -0.0639, -0.0289,  0.0943, -0.0619,  0.0664, -0.0766,  0.0229, -0.0305,
         0.0821, -0.0445, -0.0289,  0.0240, -0.0784,  0.0095,  0.0580,  0.0142,
         0.0338, -0.0203, -0.0201,  0.0928,  0.0780,  0.0795,  0.0604, -0.0684,
         0.0344, -0.0814,  0.0706, -0.0852,  0.0998, -0.0360,  0.0799, -0.0193,
        -0.0476, -0.0314,  0.0256,  0.0660, -0.0509,  0.0157,  0.0620, -0.0188,
         0.0403, -0.0052,  0.0842,  0.0565, -0.0232, -0.0277, -0.0675, -0.0752,
        -0.0930,  0.0936, -0.0643,  0.0311, -0.0762,  0.0898,  0.0158, -0.0815,
         0.0430, -0.0822,  0.0900,  0.0004,  0.0217, -0.0862,  0.0712,  0.0316,
        -0.0441, -0.0449, -0.0761, -0.0981, -0.0163,  0.0296,  0.0563, -0.0488,
        -0.0345, -0.0608,  0.0814, -0.0827, -0.0148, -0.0271,  0.0168,  0.0367,
        -0.0459,  0.0911, -0.0016,  0.0760, -0.0448,  0.0572, -0.0643,  0.0220,
         0.0069, -0.0827,  0.0192,  0.0659, -0.0365, -0.0520,  0.0154, -0.0687,
         0.0373,  0.0861,  0.0627,  0.0404,  0.0801, -0.0129, -0.0254,  0.0366,
        -0.0921,  0.0810,  0.0463, -0.0109, -0.0969,  0.0218,  0.0850,  0.0329,
        -0.0408, -0.0730,  0.0958,  0.0021,  0.0116, -0.0206,  0.0356,  0.0706,
        -0.0839, -0.0160,  0.0009, -0.0636,  0.0976,  0.0669,  0.0555,  0.0840,
        -0.0282, -0.0482,  0.0863, -0.0301, -0.0077, -0.0931,  0.0784,  0.0445,
         0.0193,  0.0838, -0.0352,  0.0345,  0.0057,  0.0355, -0.0286,  0.0751,
         0.0846,  0.0546,  0.0805, -0.0567, -0.0665,  0.0679,  0.0894, -0.0085,
         0.0068, -0.0447,  0.0170, -0.0145, -0.0250,  0.0480, -0.0444, -0.0412,
        -0.0202,  0.0601, -0.0771, -0.0073, -0.0825,  0.0395,  0.0743,  0.0152,
         0.0423, -0.0114,  0.0956,  0.0317,  0.0129, -0.0883, -0.0472, -0.0751,
         0.0152, -0.0444, -0.0324,  0.0372, -0.0752, -0.0190,  0.0903,  0.0984],
       requires_grad=True)
Parameter containing:
tensor([[ 0.0655, -0.0179,  0.0325,  ...,  0.0686, -0.0180, -0.0207],
        [ 0.0174,  0.0516,  0.0263,  ...,  0.0317,  0.0054, -0.0699],
        [ 0.0330,  0.0065, -0.0443,  ...,  0.0062,  0.0029, -0.0529],
        ...,
        [-0.0662,  0.0645,  0.0205,  ..., -0.0454, -0.0381, -0.0106],
        [ 0.0093, -0.0702,  0.0588,  ..., -0.0370,  0.0084,  0.0025],
        [ 0.0488, -0.0273, -0.0034,  ...,  0.0181, -0.0115,  0.0390]],
       requires_grad=True)
Parameter containing:
tensor([ 0.0595,  0.0499, -0.0241,  0.0484, -0.0055,  0.0426, -0.0629,  0.0329,
        -0.0481,  0.0634], requires_grad=True)


Layer params:
Parameter containing:
tensor([[ 0.0655, -0.0179,  0.0325,  ...,  0.0686, -0.0180, -0.0207],
        [ 0.0174,  0.0516,  0.0263,  ...,  0.0317,  0.0054, -0.0699],
        [ 0.0330,  0.0065, -0.0443,  ...,  0.0062,  0.0029, -0.0529],
        ...,
        [-0.0662,  0.0645,  0.0205,  ..., -0.0454, -0.0381, -0.0106],
        [ 0.0093, -0.0702,  0.0588,  ..., -0.0370,  0.0084,  0.0025],
        [ 0.0488, -0.0273, -0.0034,  ...,  0.0181, -0.0115,  0.0390]],
       requires_grad=True)
Parameter containing:
tensor([ 0.0595,  0.0499, -0.0241,  0.0484, -0.0055,  0.0426, -0.0629,  0.0329,
        -0.0481,  0.0634], requires_grad=True)
```

PyTorch模型的基本结构：有一个定义模型的层和其他组件的方法，以及一个完成计算的方法。可以打印模型或其任何子模块来了解其结构。`__init__()` `forward()` 

## 图层类型

### 线性层

神经网络层的最基本类型是线性或全连接层。在这个图层中，每个输入都会将图层的每个输出影响到由图层权重指定的程度。如果模型具有 *m* 个输入和 *n* 个输出，则权重将为 *m* x *n* 个矩阵。例如： 

```python
lin = torch.nn.Linear(3, 2)
x = torch.rand(1, 3)
print('Input:')
print(x)

print('\n\nWeight and Bias parameters:')
for param in lin.parameters():
    print(param)

y = lin(x)
print('\n\nOutput:')
print(y)
```

输出：

```python
Input:
tensor([[0.1391, 0.3932, 0.1566]])


Weight and Bias parameters:
Parameter containing:
tensor([[ 0.4025, -0.3434,  0.2328],
        [ 0.5428, -0.2218, -0.1013]], requires_grad=True)
Parameter containing:
tensor([ 0.0989, -0.5101], requires_grad=True)


Output:
tensor([[ 0.0563, -0.5376]], grad_fn=<AddmmBackward0>)
```

如果对线性层的权值进行矩阵乘法，并添加了偏差，那么会到输出的结果向量。另一个需要注意的重要功能：当用检查图层的权重时，它会将自己报告为(这是 的子类)，并让它正在使用自动分级来跟踪梯度。这是与不同的默认行为。`lin.weight` `Parameter` `Tensor` `Parameter` 

线性层广泛用于深度学习模型。它们的最常见位置之一是在分类器模型中，这些模型通常在末尾有一个或多个线性层，其中最后一层将具有*n*个输出，其中*n*是分类器地址的类数。

### 卷积层

构建卷积层是为了处理具有高度空间相关性的数据。它们在计算机视觉中非常常用，在那里它们检测特征的紧密分组，这些特征组合成更高级的特征。它们也会在其他上下文中弹出：

例如，在NLP应用程序中，单词的直接上下文（即序列中附近的其他单词）可以影响句子的含义。  在之前的一段视频中看到了卷积层在 LeNet5 中的运行情况： 

```python
import torch.functional as F


class LeNet(torch.nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # 1个输入图像通道（黑白），6个输出通道，5x5平方卷积
        # 内核
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        # 仿射运算：y = Wx + b
        self.fc1 = torch.nn.Linear(16 * 6 * 6, 120)  # 6*6 从图像尺寸
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        # （2， 2） 窗口的最大池化
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 如果大小是正方形，则只能指定单个数字
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 除批次维度之外的所有维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
```

分解一下这个模型的卷积层中发生了什么。开头为 ：`conv1` 

-   LeNet5旨在接收1x32x32的黑白图像。**卷积层构造函数的第一个参数是输入通道的数量。**在这里它是1，如果构建这个模型来查看3色通道，它将是3个。
-   卷积层就像一个窗口，扫描图像，寻找它识别的模式。这些模式称为*特征，*卷积层的参数之一是希望它学习的特征数量。**这是构造函数的第二个参数是输出要素的数量。**在这里，要求图层学习 6 个特征。
-   就在上面，把卷积层比作一个窗口 - 但是窗口有多大？**第三个参数是窗口或内核大小。**在这里，“5”表示选择了5x5内核。（如果想要一个高度与宽度不同的内核，可以为此参数指定一个元组： 例如，得到一个3x5卷积核。`(3, 5)`

卷积层的输出是激活映射，输入张量中特征存在的空间表示。 将给一个6x28x28的输出张量：

1、6是通道的数量

2、28 是高度和宽度

然后，通过 ReLU 激活函数传递卷积的输出，然后通过最大池化层传递卷积。最大池化图层在激活地图中获取彼此靠近的要素，并将它们组合在一起。它通过减小张量，将输出中的每组2x2单元格合并为单个单元格，并为该单元格分配进入其中的4个单元格的最大值来实现此目的。这提供了较低分辨率版本的激活图，尺寸为6x14x14。

下一个卷积层，期望6个输入通道（对应于第一层寻求的6个特征），有16个输出通道和一个3x3内核。它输出了一个 16x12x12 的激活映射，该映射再次通过最大池化层减少到 16x6x6。在将此输出传递到线性层之前，它被重塑为 16 * 6 * 6 = 576 个元素的向量，以供下一层使用。`conv2`

有用于寻址 1D、2D 和 3D 张量的卷积层。Conv 层构造函数还有更多可选参数，包括输入中的步幅长度(例如，仅扫描一次或每三个位置)、填充(因此可以扫描到输入的边缘)等。

### 循环层

递归神经网络(或RNN)用于顺序数据，从科学仪器到自然语言句子到DNA核苷酸的时间序列测量。RNN通过维护一个*隐藏的状态*来做到这一点，该状态充当迄今为止在序列中看到的内容的一种记忆。RNN 层的内部结构或其变体，LSTM(长短期记忆)和 GRU(门控循环单元)：适度复杂.

展示基于 LSTM 的词性标记器(一种分类器，告诉您单词是否是名词， 动词等)

```python
class LSTMTagger(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)

        # LSTM 将词嵌入作为输入，并输出隐藏状态
        # 具有维度hidden_dim。
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim)

        # 从隐藏状态空间映射到标记空间的线性层
        self.hidden2tag = torch.nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
```

构造函数有四个参数：

-   `vocab_size`是输入词汇表中的单词数。每个单词都是-维空间中的一个热向量或单位向量
-   `tagset_size`是输出集中的标记数
-   `embedding_dim`是词汇的*嵌入*空间的大小,嵌入将词汇映射到低维空间中,其中具有相似含义的单词在空间中紧密地结合在一起
-   `hidden_dim`是 LSTM 内存的大小

输入将是一个句子，其中的单词表示为单热向量的索引。然后嵌入层会将这些映射到维空间。LSTM 获取此嵌入序列并对其进行迭代，并字段长度的输出向量。最后一个线性层充当分类器;应用于最后一层的输出会将输出转换为一组规范化的估计概率，给定单词映射到给定标记。`embedding_dim` `hidden_dim` `log_softmax()` ==pytorch.org [序列模型和 LSTM 网络](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)== 

### 数据层

数据操作层，**最大池化**(及其孪生，最小池化)通过组合单元并将输入单元的最大值分配给输出单元来减少张量，例如： 

**Max pooling**

```python

my_tensor = torch.rand(1, 6, 6)
print(my_tensor)

maxpool_layer = torch.nn.MaxPool2d(3)
print(maxpool_layer(my_tensor))
```

输出：

```python
tensor([[[0.0379, 0.3152, 0.1242, 0.8418, 0.9264, 0.8736],
         [0.3667, 0.6202, 0.2350, 0.6264, 0.7133, 0.3880],
         [0.4206, 0.5514, 0.6659, 0.8048, 0.9972, 0.8409],
         [0.6386, 0.1197, 0.5559, 0.1287, 0.1476, 0.0639],
         [0.2463, 0.4152, 0.5455, 0.5225, 0.0562, 0.5634],
         [0.0470, 0.9231, 0.1421, 0.5820, 0.7826, 0.9549]]])
tensor([[[0.6659, 0.9972],
         [0.9231, 0.9549]]])
```

仔细查看上面的值，会发现最大池化输出中的每个值都是 6x6 输入的每个象限的最大值。 

**Normalization layers**

**归一化层在**将一个层的输出馈送到另一个层之前重新居中并对其进行归一化。将中间张量居中并缩放具有许多有益的效果，例如允许使用更高的学习速率而不会使梯度爆炸/消失。 

```python
my_tensor = torch.rand(1, 4, 4) * 20 + 5
print(my_tensor)

print(my_tensor.mean())

norm_layer = torch.nn.BatchNorm1d(4)
normed_tensor = norm_layer(my_tensor)
print(normed_tensor)

print(normed_tensor.mean())
```

```python
tensor([[[ 6.5077,  9.0456, 16.6335, 22.1080],
         [10.8637,  7.5800,  6.3049, 19.1949],
         [ 8.3028,  7.4120, 23.4289,  5.2189],
         [24.2918,  7.0045, 21.6432, 10.4632]]])
tensor(12.8752)
tensor([[[-1.1439, -0.7330,  0.4953,  1.3816],
         [-0.0243, -0.6781, -0.9319,  1.6343],
         [-0.3866, -0.5101,  1.7109, -0.8142],
         [ 1.1592, -1.2149,  0.7955, -0.7399]]],
       grad_fn=<NativeBatchNormBackward0>)
tensor(2.6077e-08, grad_fn=<MeanBackward0>)
```

运行上面的单元格，添加了一个大的比例因子和偏移量到输入张量;应该看到输入张量在 15 附近的某个地方。在通过归一化层运行它之后，可以看到值较小，并且分组在零附近 - 实际上，平均值应该非常小（>1e-8）。`mean()`

这是有益的，因为许多激活函数的最大梯度接近0，但有时由于输入的梯度消失或爆炸，使它们远离零。将数据集中在最陡峭的梯度区域周围往往意味着更快，更好的学习和更高的可行学习率。

 **Dropout layers** 

**正则层** 是一种工具，用于鼓励模型中*的稀疏表示* ，也就是说，推动它使用较少的数据进行推理。工作原理是在训练期间随机设置输入张量的部分，正则层总是关闭以进行推理。这将强制模型针对此屏蔽或简化的数据集进行学习。例如：

```python
my_tensor = torch.rand(1, 4, 4)

dropout = torch.nn.Dropout(p=0.4)
print(dropout(my_tensor))
print(dropout(my_tensor))
```

输出：

```python
tensor([[[0.8258, 0.0000, 1.2489, 0.9706],
         [0.0000, 0.0000, 0.6511, 1.2401],
         [0.0000, 0.0000, 0.0000, 0.3941],
         [0.0000, 0.9308, 0.4133, 0.0000]]])
tensor([[[0.8258, 0.5030, 0.0000, 0.0000],
         [0.2400, 0.0000, 0.6511, 0.0000],
         [0.4454, 0.3431, 1.1566, 0.0000],
         [0.2541, 0.9308, 0.4133, 0.1743]]])
```

在上面，可以看到压差对样本张量的影响。可以使用可选参数来设置单个权重下降的概率;如果不这样做，则默认为 0.5。`p` 

## 激活函数

激活函数使深度学习成为可能。神经网络实际上是一个程序具有许多参数模拟数学函数，如果所做的只是重复地按层权重的多个张量，只能模拟线性函数，此外，拥有多层是没有意义的，因为整个网络将减少可以减少到单个矩阵乘法。在层之间插入非线性激活函数是允许深度学习模型模拟任何函数而不仅仅是线性函数的原因。

`torch.nn.Module`具有封装所有主要激活函数的对象，包括ReLU及其许多变体，谭，硬坦，sigmoid等。它还包括其他功能，例如Softmax，这些功能在模型的输出阶段最有用。

## 损失函数

损失函数告诉模型的预测与正确答案的距离。PyTorch 包含各种损失函数，包括常见的 MSE(均方误差 = L2 范数)、交叉熵损失和负似然损失(对分类器有用)等。 

