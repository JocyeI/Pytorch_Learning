# 第十三章 Pytorch Module



## nn

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

torch.nn.Module

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