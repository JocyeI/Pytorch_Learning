# 第十章 Pytorch Utils

## torch.utils.data

Pytorch数据加载实用程序的核心是``torch.utils.data.DataLoader``类，它表示一个Python可迭代的数据集，支持map样式的数据和可迭代的数据，也可用自定义数据加载顺序，批次，单进程，多进程数据加载，自动内存固定等.

 [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)的构造函数参数配置具有签名：

```python
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False)
```

### 数据集类型

[`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)构造函数最重要的参数：它表示从加载数据的数据集对象，Pytorch支持两种不同类型的数据集：

-   map数据集
-   iter可迭代数据集

### map数据集

map数据集表示从(可能是非整数)索引/键到数据样本的映射，例如：当使用访问时，此类数据集可用从磁盘上的文件读取图像及对应的标签(dataset[inx])。

### iter数据集

iter可迭代数据集实现了 [`IterableDataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset) 子类的实例，表示可迭代的数据样本，这种类型的数据姐特别适合用于随机读取成本高昂甚至不可能的情况，并且批大小取决于提取的数据(\_\_iter\_\_())，例如：调用时，数据集可用从数据库、远程服务器甚至实时生成的日志读取的数据流(iter(dataset)).

**注意：** 使用多进程数据加载的可迭代数据集时，在每个工作进程上复制相同的数据集对象，因此必须以不同的方式配置副本，以避免数据产生重复的情况.

### 数据采样

[`torch.utils.data.Sampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler)，对于可迭代的数据集，数据加载顺序完全由用户定义对可迭代对象控制，这更容易地实现快速读取和动态批量大小(例如：通过每次产生批处理样本).  [`torch.utils.data.Sampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler)类用于指定数据加载中使用的索引/键的顺序，它们表示指向数据集的索引指向的可迭代对象，例如：在随机梯度(SGD)环境中，[`torch.utils.data.Sampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler)可用随机排列索引列表并一次产生每个索引，或者为小批量SGD产生少量索引。

顺序采样器或随机采样器将基于[`torch.utils.data.Sampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler)的参数自动构造。或者，用户可以使用该参数指定一个自定义 [`Sampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler) 对象，该对象每次都会生成要提取的下一个索引/键。

可以传递一次生成批处理索引列表的自定义[`torch.utils.data.Sampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler)作为参数。还可以通过参数启用自动批处理。

### 批处理

加载批处理和非批处理数据，DataLOader通过参数(具有默认函数)自动将单个图区的数据样本整理成批

### 自动批处理

自动批处理(默认)，这是最常见的情况，对于获取一小批数据并将它们整理成批处理样本，即包含张力，其中一个维度就是批处理的维度(通常指第一个维度).

当(默认)不是时，数据加载器产生批处理的样本而不是单个样本，参数用于指定数据加载程序如何批量获取数据集密，对于map样式的数据集，用户可用指定，这样一次生成的一个键列表

**注意：** 从具有多处理的可迭代样式数据集中提取时，该参数将删除每个工作线程的数据集副本的最后一个非完整批次的数据.

使用sample中的索引获取样本列表后，作为参数传递的函数将用于，将样本列表整理为批次，在这种情况下，从map样式数据加载大致等同于：

````python
for indices in batch_sampler:
    yield collate_fn([dataset[i] for i in indices])
````

可迭代样式的数据集加载大致等同于：

```python
dataset_iter = iter(dataset)
for indices in batch_sampler:
    yield collate_fn([next(dataset_iter) for _ in indices])
```

自定义可用于自定义排序规则，例如，将顺序数据填充到批处理的最大长度 

### 禁用自动批处理

在某些情况下，用户可能希望在数据集代码中手动处理批处理，或者只是加载单个示例。例如，直接加载批处理数据(例如，从数据库批量读取或读取连续的内存块)可能更便宜，或者批处理大小取决于数据，或者程序被设计为处理单个样本。在这些情况下，最好不要使用自动批处理(其中用于整理示例)，而是让数据加载程序直接返回对象的每个成员。

 当和均为时(默认值已为 )则禁用自动批处理。从中获得的每个样本都使用作为参数传递的函数进行处理 

**当禁用自动批处理时**，默认值只是将 NumPy 数组转换为 PyTorch 张量，并保持其他所有内容不变。`collate_fn`

在这种情况下，从map样式数据集加载大致等同于：

```python
for index in sampler:
    yield collate_fn(dataset[index])
```

从可迭代样式数据加载大致等同于：

```python
for data in iter(dataset):
    yield collate_fn(data)
```

### 使用collate_fn

启用或禁用自动批处理时，使用略有不同。

**禁用自动批处理时**，将对每个单独的数据样本进行调用，并从数据加载器迭代器生成输出。在这种情况下，默认值只是转换 PyTorch 张量中的 NumPy 数组。`collate_fn` `collate_fn`

**启用自动批处理后**，每次都会使用数据样本列表进行调用。预计它会将输入样本整理成一个批处理，以便从数据加载器迭代器生成。例如，如果每个数据样本由一个3通道图像和一个整数类标签组成，即数据集的每个元素都返回一个元组，则默认将此类元组的列表整理成批处理图像张量和批处理类标签张量的单个元组。

特别是，默认值具有以下属性：`(image, class_index)` `collate_fn` `collate_fn`

-   它始终将新维度作为批次维度。
-   它会自动将 NumPy 数组和蟒蛇数值转换为 PyTorch 张量。
-   它保留数据结构，例如，如果每个样本都是字典，它会输出具有相同键集但批量张量作为值的字典(如果值无法转换为张量，则将其列出)。对于 s等也是如此。`list` `tuple` `namedtuple`

用户可以使用自定义来实现自定义批处理，例如，沿第一个维度以外的维度进行整理，填充各种长度的序列，或添加对自定义数据类型的支持。

### 单多进程加载

默认情况下 [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) 使用单进程数据加载。

在 Python 进程中，[全局解释器锁 （GIL）](https://wiki.python.org/moin/GlobalInterpreterLock) 会阻止跨线程真正完全并行化 Python 代码。为了避免在数据加载时阻塞计算代码，PyTorch 提供了一个简单的开关，只需将参数设置为正整数即可执行多进程数据加载。`num_workers`

#### 单进程数据加载（默认）

在此模式下，数据提取是在初始化 [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) 的同一过程中完成的。因此，数据加载可能会阻止计算。但是，当用于在进程之间共享数据的资源（例如，共享内存、文件描述符）有限时，或者当整个数据集很小并且可以完全加载到内存中时，此模式可能是首选。此外，单进程加载通常显示更具可读性的错误跟踪，因此对于调试非常有用。

#### 多进程数据加载

将参数设置为正整数将打开具有指定数量的加载程序工作进程的多进程数据加载。`num_workers`

**注意：**

经过多次迭代后，加载程序工作进程将消耗与父进程中从工作进程访问的所有 Python 对象的父进程相同的 CPU 内存量。如果 Dataset 包含大量数据（例如，您在 Dataset 构造时加载了一个非常大的文件名列表）和/或您正在使用大量工作线程（总体内存使用量为 ），则这可能会有问题。最简单的解决方法是将Python对象替换为非引用表示形式，例如 Pandas, Numpy or PyArrow 对象 

### 特定行为

#### 特定于平台的行为

由于工作线程依赖于 Python  [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing) 与 Unix 相比，工作线程在 Windows 上的启动行为是不同的。

-   在 Unix 上， 是默认的 [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing) 启动方法。使用 ，子工作线程通常可以直接通过克隆的地址空间访问 和 Python 参数函数。`fork()``fork()``dataset`
-   在视窗或 MacOS 上， 是默认 [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing) 启动方法。使用 ，将启动另一个运行主脚本的解释器，然后是通过 [`pickle`](https://docs.python.org/3/library/pickle.html#module-pickle) 序列化接收 的内部 worker 函数和其他参数。`spawn()` `spawn()` `dataset` `collate_fn`

此单独的序列化意味着您应采取两个步骤来确保在使用多进程数据加载时与 Windows 兼容：

-   将大多数主脚本的代码包装在 块中，以确保在每个工作进程启动时它不会再次运行（很可能生成错误）。您可以将数据集和 [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) 实例创建逻辑放置在此处，因为它不需要在工作线程中重新执行。`if __name__ == '__main__':`
-   确保任何自定义 或代码都声明为检查之外的顶级定义。这可确保它们在工作进程中可用。（这是必需的，因为函数仅作为引用进行腌制，而不是 。`collate_fn` `worker_init_fn` `dataset` `__main__` `bytecode`



#### 多进程数据加载中的随机性

默认情况下，每个工作线程的 PyTorch 种子将设置为 ，其中 是由主进程使用其 RNG（从而强制使用 RNG 状态）或指定的 长进程生成的。但是，其他库的种子可能会在初始化工作线程时重复，从而导致每个工作线程返回相同的随机数。

`base_seed + worker_id` `base_seed` `generator`

可以使用 [`torch.utils.data.get_worker_info().`](https://pytorch.org/docs/stable/data.html#torch.utils.data.get_worker_info)seed 或 torch.initial_seed[()](https://pytorch.org/docs/stable/generated/torch.initial_seed.html#torch.initial_seed) 访问每个工作线程的 PyTorch 种子集，并在加载数据之前使用它来设定其他库的种子。`worker_init_fn`

### 内存固定

当主机到 GPU 副本源自固定(页面锁定)内存时，它们的速度要快得多，对于数据加载，传递到 [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) 将自动将提取的数据张量放在固定内存中，从而可以更快地将数据传输到启用了 CUDA 的 GPU。`pin_memory=True`

默认内存固定逻辑仅识别张量和映射以及包含张量的可迭代对象。默认情况下，如果固定逻辑看到一个批处理是自定义类型(如果有返回自定义批处理类型的批处理，则会发生这种情况)，或者如果批处理的每个元素都是自定义类型，则固定逻辑将无法识别它们，并且它将返回该批处理(或这些元素)，而不会固定内存。

若要为自定义批处理或数据类型启用内存固定，在自定义类型上定义一个方法。`collate_fn` `pin_memory()`

```python
class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)
        self.tgt = torch.stack(transposed_data[1], 0)

    # 自定义类型上的自定义内存固定方法
    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self

def collate_wrapper(batch):
    return SimpleCustomBatch(batch)

inps = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
tgts = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
dataset = TensorDataset(inps, tgts)

loader = DataLoader(dataset, batch_size=2, collate_fn=collate_wrapper,
                    pin_memory=True)

for batch_ndx, sample in enumerate(loader):
    print(sample.inp.is_pinned())
    print(sample.tgt.is_pinned())
```



### DataLoader

数据加载器。组合数据集和采样器，并提供对给定数据集的可迭代。[`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) 支持map样式和可迭代样式的数据集，具有单进程或多进程加载、自定义加载顺序以及可选的自动批处理(排序规则)和内存固定功能


```python
torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=None, sampler=None, batch_sampler=None, 
                            num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, 
                            worker_init_fn=None, multiprocessing_context=None, generator=None, *, 
                            prefetch_factor=2, persistent_workers=False, pin_memory_device='')
```

参数：

-   **dataset** ([*Dataset*](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset))：从中加载数据的数据集
-   **batch_size** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* *optional*)：每批要加载多少个样本（默认值：）.1
-   **shuffle** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*)：设置在每个 epoch 时是否进行重新洗牌数据（默认值：True False）
-   **sampler** ([*Sampler*](https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler) *or* Iterable, *optional*) ：定义从数据集中抽取样本的策略。可以是任何已实施的。如果指定，则不能指定
-   **batch_sampler** ([*Sampler*](https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler) *or* Iterable, optional)：一次返回一批索引
-   **num_workers** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* *optional*)：用于数据加载的子进程数。表示数据将在主进程中加载
-   **collate_fn** (callable, *optional*)：合并样本列表以形成小批量张量。从地图样式数据集中使用批量加载时使用
-   **pin_memory** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*)：如果是数据加载器将在返回之前将张量复制到设备/CUDA 固定内存中。如果数据元素是自定义类型，或者返回的批次是自定义类型
-   **drop_last** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*)：设置为丢弃最后一个不完整的批次，如果数据集大小不能被批次大小整除。如果数据集的大小不能被批大小整除，那么最后一批将更小
-   **timeout** (numeric, *optional*)：如果为正，则从工人那里收集批次的超时值。应始终为非负数
-   **worker_init_fn** (callable, *optional*)：如果不是，这将在每个工作子进程上调用，并在播种之后和数据加载之前以工作人员 ID（一个 int in ）作为输入
-   **generator** ([*torch.Generator*](https://pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)*,* *optional*)：如果不是，则 RandomSampler 将使用此 RNG 来生成随机索引和多处理以生成 base_seed 用于工作人员
-   **prefetch_factor** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* *optional*, *keyword-only arg*)：每个工作人员预先加载的批次数。意味着将在所有工作人员中预取总共 2 * num_workers 个批次
-   **persistent_workers** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*)：如果是，数据加载器将不会在数据集被使用一次后关闭工作进程。这允许保持工作人员数据集实例处于活动状态
-   **pin_memory_device** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)*,* *optional*)：如果 pin_memory 设置为 true，数据加载器将在返回之前将张量复制到设备固定内存中

**注意：**

`len(dataloader)`启发式基于所用采样器的长度。当 是 [`IterableDataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset) 时，它会返回基于 的估计值，并根据 进行适当的舍入，而不考虑多进程加载配置。这代表了PyTorch可以做出的最佳猜测，因为PyTorch信任用户代码正确处理多进程加载以避免重复数据， 但是，如果分片导致多个工作线程具有不完整的最后批次，则此估计值仍然可能不准确，因为其他完整的批次可以分解为多个批次，并且在设置时可以丢弃多个批次的样本。但PyTorch通常无法检测到此类病例。`drop_last` 

### Dataset

```python
torch.utils.data.Dataset(*args, **kwds)
```

表示[`Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)的抽象类。表示从键到数据样本的映射的所有数据集都应对其进行子类化。所有子类都应该覆盖，支持获取给定键的数据样本。子类也可以选择覆盖，这有望通过许多[`Sampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler) 实现和[`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)的默认选项返回数据集的大小。

`__getitem__()` `__len__()`

**注意：**

默认情况下[`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)构造一个生成整数索引的索引采样器。要使其适用于具有非积分索引/键的地图样式数据集，必须提供自定义采样器.

### IterableDataset

```python
torch.utils.data.IterableDataset(*args, **kwds)
```

可迭代数据集。表示数据样本可迭代的所有数据集都应对其进行子类化。当数据来自流时，这种形式的数据集特别有用。

所有子类都应该覆盖 ，这将返回此数据集中样本的迭代器。`__iter__()`，当子类与[`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)一起使用时，数据集中的每个项都将从[`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)迭代器生成。当 时，每个工作进程将具有不同的数据集对象副本，因此通常需要单独配置每个副本，以避免从工作线程返回重复数据。[`get_worker_info()`](https://pytorch.org/docs/stable/data.html#torch.utils.data.get_worker_info)在工作进程中调用时，返回有关工作线程的信息。它可以在数据集的方法或[`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)的选项中使用，以修改每个副本的行为。`num_workers > 0` `__iter__()` `worker_init_fn`

示例1：在以下位置的所有工作人员之间拆分工作负载：\_\_iter\_\_()

```python
class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # 单进程数据加载，返回完整迭代器
            iter_start = self.start
            iter_end = self.end
        else:  # 在工作进程中
            # 拆分工作负载
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        return iter(range(iter_start, iter_end))
# 应提供 range(3, 7), i.e., [3, 4, 5, 6].
ds = MyIterableDataset(start=3, end=7)

# 单进程加载
print(list(torch.utils.data.DataLoader(ds, num_workers=0)))

# 使用两个工作进程加载多个进程
# 工作线程 0 已获取 [3， 4]。 工作线程 1 已获取 [5， 6]。
print(list(torch.utils.data.DataLoader(ds, num_workers=2)))

# 拥有更多员工
print(list(torch.utils.data.DataLoader(ds, num_workers=20)))
```

示例2：在所有工作线程之间拆分工作负载：worker_init_fn

```python
class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end
    def __iter__(self):
        return iter(range(self.start, self.end))
# 应提供与范围 （3， 7） 相同的数据集，即 [3， 4， 5， 6]。
ds = MyIterableDataset(start=3, end=7)

# 单进程加载
print(list(torch.utils.data.DataLoader(ds, num_workers=0)))
# 直接执行多进程加载会产生重复数据
print(list(torch.utils.data.DataLoader(ds, num_workers=2)))

# 定义一个“worker_init_fn”，以不同的方式配置每个数据集副本
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # 此工作进程中的数据集副本
    overall_start = dataset.start
    overall_end = dataset.end
    # 将数据集配置为仅处理拆分的工作负载
    per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)

# 使用自定义“worker_init_fn”进行多进程加载
# 工作线程 0 已获取 [3， 4]。 工作线程 1 已获取 [5， 6]。
print(list(torch.utils.data.DataLoader(ds, num_workers=2, worker_init_fn=worker_init_fn)))

# 拥有更多员工
print(list(torch.utils.data.DataLoader(ds, num_workers=20, worker_init_fn=worker_init_fn)))
```

### TensorDataset

数据集包装张量，每个样本将通过第一维度索引张量来进行检索

```python
torch.utils.data.TensorDataset(*tensors)
```

参数：

-   tensor(tensor)：与第一维大小形同的张量

source：

```python
class ConcatDataset(Dataset[T_co]):
   r"""
   数据集作为多个数据集的串联。此类可用于组合不同的现有数据集。
	参数：
        数据集（序列）：要串联的数据集列表
    """
    datasets: List[Dataset[T_co]]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super(ConcatDataset, self).__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, # “数据集不应该是一个空的可迭代的” # 类型： 忽略[arg-type]
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes

```

### ConcatDataset

数据作为多个数据集的串联，此类可用于组合不同的现有数据集.

```python
torch.utils.data.ConcatDataset(datasets)
```

参数：

-    **datasets** (*sequence*)： 要连接的数据集列表 

source:

```python
class ConcatDataset(Dataset[T_co]):
   r"""数据集作为多个数据集的串联。此类可用于组合不同的现有数据集。
	参数：
        数据集（序列）：要串联的数据集列表
    """
    datasets: List[Dataset[T_co]]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super(ConcatDataset, self).__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, # “数据集不应该是一个空的可迭代的” # 类型： 忽略[arg-type]
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes
```

### Subset

指定索引处的数据集的子集

```python
torch.utils.data.Subset(dataset, indices)
```

参数：

-   **dataset** ([*Dataset*](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset))：整个数据集
-   **indices** (*sequence*)：为子集选择整个集合中的索引

source：

```python
class Subset(Dataset[T_co]):
   r"""
   指定索引处的数据集的子集。
   参数：
        数据集 （数据集）：整个数据集
        索引（序列）：为子集选择的整套索引
    """
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)
```

### Default_collate

该函数接受一批数据并将批处理中原始放入具有附加外部维度(批处理大小的张量)中，输出类型可用是[`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)张量，或保持不变，具体取决于输入类型，当在[`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)中定义batch_size或batch_sampler时，可被用作排序规则的默认函数

```python
torch.utils.data.default_collate(batch)
```

下列为输入类型(基于批处理和zoo那个原始的类型)到输出类型的映射：


-   [`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) -> [`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) (with an added outer dimension batch size)
-   NumPy Arrays -> [`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)
-   float -> [`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)
-   int -> [`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)
-   str -> str (unchanged)
-   bytes -> bytes (unchanged)
-   Mapping[K, V_i] -> Mapping[K, default_collate([V_1, V_2, …])]
-   NamedTuple[V1_i, V2_i, …] -> NamedTuple[default_collate([V1_1, V1_2, …]), default_collate([V2_1, V2_2, …]), …]
-   Sequence[V1_i, V2_i, …] -> Sequence[default_collate([V1_1, V1_2, …]), default_collate([V2_1, V2_2, …]), …]

参数：

-   batch_size：整理的单个批次

举例：

```python
# 一批“int”的示例：
default_collate([0, 1, 2, 3])
# 一批“str”的示例：
default_collate(['a', 'b', 'c'])
# 批处理中“地图map”的示例：
default_collate([{'A': 0, 'B': 1}, {'A': 100, 'B': 100}])
# 批处理中具有“命名单元”的示例：
Point = namedtuple('Point', ['x', 'y'])
default_collate([Point(0, 0), Point(1, 1)])
# 批处理中“元组”的示例：
default_collate([(0, 1), (2, 3)])
# 批处理中“列表”的示例：
default_collate([[0, 1], [2, 3]])
```

### ChainDataset

用于连接多个 [`IterableDataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset)的数据集，此类可用于组合不同的现有数据集流，连接操作时动态完成的，因此将大型数据集与此类连接起来将是有效的.

```python
torch.utils.data.ChainDataset(datasets)
```

参数：

-    **datasets** (*iterable of IterableDataset*)：要链接在一起的数据集

source：

```python
class ChainDataset(IterableDataset):
    r"""
    用于链接多个的数据集：类：'可迭代数据集's。此类可用于组合不同的现有数据集流。这
    链式操作是动态完成的，因此可以大规模连接，具有此类的数据集将是有效的。
    
	参数：
        数据集（可迭代的可迭代数据集）：要链接在一起的数据集
    """
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super(ChainDataset, self).__init__()
        self.datasets = datasets

    def __iter__(self):
        for d in self.datasets:
            assert isinstance(d, IterableDataset), "ChainDataset only supports IterableDataset"
            for x in d:
                yield x

    def __len__(self):
        total = 0
        for d in self.datasets:
            assert isinstance(d, IterableDataset), "ChainDataset only supports IterableDataset"
            total += len(d)  # type: ignore[arg-type]
        return total
```

### Default_convert

将每个Numpy数组元素转换为[`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)，如果输入是序列，集合或映射，他将尝试将内部的每个元素转换为[`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)。如果输入不是Numpy数组，则保存不变，当[`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)中未定义batch_sampler和batch_size时，者将被用作排序规则的默认函数，

```python
torch.utils.data.default_convert(data)
```

参数：

-    **data**：要转换的单个数据点

举例：

```python
# 带有“整数”的示例
default_convert(0)
# 数字数组的示例
default_convert(np.array([0, 1]))
# 命名示例
Point = namedtuple('Point', ['x', 'y'])
default_convert(Point(0, 0))
default_convert(Point(np.array(0), np.array(0)))
# 列表示例
default_convert([np.array([0, 1]), np.array([2, 3])])
```

### Get_worker_info()

返回有关当前的[`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)迭代器工作进程的信息，在工作线程中调用时，浙江返回一个包装具有以下属性的对象：

-   `id`：当前辅助角色 ID。
-   `num_workers`：工人总数。
-   `seed`：当前工作线程的随机种子集。此值由主进程 RNG 和辅助角色 ID 确定。
-   `dataset`：**此**过程中数据集对象的副本。请注意，这将是不同进程中的不同对象，而不是主进程中的对象。

在主进程中调用时，将返回 `None`

**注意：**

当在传递给 [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) 时使用时，此方法可用于以不同的方式设置每个工作进程，例如，将对象配置为仅读取分片数据集的特定部分，或者用于为数据集代码中使用的其他库设定种子。`worker_init_fn` `worker_id` `dataset` `seed` 

### Random_split

将数据集随机拆分为给定长度非重叠数据集，(可选)固定发生器以获得可重复的结果

```python
torch.utils.data.random_split(dataset, lengths, generator=<torch._C.Generator object>)
```

source:

```python
def random_split(dataset: Dataset[T], lengths: Sequence[int],
                 generator: Optional[Generator] = default_generator) -> List[Subset[T]]:
    r"""
    将数据集随机拆分为给定长度的非重叠新数据集。
    （可选）固定发生器以获得可重复的结果，
    例如：
    random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
	参数：
        数据集 （数据集）：要拆分的数据集
        长度（序列）：要生成的拆分的长度
        生成器（生成器）：用于随机排列的生成器。
    """
    # 无法验证数据集是否为“已调整大小”
    if sum(lengths) != len(dataset):    # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()
    return [Subset(dataset, indices[offset - length : offset]) for offset, 
            length in zip(_accumulate(lengths), lengths)]
```

参数：

-   **dataset** ([*Dataset*](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset))：要拆分的数据集
-   **lengths** (*sequence*)：要拆分的长度
-   **generator** ([*Generator*](https://pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator))：用于随机排列的生成器。

### Sampler

索引采样器的基类，每个Sampler子类都必须提供一个方法，提供一种可迭代数据集元素索引的方法，以及一个返回爹嗲气长度的方法.(\_\_iter\_\_()，\_\_len\_\_())

```python
torch.utils.data.Sampler(data_source)
```

**注意：**

[`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)并不严格要求该方法，但在涉及[`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)长度的任何计算中都应该使用此方法(\_\_len\_\_())

source：

```python
class Sampler(Generic[T_co]):
    r"""所有采样器的基类。

每个采样器子类都必须提供一个 ：meth：'__iter__' 方法，提供
    迭代数据集元素索引的方法，以及：meth：'__len__'方法
    返回返回的迭代器的长度。

    注意：： ：meth：'__len__' 方法不是严格要求的
              ：类：“~火炬.utils.data.DataLoader”，但在任何
              计算涉及一个：类：“~火炬.utils.data.DataLoader”的长度。
    """

    def __init__(self, data_source: Optional[Sized]) -> None:
        pass

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError


    # NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
    #
    # Many times we have an abstract class representing a collection/iterable of
    # data, e.g., `torch.utils.data.Sampler`, with its subclasses optionally
    # implementing a `__len__` method. In such cases, we must make sure to not
    # provide a default implementation, because both straightforward default
    # implementations have their issues:
    #
    # + “返回未实现”：
    # 调用“subclass_instance”引发：
    # 类型错误：“未实现类型”对象不能解释为整数
    #
    # + “引发未实现错误（）”：
    # 这可以防止触发某些回退行为。例如，内置
    # “列表（X）”首先尝试调用“len（X）”，并执行不同的代码
    # 路径，如果未找到该方法或返回“未实现”，则返回
    # 引发“未实现错误”将传播并发出调用
    # 失败，其中可以使用“__iter__”来完成调用。
    #
    # 因此，唯一明智的两件事
    #
    # + **not** 提供默认的“__len__”。
    #
    # + 引发一个“类型错误”，这是 Python 在用户调用时使用的
    # 未在对象上定义的方法。
    # （@ssnl验证这是否至少在 Python 3.7 上有效。
```

### SequentialSampler

按顺序对元素进行采样，始终按相同的顺序进行采样

```python
torch.utils.data.SequentialSampler(data_source)
```

参数：

-   **data_source** ([*Dataset*](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset))：要从中取样的数据集

source：

```python
class SequentialSampler(Sampler[int]):
    r"""
    按顺序对元素进行采样，始终以相同的顺序进行采样。
	参数：
        data_source（数据集）：要从中采样的数据集
    """
    data_source: Sized

    def __init__(self, data_source: Sized) -> None:
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)
```

### RandomSampler

随机采样元素，入宫没有替换，则从随机数据集中采样，如果用替换，那么永华可用指定绘制`num_samples`

```python
torch.utils.data.RandomSampler(data_source, replacement=False, num_samples=None, generator=None)
```

参数：

-   **data_source** ([*Dataset*](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset))：要从中取样的数据集
-   **replacement** ([*bool*](https://docs.python.org/3/library/functions.html#bool))：如果 ，则按需绘制样本并进行替换，默认值为“假”“真”
-   **num_samples** ([*int*](https://docs.python.org/3/library/functions.html#int))：要绘制的样本数，默认为“len（数据集）”。
-   **generator** ([*Generator*](https://pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator))：用于采样的发生器。

source：

```python
class RandomSampler(Sampler[int]):
   r"""
   随机采样元素。如果没有替换，则从随机数据集中取样。
   如果使用替换，则用户可以指定：attr：'num_samples'进行绘制。
	参数：
        data_source（数据集）：要从中采样的数据集
        替换（布尔）：如果“True”，则按需抽取样本并进行替换，默认值为“False”。
        num_samples（int）：要绘制的样本数，默认值为“len（数据集）”。
        发生器（发生器）：用于采样的发生器。
    """
    data_source: Sized
    replacement: bool

    def __init__(self, data_source: Sized, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.replacement, bool):
            raise TypeError("replacement should be a boolean value, but got "
                            "replacement={}".format(self.replacement))

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self) -> int:
        # 数据集大小在运行时可能会更改
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64,
                                         generator=generator).tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), 
                                     dtype=torch.int64, generator=generator).tolist()
        else:
            for _ in range(self.num_samples // n):
                yield from torch.randperm(n, generator=generator).tolist()
            yield from torch.randperm(n, generator=generator).tolist()[:self.num_samples % n]

    def __len__(self) -> int:
        return self.num_samples
```

### SubsetRandomSampler

从给定索引的列表中随机抽取元素取样，无需替换

```python
torch.utils.data.SubsetRandomSampler(indices, generator=None)
```

参数：

-   **indices** (*sequence*)：系列索引
-   **generator** ([*Generator*](https://pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator))：用于采样的发生器。

source：

```python
class SubsetRandomSampler(Sampler[int]):
    r"""
    从给定的索引列表中随机抽取元素，无需替换。
	参数：
        索引（序列）：一系列索引
    """
    indices: Sequence[int]

    def __init__(self, indices: Sequence[int], generator=None) -> None:
        self.indices = indices
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        for i in torch.randperm(len(self.indices), generator=self.generator):
            yield self.indices[i]

    def __len__(self) -> int:
        return len(self.indices)
```

### WeightedRandomSampler

 从给定概率（权重）中采样元素。`[0,..,len(weights)-1]` 

```python
torch.utils.data.WeightedRandomSampler(weights, num_samples, replacement=True, generator=None)
```

参数：

-   **weights** (*sequence*)：系列权重，不必加起来一个
-   **num_samples** ([*int*](https://docs.python.org/3/library/functions.html#int))：要绘制的样本数
-   **replacement** ([*bool*](https://docs.python.org/3/library/functions.html#bool))：如果 ，则抽取替换样品。如果不是，则在没有替换的情况下绘制它们，这意味着当为一行绘制示例索引时，无法为该行再次绘制该索引。
-   **generator** ([*Generator*](https://pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator))：用于采样的发生器。



source：

```python
class WeightedRandomSampler(Sampler[int]):
    r"""用给定的概率（权重）从''[0,..,len（权重）-1]''中采样元素。

	参数：
        权重（序列）：权重序列，不必加起来一个
        num_samples（国际）：要抽取的样本数
        替换（布尔值）：如果为“True”，则抽取替换样品。如果不是，则绘制它们而不进行替换，
        这意味着当为一行绘制示例索引，不能为该行再次绘制。
        发生器（发生器）：用于采样的发生器。

    Example:
        >>> list(WeightedRandomSampler([0.1, 0.9, 0.4, 0.7, 3.0, 0.6], 5, replacement=True))
        [4, 4, 1, 4, 5]
        >>> list(WeightedRandomSampler([0.9, 0.4, 0.05, 0.2, 0.3, 0.1], 5, replacement=False))
        [0, 1, 4, 3, 2]
    """
    weights: Tensor
    num_samples: int
    replacement: bool

    def __init__(self, weights: Sequence[float], num_samples: int,
                 replacement: bool = True, generator=None) -> None:
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement, 
                                        generator=self.generator)
        yield from iter(rand_tensor.tolist())

    def __len__(self) -> int:
        return self.num_samples
```

举例：

```python
>>> list(WeightedRandomSampler([0.1, 0.9, 0.4, 0.7, 3.0, 0.6], 5, replacement=True))
[4, 4, 1, 4, 5]
>>> list(WeightedRandomSampler([0.9, 0.4, 0.05, 0.2, 0.3, 0.1], 5, replacement=False))
[0, 1, 4, 3, 2]
```

### BatchSampler

包装一个采样器以生成一小批索引

```python
torch.utils.data.BatchSampler(sampler, batch_size, drop_last)
```

参数：

-   **sampler** ([*Sampler*](https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler) *or* *Iterable*)：基本采样器。可以是任何可迭代对象
-   **batch_size** ([*int*](https://docs.python.org/3/library/functions.html#int))：批次的大小。
-   **drop_last** ([*bool*](https://docs.python.org/3/library/functions.html#bool))：如果 ，则采样器将丢弃最后一批，如果其大小小于“True”batch_size

source：

```python
class BatchSampler(Sampler[List[int]]):
    r"""包装另一个采样器以生成一小批索引。
	参数：
        采样器（采样器或可迭代）：基本采样器。可以是任何可迭代对象
        batch_size（国际）：迷你批的大小。
        drop_last（布尔）：如果为“True”，则采样器将丢弃最后一批，如果
            它的大小将小于“batch_size”

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, 
                 sampler: Union[Sampler[int], 
                 Iterable[int]], 
                 batch_size: int, 
                 drop_last: bool) -> None:
       # 由于集合.abc.迭代不检查“__getitem__”，因此
        # 是对象成为可迭代对象的一种方式，我们不做“实例”
        # 检查这里。
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
        if self.drop_last:
            sampler_iter = iter(self.sampler)
            while True:
                try:
                    batch = [next(sampler_iter) for _ in range(self.batch_size)]
                    yield batch
                except StopIteration:
                    break
        else:
            batch = [0] * self.batch_size
            idx_in_batch = 0
            for idx in self.sampler:
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    yield batch
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]

    def __len__(self) -> int:
      # 只有在自我采样器已实现__len__时才能调用
        # 我们无法强制执行此条件，因此我们关闭了
        # 实现如下。
        # 有点相关：请参阅注意 [ Python 抽象基类中缺少默认的“__len__”]
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]

```

举例：

```python
>>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
>>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
[[0, 1, 2], [3, 4, 5], [6, 7, 8]]
```

### DistributedSampler

将数据加载限制为数据集子集的采样器，它与[`torch.nn.parallel.DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel)特别有用，在这种情况下，每个进程都可以将一个实例作为 [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) 采样器传递，并加载原始数据集的一个子集，该子集是该实例独有的。`DistributedSampler` 

```python
torch.utils.data.distributed.DistributedSampler(dataset, 
                                                num_replicas=None, 
                                                rank=None, 
                                                shuffle=True, 
                                                seed=0, 
                                                drop_last=False)
```

**注意：**

假定数据集的大小时恒定的，并且它的任何实例始终以相同的顺序返回相同的元素

参数：

-   **dataset**：用于采样的数据集。
-   **num_replicas** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* *optional*)：参与分布式训练的进程数。默认情况下， 从当前分布式组中检索。'world_size'
-   **rank** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* *optional*)：当前进程的排名。默认情况下， 从当前分布式组中检索num_replicas“排名”
-   **shuffle** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*)：如果（默认），采样器将随机排列索引。
-   **seed** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* *optional*)：随机种子用于随机排列采样器，如果 .此数字在分布式组中的所有进程中应相同。默认值：.'随机播放 = 真''0'
-   **drop_last** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*)：如果 ，则采样器将删除数据的尾部，使其在副本数量之间均匀可整除。如果为 ，则采样器将添加额外的索引，以使数据在副本之间均匀可整除。默认值：“真”假“

在分布式模式下，在创建迭代器**之前**在每个 epoch 的开头调用该方法对于使随机播放在多个 epoch 中正常工作是必要的。否则，将始终使用相同的排序。`set_epoch()``DataLoader` 

source：

```python
class DistributedSampler(Sampler[T_co]):
    r"""采样器，将数据加载限制为数据集的子集。

	它与
    ：类：“火炬”.nn.并行.分布式数据并行“。在这种情况下，每个
    进程可以将一个 ：class：'~炬.utils.data.分布式采样器' 实例作为
    ：类：“~火炬.utils.data.DataLoader”采样器，并加载
    它独有的原始数据集。

    .. 注意：：
        假设数据集的大小是恒定的，并且它的任何实例始终
        以相同的顺序返回相同的元素。

    参数：
        数据集：用于采样的数据集。
        num_replicas（国际，可选）：参与的进程数
            分布式训练。默认情况下，world_size 将从
            当前分布式组。
        排名（整数，可选）：当前进程在 ：attr：'num_replicas' 中的排名。
            默认情况下，：attr：'排名'是从当前分布中检索的群。
        随机播放（布尔，可选）：如果为“True”（默认），采样器将随机播放指标。
        种子（int，可选）：随机种子用于随机排列采样器，如果
            ：阿特尔：'洗牌=真'。此数字在所有范围内应相同
            分布式组中的进程。默认值：“0”。
       drop_last（布尔，可选）：如果为“True”，则采样器将删除
            数据的尾部，使其在数量上均匀可整除
            副本。如果为“False”，则采样器将添加额外的索引以进行
            数据在副本之间均匀可整除。默认值：“假”。

   ..警告：：
        在分布式模式下，在 以下位置调用 ：meth：'set_epoch' 方法
        每个纪元的开始 **之前** 创建 ：类：'数据加载器' 迭代器
        对于在多个时代中正确进行洗牌是必要的。否则
        将始终使用相同的排序。

    Example::

        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # 如果数据集长度可被副本数整除，则存在
        # 不需要删除任何数据，因为数据集将被平均拆分。
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # 拆分为可均匀整除的最近可用长度。
            # 这是为了确保每个排名在以下情况下接收相同数量的数据
            # 使用此采样器。
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
           # 基于时代和种子的确定性洗牌
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
           #添加额外的样本以使其均匀可分割
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # 删除数据尾部，使其均匀可整除。
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # 子样本
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
       r"""
        设置此采样器的纪元。当 ：attr：'随机播放 = True'时，这可确保所有副本
        对每个纪元使用不同的随机排序。否则，此的下一个迭代
        采样器将产生相同的顺序。

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
```

举例：

```python
sampler = DistributedSampler(dataset) if is_distributed else None
loader = DataLoader(dataset, shuffle=(sampler is None),
                    sampler=sampler)
for epoch in range(start_epoch, n_epochs):
    if is_distributed:
        sampler.set_epoch(epoch)
    train(loader)
```

