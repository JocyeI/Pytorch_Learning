# 第九章 CUDA设备

## CUDA简介

[`torch.cuda`](https://pytorch.org/docs/stable/cuda.html#module-torch.cuda)是用于设置和运行库达的操作，它跟着当前所选的GPU，默认情况下，分配的所有CUDA张量都将在该设备上创建，可用使用[`torch.cuda.device`](https://pytorch.org/docs/stable/generated/torch.cuda.device.html#torch.cuda.device)上下文管理管理器更改所选设备。一旦分配了张量，无论所选设备如何，都可用对其执行操作，并且结果将始终放置在与张量相同的设备上。

默认情况下不允许跨GPU操作，但``copy_()``和其他类似复制功能的方法有`cuda()`和`to()`，如果不启用对等内存访问，那么任何尝试在分布不同设备上的张量进行操作都会导致错误。

举例：通过小例子进行展示这类问题

```python
cuda = torch.device('cuda')     # 默认CUDA设备
cuda0 = torch.device('cuda:0')
cuda2 = torch.device('cuda:2')  # GPU 2（这些是 0 索引）

x = torch.tensor([1., 2.], device=cuda0)
# x.device is device(type='cuda', index=0)
y = torch.tensor([1., 2.]).cuda()
# y.device is device(type='cuda', index=0)

with torch.cuda.device(1):
    # 在 GPU 1 上分配张量
    a = torch.tensor([1., 2.], device=cuda)

    # 将张量从中央处理器传输到 GPU 1
    b = torch.tensor([1., 2.]).cuda()
    # a.device and b.device are device(type='cuda', index=1)

    # 也可以使用“Tensor.to”来传递张量
    b2 = torch.tensor([1., 2.]).to(device=cuda)
    # b.device and b2.device are device(type='cuda', index=1)

    c = a + b
    # c.device is device(type='cuda', index=1)

    z = x + y
    # z.device is device(type='cuda', index=0)

    # 即使在上下文中，您也可以指定设备
    # （或为 .cuda 调用提供 GPU 索引）
    d = torch.randn(2, device=cuda2)
    e = torch.randn(2).to(cuda2)
    f = torch.randn(2).cuda(cuda2)
    # d.device, e.device, and f.device are all device(type='cuda', index=2)
```

## 张量-32

TensorFloat-32(TF32) on Ampere devices，数据类型为float32的张量。从PyTorch 1.7开始，有一个名为allow_tf32的新标志。此标志在 PyTorch 1.7 到 PyTorch 1.11 中默认为 True，在 PyTorch 1.12 及更高版本中默认为 False。此标志控制是否允许 PyTorch 在内部使用自安培以来在新的 NVIDIA GPU 上可用的张量内核来计算矩阵(矩阵乘法和批处理矩阵乘法)和卷积。

TF32张量磁芯旨在通过将输入数据舍入为具有10位尾数，并以FP32精度累积结果，保持FP32动态范围，从而在matmul上实现更好的性能和在手电筒上的卷积.float32张量。

矩阵和卷积是分开控制的，它们的相应标志可以在以下位置访问：

```python
# 下面的标志控制是否允许在垫上使用 TF32。此标志默认为“假”
# 在 PyTorch 1.12 及更高版本中。
torch.backends.cuda.matmul.allow_tf32 = True

# 下面的标志控制是否允许在 cuDNN 上使用 TF32。此标志默认为 True。
torch.backends.cudnn.allow_tf32 = True
```

**注意：**

处理矩阵和卷积本身之外，内部使用矩阵或卷积的函数和nn模块都会受到影响，其中包括 nn.Linear, nn.Conv*, cdist, tensordot, affine grid and grid sample, adaptive log softmax, GRU and LSTM. 

通过下面例子，了解精度和速度：

```python
a_full = torch.randn(10240, 10240, dtype=torch.double, device='cuda')
b_full = torch.randn(10240, 10240, dtype=torch.double, device='cuda')
ab_full = a_full @ b_full
mean = ab_full.abs().mean()  # 80.7277

a = a_full.float()
b = b_full.float()

# 在 TF32 模式下进行点积操作
torch.backends.cuda.matmul.allow_tf32 = True
ab_tf32 = a @ b  # takes 0.016s on GA100
error = (ab_tf32 - ab_full).abs().max()  # 0.1747
relative_error = error / mean  # 0.0022

# 禁用 TF32
torch.backends.cuda.matmul.allow_tf32 = False
ab_fp32 = a @ b  # takes 0.11s on GA100
error = (ab_fp32 - ab_full).abs().max()  # 0.0031
relative_error = error / mean  # 0.000039
```

从上面的例子中，可以看到，在启用TF32的情况下，速度快了大约7倍，相对误差与双精度相比大约大了2个数量级。如果需要完整的 FP32 精度，用户可以通过以下方式禁用 TF32： 

```python
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

For more information about TF32, see:

-   [TensorFloat-32](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/)
-   [CUDA 11](https://devblogs.nvidia.com/cuda-11-features-revealed/)
-   [Ampere architecture](https://devblogs.nvidia.com/nvidia-ampere-architecture-in-depth/)

## 降低精度

fp16 GEMM可能通过一些中间的精度降低来完成(例如：在fp16而不是fp32中)。这些选择性的精度降低可以在某些工作负载(特别是那些具有大 k 维的工作负载)和 GPU 架构上实现更高的性能，但代价是数值精度和溢出的可能性。

V100 上的一些基准测试数据示例：

```python
[--------------------------- bench_gemm_transformer --------------------------]
      [  m ,  k  ,  n  ]    |  allow_fp16_reduc=True  |  allow_fp16_reduc=False
1 threads: --------------------------------------------------------------------
      [4096, 4048, 4096]    |           1634.6        |           1639.8
      [4096, 4056, 4096]    |           1670.8        |           1661.9
      [4096, 4080, 4096]    |           1664.2        |           1658.3
      [4096, 4096, 4096]    |           1639.4        |           1651.0
      [4096, 4104, 4096]    |           1677.4        |           1674.9
      [4096, 4128, 4096]    |           1655.7        |           1646.0
      [4096, 4144, 4096]    |           1796.8        |           2519.6
      [4096, 5096, 4096]    |           2094.6        |           3190.0
      [4096, 5104, 4096]    |           2144.0        |           2663.5
      [4096, 5112, 4096]    |           2149.1        |           2766.9
      [4096, 5120, 4096]    |           2142.8        |           2631.0
      [4096, 9728, 4096]    |           3875.1        |           5779.8
      [4096, 16384, 4096]   |           6182.9        |           9656.5
(times in microseconds).
```

如果全精度降低，用户可以通过以下方式禁用 fp16 GEMM 中的精度降低： 

```python

```python
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
```

## 异步执行

默认情况下，GPU 操作是异步的。调用使用 GPU 的函数时，操作将排队到特定设备，但不一定要等到以后才执行。这能够并行执行更多计算，包括在CPU或其他GPU上的操作。

通常，异步计算的效果对调用方来说是不可见的，因为每个设备都按照排队的顺序执行操作，并且PyTorch在CPU和GPU之间或两个GPU之间复制数据时自动执行必要的同步。因此，计算将像每个操作都是同步执行一样进行。

但可以通过设置环境变量来强制同步计算。当GPU上发生错误时，这可能很方便(对于异步执行，在实际执行操作之前不会报告此类错误，因此堆栈跟踪不会显示请求的位置) `CUDA_LAUNCH_BLOCKING=1`

异步计算的结果是，没有同步的时间测量不准确。为了获得精确的测量结果，应该调用 [`torch.cuda.synchronize()`](https://pytorch.org/docs/stable/generated/torch.cuda.synchronize.html#torch.cuda.synchronize)或使用 [`torch.cuda.Event`](https://pytorch.org/docs/stable/generated/torch.cuda.Event.html#torch.cuda.Event)，记录时间的事件如下：

```python
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()

# 在此处运行一些内容
end_event.record()

# 等待事件被记录下来！
torch.cuda.synchronize()  
elapsed_time_ms = start_event.elapsed_time(end_event)
```

存在一些列外，一些函数(如：to()和copy_())允许显示参数，这允许调用方在不必要的情况下绕过同步.

## 库达流

 [CUDA stream](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams) 属于特定设备的线性执行序列。通常不需要显式创建一个：默认情况下，每个设备都使用自己的“默认”流。每个流中的操作按其创建顺序序列化，但来自不同流的操作可以按任何相对顺序并发执行，除非使用显式同步函数 [`synchronize()`](https://pytorch.org/docs/stable/generated/torch.cuda.synchronize.html#torch.cuda.synchronize) or [`wait_stream()`](https://pytorch.org/docs/stable/generated/torch.cuda.Stream.html#torch.cuda.Stream.wait_stream)例如，下面的代码不正确：

```python
cuda = torch.device('cuda')
s = torch.cuda.Stream()  # 创建一个新的cuda stream
A = torch.empty((100, 100), device=cuda).normal_(0.0, 1.0)
with torch.cuda.stream(s):
    # sum（） 可以在 normal_（） 完成之前开始执行！
    B = torch.sum(A)
```

当“当前流”是默认流时，PyTorch 会在数据移动时自动执行必要的同步，如上所述。但是，在使用非默认流时，用户有责任确保正确的同步。 

## 向后传递

每个向后 CUDA 操作都运行在用于其相应正向操作的同一流上。如果正向传递在不同流上并行运行独立的操作，这有助于反向传递利用相同的并行性。相对于周围操作的向后调用的流语义与任何其他调用相同。向后传递插入内部同步，以确保即使在上一段所述的反向操作在多个流上运行时也是如此。更具体地说，当调用  [`autograd.backward`](https://pytorch.org/docs/stable/generated/torch.autograd.backward.html#torch.autograd.backward), [`autograd.grad`](https://pytorch.org/docs/stable/generated/torch.autograd.grad.html#torch.autograd.grad), or [`tensor.backward`](https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html#torch.Tensor.backward),  并选择性地提供CUDA张量作为初始梯度(e.g., [`autograd.backward(..., grad_tensors=initial_grads)`](https://pytorch.org/docs/stable/generated/torch.autograd.backward.html#torch.autograd.backward), [`autograd.grad(..., grad_outputs=initial_grads)`](https://pytorch.org/docs/stable/generated/torch.autograd.grad.html#torch.autograd.grad), or [`tensor.backward(..., gradient=initial_grad)`](https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html#torch.Tensor.backward)

行为：

1.  (可选)填充初始渐变，
2.  调用反向传递
3.  使用渐变

与任何一组操作具有相同的流，语义关系：

```python
s = torch.cuda.Stream()

# 安全，渐变在与backed（）相同的流上下文中使用
with torch.cuda.stream(s):
    loss.backward()
    use grads

# 不安全的
with torch.cuda.stream(s):
    loss.backward()
use grads

# 安全，带同步
with torch.cuda.stream(s):
    loss.backward()
torch.cuda.current_stream().wait_stream(s)
use grads

# 安全、填充初始分级和向后调用位于同一流上下文中
with torch.cuda.stream(s):
    loss.backward(gradient=torch.ones_like(loss))

# 不安全、填充initial_grad和向后调用位于不同的流上下文中，
# 无同步
initial_grad = torch.ones_like(loss)
with torch.cuda.stream(s):
    loss.backward(gradient=initial_grad)

# 安全，同步
initial_grad = torch.ones_like(loss)
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    initial_grad.record_stream(s)
    loss.backward(gradient=initial_grad)
```

**注意：在默认流上使用渐变**

在以前版本的PyTorch(1.9及更早版本)中，autograd引擎始终将默认流与所有向后操作同步，因此以下模式： 

```python
with torch.cuda.stream(s):
    loss.backward()
use grads
```

只要在默认流上发生，就是安全的。在目前的PyTorch中，这种模式不再安全。如果和位于不同的流上下文中，则必须同步流：`use grads` `backward()` `use grads` 

```python
with torch.cuda.stream(s):
    loss.backward()
torch.cuda.current_stream().wait_stream(s)
use grads
```

即使在默认流中：use grads

## 内存管理

PyTorch 使用缓存内存分配器来加速内存分配。这允许在没有设备同步的情况下快速释放内存。但是，由分配器管理的未使用内存仍将显示为在中使用.

1、使用[`memory_allocated()`](https://pytorch.org/docs/stable/generated/torch.cuda.memory_allocated.html#torch.cuda.memory_allocated) 和 [`max_memory_allocated()`](https://pytorch.org/docs/stable/generated/torch.cuda.max_memory_allocated.html#torch.cuda.max_memory_allocated)来监视张量占用的内存，

2、使用 [`memory_reserved()`](https://pytorch.org/docs/stable/generated/torch.cuda.memory_reserved.html#torch.cuda.memory_reserved) 和 [`max_memory_reserved()`](https://pytorch.org/docs/stable/generated/torch.cuda.max_memory_reserved.html#torch.cuda.max_memory_reserved) 来监视缓存分配器管理的内存总量

3、调用 [`empty_cache()`](https://pytorch.org/docs/stable/generated/torch.cuda.empty_cache.html#torch.cuda.empty_cache) 会从 PyTorch 释放所有**未使用的**缓存内存，以便其他 GPU 应用程序可以使用这些内存。

但是，张量占用的 GPU 内存不会被释放，因此它不能增加可用于 PyTorch 的 GPU 内存量。

对于要求更改的用户，可通过memory_stats()提供了更全面的基准测试，还提供了memory_shaphot()捕获内存分布的完整快照的功能：

1、使用缓存分配器可能会干扰内存检查工具，如要使用调试内存错误，请在环境中设置为禁用缓存。`cuda-memcheck` `cuda-memcheck` `PYTORCH_NO_CUDA_MEMORY_CACHING=1` 

2、缓存分配器的行为可以通过环境变量 来控制。格式为“可用选项”：

`PYTORCH_CUDA_ALLOC_CONF` `PYTORCH_CUDA_ALLOC_CONF=:,...` 

-   `max_split_size_mb`防止分配器拆分大于此大小(以 MB 为单位)的块。这有助于防止碎片，并可能允许某些边缘工作负载在不耗尽内存的情况下完成。绩效成本的范围可以从“零”到“次国家”，具体取决于分配模式。默认值是无限的，即所有块都可以拆分。[`memory_stats()`](https://pytorch.org/docs/stable/generated/torch.cuda.memory_stats.html#torch.cuda.memory_stats) 和 [`memory_summary()`](https://pytorch.org/docs/stable/generated/torch.cuda.memory_summary.html#torch.cuda.memory_summary) 方法对于优化非常有用。对于由于“内存不足”而中止并显示大量非活动拆分块的工作负载，应将此选项用作最后的手段。

-   `roundup_power2_divisions`有助于将请求的分配大小舍入到最接近的幂 2 除法，并更好地利用块。在当前的 CUDA缓存分配器中，大小以块大小 512 的倍数向上舍入，因此这适用于较小的大小。但是，对于大型的近邻分配，这可能是低效的，因为每个分配都将转到不同大小的块，并且这些块的重用被最小化。这可能会创建大量未使用的块，并会浪费 GPU 内存容量。此选项允许将分配大小舍入为最接近的 2 次方除法。例如，如果我们需要将大小四舍五入为 1200，并且如果除法数为 4，则大小 1200 位于 1024 和 2048 之间，如果我们在它们之间进行 4 次划分，则值为 1024、1280、1536 和 1792。因此，1200的分配大小将四舍五入为1280，作为最接近的功率-2除法的上限

-   `garbage_collection_threshold`有助于主动回收未使用的 GPU 内存，以避免触发代价高昂的同步和全部回收操作(release_cached_blocks)，这可能不利于延迟关键型 GPU 应用程序(例如服务器)。设置此阈值(例如，0.8)后，如果 GPU 内存容量使用率超过阈值(即分配给 GPU 应用程序的总内存的 80%)，分配器将开始回收 GPU 内存块。该算法更喜欢先释放旧的和未使用的块，以避免释放正在被重用的块。阈值应介于大于 0.0 和小于 1.0 之间。

## 计划缓存

对于每个CUDA设备，cuFF计划的LRU缓存用于具有相同集合形状和相同配置的库达张量上加速重复允许FFT方法 

例如：[`torch.fft.fft()`](https://pytorch.org/docs/stable/generated/torch.fft.fft.html#torch.fft.fft) 

使用以下 API 控制和查询当前设备缓存的属性：

-   `torch.backends.cuda.cufft_plan_cache.max_size`给出了缓存的容量(在 CUDA 10 及更高版本上默认为 4096，在较旧的 CUDA 版本上默认为 1023)。设置此值会直接修改容量。
-   `torch.backends.cuda.cufft_plan_cache.size`给出当前驻留在缓存中的计划数。
-   `torch.backends.cuda.cufft_plan_cache.clear()`清除缓存。
-   若要控制和查询非默认设备的计划缓存，可以使用 [`torch.device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device) 对象或设备索引为对象编制索引，并访问上述属性之一。例如，要为设备设置缓存的容量，可以写入。
-   `torch.backends.cuda.cufft_plan_cache` `1` `torch.backends.cuda.cufft_plan_cache[1].max_size = 10`

## 及时编译

在CUDA张量上执行时，PyTorch实时编译了一些操作，例如火炬.special.zeta。这种编译可能非常耗时(最多几秒钟，具体取决于您的硬件和软件)，并且对于单个操作员来说可能会发生多次，因为许多PyTorch操作员实际上从各种内核中进行选择，每个内核必须编译一次，具体取决于他们的输入。此编译每个进程发生一次，如果使用内核缓存，则只发生一次。

默认情况下，如果定义了XDG_CACHE_HOME：

PyTorch 会在 $XDG_CACHE_HOME/torch/内核中创建内核缓存，如果没有定义，则$HOME/.cache/torch/内核（在 Windows 上，内核缓存尚不受支持）。可以使用两个环境变量直接控制缓存行为。如果USE_PYTORCH_KERNEL_CACHE设置为 0，则不会使用任何缓存，如果设置PYTORCH_KERNEL_CACHE_PATH，则该路径将用作内核缓存而不是默认位置

## 实践

由于PyTorch的结构，可能需要显式编写与设备无关的(CPU或GPU)代码;一个例子可能是创建一个新的张量作为递归神经网络的初始隐藏状态。 

1、第一步是确定是否应使用 GPU。一种常见的模式是使用Python的模块来读取用户参数，并具有一个可用于禁用CUDA的标志，  [`is_available()`](https://pytorch.org/docs/stable/generated/torch.cuda.is_available.html#torch.cuda.is_available)在下文中，将生成一个可用于将张量移动到 CPU 或 CUDA 的 [`torch.device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device) 对象。

`argparse` 

`args.device` 

```python
import argparse
import torch

parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
args = parser.parse_args()
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')
```

可以用它来在所需的设备上创建张量：`args.device` 

```python
x = torch.empty((8, 42), device=args.device)
net = Network().to(device=args.device)
```

在许多情况下可用于生成与设备无关的代码，下面是使用数据加载器时的示例： 

```python
cuda0 = torch.device('cuda:0')  # CUDA GPU 0
for i, x in enumerate(train_loader):
    x = x.to(cuda0)
```

在系统上使用多个 GPU 时，可以使用环境标志来管理哪些 GPU 可供 PyTorch 使用。如上所述，要手动控制在哪个 GPU 上创建张量，最佳做法是使用 [`torch.cuda.device`](https://pytorch.org/docs/stable/generated/torch.cuda.device.html#torch.cuda.device) 上下文管理器.`CUDA_VISIBLE_DEVICES` 

```python
print("Outside device is 0")  # 在设备 0 上（大多数情况下为默认值）
with torch.cuda.device(1):
    print("Inside device is 1")  # On device 1
print("Outside device is still 0")  # On device 0
```



如果有一个张量，并且想在同一设备上创建一个相同类型的新张量，那么可以使用一种方法[`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) 。虽然前面提到的工厂函数 [创建操作](https://pytorch.org/docs/stable/torch.html#tensor-creation-ops) 取决于当前的GPU上下文和您传入的属性参数，但方法保留了张量的设备和其他属性。

`torch.Tensor.new_*` 

`torch.*` 

`torch.Tensor.new_*`

这是创建模块时的建议做法，在这些模块中，需要在正向传递期间在内部创建新张量。

```python
cuda = torch.device('cuda')
x_cpu = torch.empty(2)
x_gpu = torch.empty(2, device=cuda)
x_cpu_long = torch.empty(2, dtype=torch.int64)

y_cpu = x_cpu.new_full([3, 2], fill_value=0.3)
print(y_cpu)

    tensor([[ 0.3000,  0.3000],
            [ 0.3000,  0.3000],
            [ 0.3000,  0.3000]])

y_gpu = x_gpu.new_full([3, 2], fill_value=-5)
print(y_gpu)

    tensor([[-5.0000, -5.0000],
            [-5.0000, -5.0000],
            [-5.0000, -5.0000]], device='cuda:0')

y_cpu_long = x_cpu_long.new_tensor([[1, 2, 3]])
print(y_cpu_long)

    tensor([[ 1,  2,  3]])
```

如果要创建与另一个张量具有相同类型和大小的张量，并用 1 或 0 填充它，[`ones_like)()`](https://pytorch.org/docs/stable/generated/torch.ones_like.html#torch.ones_like) 或 [`zeros_like()`](https://pytorch.org/docs/stable/generated/torch.zeros_like.html#torch.zeros_like) 作为方便的帮助器函数提供（它们还保留张量的 [`torch.device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device) 和 [`torch.dtype`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)）。 

```python
x_cpu = torch.empty(2, 3)
x_gpu = torch.empty(2, 3)

y_cpu = torch.ones_like(x_cpu)
y_gpu = torch.zeros_like(x_gpu)
```

## 内存缓冲

**使用固定内存缓冲区.**

注意： 如果过度使用固定内存，则在 RAM 不足时可能会导致严重问题，通常是一项代价高昂的操作。 

1、当主机到 GPU 副本源自固定(页面锁定)内存时，它们的速度要快得多。CPU 张量和存储公开一个 [`pin_memory()`](https://pytorch.org/docs/stable/generated/torch.Tensor.pin_memory.html#torch.Tensor.pin_memory) 方法，该方法返回对象的副本，并将数据放在固定区域中。此外，固定张量或存储后，可以使用异步 GPU 副本。只需将附加参数传递给 [`to()`](https://pytorch.org/docs/stable/generated/torch.Tensor.to.html#torch.Tensor.to) 或 [`cuda()`](https://pytorch.org/docs/stable/generated/torch.Tensor.cuda.html#torch.Tensor.cuda) 调用即可。这可用于将数据传输与计算重叠。

`non_blocking=True`

2、可以通过传递给其构造函数来使[`数据加载器`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)返回放置在固定内存中的批处理。 

pin_memory=True 

## 库达图

CUDA 图是 CUDA 流及其依赖流执行的工作(主要是内核及其参数)的记录。

关于基础 CUDA API 的一般原则和详细信息：

1、 [CUDA 图形入门](https://developer.nvidia.com/blog/cuda-graphs/)

2、CUDA C 编程指南的[图形部分](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)。 

PyTorch 支持使用[流捕获](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#creating-a-graph-using-stream-capture)构建 CUDA 图，这会将库达流置于捕获模式。颁发给捕获流的 CUDA 工作实际上不会在 GPU 上运行。相反，工作记录在图表中。

捕获后，可以启动图形以根据需要多次运行GPU工作。每次重播都使用相同的参数运行相同的内核。对于指针参数，这意味着使用相同的内存地址。通过在每次重播之前用新数据(例如，来自新批次的数据)填充输入内存，可以对新数据重新运行相同的工作。 

3、为什么选择库达图

重放图形会牺牲典型急切执行的动态灵活性，以换取 **大大降低的 CPU 开销**。图形的参数和内核是固定的，因此图形重播会跳过参数设置和内核调度的所有层，包括 Python、C++和 CUDA 驱动程序开销。在引擎盖下，重播将整个图形的工作提交到GPU，只需调用一次[即可调用cudaGraphLaunch](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1g1accfe1da0c605a577c22d9751a09597)。重播中的内核在 GPU 上的执行速度也略快，但消除 CPU 开销是主要优势。

如果全部或部分网络是图形安全的(通常这意味着静态形状和静态控制流，但请参阅其他[约束](https://pytorch.org/docs/stable/notes/cuda.html?highlight=cuda#capture-constraints))，并且怀疑其运行时至少在某种程度上受到CPU限制，则应尝试CUDA图。



PyTorch通过原始 [`torch.cuda.CUDAGraph`](https://pytorch.org/docs/stable/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph) 类和两个方便包装， [`torch.cuda.graph`](https://pytorch.org/docs/stable/generated/torch.cuda.graph.html#torch.cuda.graph) 和 [`torch.cuda.make_graphed_callables`](https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html#torch.cuda.make_graphed_callables). 

[`torch.cuda.graph`](https://pytorch.org/docs/stable/generated/torch.cuda.graph.html#torch.cuda.graph) 是一个简单、通用的上下文管理器，可在其上下文中捕获 CUDA 工作。在捕获之前，通过运行一些预先迭代来预热要捕获的工作负荷。预热必须在侧流上进行。由于图形在每次重播中读取和写入相同的内存地址，因此必须维护对张量的长期引用，这些张量在捕获期间保存输入和输出数据。要对新的输入数据运行图形，请将新数据复制到捕获的输入张量，重放图形，然后从捕获的输出张量中读取新输出。例：

```python
g = torch.cuda.CUDAGraph()

# 用于捕获的占位符输入
static_input = torch.empty((5,), device="cuda")

# 捕获前准备
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        static_output = static_input * 2
torch.cuda.current_stream().wait_stream(s)

# 捕获图形
# 要允许捕获，请在上下文中自动将侧流设置为当前流
with torch.cuda.graph(g):
    static_output = static_input * 2

# 用要计算的新数据填充图形的输入内存
static_input.copy_(torch.full((5,), 3, device="cuda"))
g.replay()
# static_output保存结果
print(static_output)  # full of 3 * 2 = 6

# 用更多要计算的数据填充图形的输入内存
static_input.copy_(torch.full((5,), 4, device="cuda"))
g.replay()
print(static_output)  # full of 4 * 2 = 8
```

详情见：  [Whole-network capture](https://pytorch.org/docs/stable/notes/cuda.html?highlight=cuda#whole-network-capture), [Usage with torch.cuda.amp](https://pytorch.org/docs/stable/notes/cuda.html?highlight=cuda#graphs-with-amp), and [Usage with multiple streams](https://pytorch.org/docs/stable/notes/cuda.html?highlight=cuda#multistream-capture) 

[`make_graphed_callables`](https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html#torch.cuda.make_graphed_callables)更复杂。[`make_graphed_callables`](https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html#torch.cuda.make_graphed_callables)接受蟒蛇函数和[`torch`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module).对于每个传递的函数或模块，它创建向前传递和向后传递工作的单独图形。 

## 约束

如果一组操作不违反以下任何约束，则*可捕获*该操作。

约束适用于[`torch.cuda.graph`](https://pytorch.org/docs/stable/generated/torch.cuda.graph.html#torch.cuda.graph)上下文和所有工作在任何可调用传递给[`torch.cuda.make_graphed_callables()`](https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html#torch.cuda.make_graphed_callables). 

违反以下任何一项都可能导致运行时错误：

-   必须在非默认流上进行捕获(仅当使用原始 [`CUDAGraph.capture_begin`](https://pytorch.org/docs/stable/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.capture_begin)和 [`CUDAGraph.capture_end`](https://pytorch.org/docs/stable/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.capture_end)调用时，才需要考虑此问题,[`graph`](https://pytorch.org/docs/stable/generated/torch.cuda.graph.html#torch.cuda.graph) 和[`make_graphed_callables)`](https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html#torch.cuda.make_graphed_callables)设置一个侧流。
-   禁止将 CPU 与 GPU 同步的操作(例如，调用)
-   库达允许 RNG 操作，但必须使用默认生成器,例如显式构造新的 [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator) 实例并将其作为参数传递给 RNG 函数

违反以下任何一项都可能导致静默数字错误或未定义的行为：

-   在一个进程中，一次只能进行一次捕获
-   在捕获过程中，任何未捕获的 CUDA 工作都不得在此过程(在任何线程上)中运行
-   未捕获 CPU 工作。如果捕获的操作包括 CPU 工作，则在重播期间将省略该工作
-   每次重播读取和写入相同的(虚拟)内存地址
-   禁止动态控制流(基于 CPU 或 GPU 数据）
-   禁止使用动态形状。该图假设捕获的 op 序列中的每个张量在每次重放中都具有相同的大小和布局。
-   允许在捕获中使用多个流，但有[一些限制](https://pytorch.org/docs/stable/notes/cuda.html?highlight=cuda#multistream-capture)。

### 非约束

-   捕获后，可以在任何流上重播图形

## 全网捕获

如果整个网络是可捕获的，则可用捕获并重播整个迭代：

```python
N, D_in, H, D_out = 640, 4096, 2048, 1024
model = torch.nn.Sequential(torch.nn.Linear(D_in, H),
                            torch.nn.Dropout(p=0.2),
                            torch.nn.Linear(H, D_out),
                            torch.nn.Dropout(p=0.1)).cuda()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 用于捕获的占位符
static_input = torch.randn(N, D_in, device='cuda')
static_target = torch.randn(N, D_out, device='cuda')

# 预热
# 为方便起见，此处使用static_input和static_target，
# 但在实际设置中，因为预热包括优化器.step（）
# 必须使用几批真实数据。
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for i in range(3):
        optimizer.zero_grad(set_to_none=True)
        y_pred = model(static_input)
        loss = loss_fn(y_pred, static_target)
        loss.backward()
        optimizer.step()
torch.cuda.current_stream().wait_stream(s)

# 捕获
g = torch.cuda.CUDAGraph()
# 在捕获前将灰度设置为无，因此向后（）将创建
# .grad 属性，其中包含来自图形的专用池的分配
optimizer.zero_grad(set_to_none=True)
with torch.cuda.graph(g):
    static_y_pred = model(static_input)
    static_loss = loss_fn(static_y_pred, static_target)
    static_loss.backward()
    optimizer.step()

real_inputs = [torch.rand_like(static_input) for _ in range(10)]
real_targets = [torch.rand_like(static_target) for _ in range(10)]

for data, target in zip(real_inputs, real_targets):
   # 用要计算的新数据填充图形的输入内存
    static_input.copy_(data)
    static_target.copy_(target)
    # 重放（） 包括前进、后退和单步。
    # 您甚至不需要在迭代之间调用 optimizer.zero_grad（）
    # 因为捕获的向后重填静态 .grad 张量到位。
    g.replay()
    # 参数已更新。static_y_pred、static_loss 和 .grad
    # 属性保存此迭代数据计算的值。
```

## 部分捕获

如果某些网络不安全而无法捕获(例如，由于动态控制流、动态形状、CPU 同步或基本的 CPU 端逻辑)，可以急切地运行不安全的部分并使用[`torch.cuda.make_graphed_callables()`](https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html#torch.cuda.make_graphed_callables) 仅绘制捕获安全部件的图形 

默认情况下，[`make_graphed_callables()`](https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html#torch.cuda.make_graphed_callables) 返回的可调用对象是自动降级感知的，并且可以在训练循环中用作函数或 nn 的直接替换传递的模块。 

在以下示例中，依赖于数据的动态控制流意味着网络无法端到端捕获，但[`make_graphed_callables()`](https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html#torch.cuda.make_graphed_callables)允许捕获图形并运行图形安全部分，而无需考虑： 

```python
N, D_in, H, D_out = 640, 4096, 2048, 1024

module1 = torch.nn.Linear(D_in, H).cuda()
module2 = torch.nn.Linear(H, D_out).cuda()
module3 = torch.nn.Linear(H, D_out).cuda()

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(chain(module1.parameters(),
                                  module2.parameters(),
                                  module3.parameters()),
                            lr=0.1)

# 用于捕获的示例输入
# requires_grad样本输入的状态必须匹配
# requires_grad每个可调用项将看到的实际输入的状态。
x = torch.randn(N, D_in, device='cuda')
h = torch.randn(N, H, device='cuda', requires_grad=True)

module1 = torch.cuda.make_graphed_callables(module1, (x,))
module2 = torch.cuda.make_graphed_callables(module2, (h,))
module3 = torch.cuda.make_graphed_callables(module3, (h,))

real_inputs = [torch.rand_like(x) for _ in range(10)]
real_targets = [torch.randn(N, D_out, device="cuda") for _ in range(10)]

for data, target in zip(real_inputs, real_targets):
    optimizer.zero_grad(set_to_none=True)

    tmp = module1(data)  # 正向运算以图形形式运行

    if tmp.sum().item() > 0:
        tmp = module2(tmp)  # 正向运算以图形形式运行
    else:
        tmp = module3(tmp)  # 正向运算以图形形式运行

    loss = loss_fn(tmp, target)
    # 模块 2 或模块 3（以选择者为准）的向后运算，
    # 以及模块 1 的向后操作，以图形形式运行
    loss.backward()
    optimizer.step()
```

### with torch.cuda.amp

对于典型的优化器， [`GradScaler.step`](https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler.step) 会将 CPU 与 GPU 同步，这在捕获过程中是被禁止的。若要避免错误，请使用[部分网络捕获](https://pytorch.org/docs/stable/notes/cuda.html#partial-network-capture)，或者(如果前进、丢失和向后捕获是捕获安全的)向前、丢失和向后捕获，但不使用优化器步骤： 

```python
# 预热
# 在真实设置中，使用几批真实数据
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for i in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            y_pred = model(static_input)
            loss = loss_fn(y_pred, static_target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
torch.cuda.current_stream().wait_stream(s)

# 捕获
g = torch.cuda.CUDAGraph()
optimizer.zero_grad(set_to_none=True)
with torch.cuda.graph(g):
    with torch.cuda.amp.autocast():
        static_y_pred = model(static_input)
        static_loss = loss_fn(static_y_pred, static_target)
    scaler.scale(static_loss).backward()
    # 不捕获缩放器（优化器）或缩放器更新（）

real_inputs = [torch.rand_like(static_input) for _ in range(10)]
real_targets = [torch.rand_like(static_target) for _ in range(10)]

for data, target in zip(real_inputs, real_targets):
    static_input.copy_(data)
    static_target.copy_(target)
    g.replay()
    # 急切地运行缩放器步骤和缩放器更新
    scaler.step(optimizer)
    scaler.update()
```

### with multiple streams

捕获模式会自动传播到与捕获流同步的任何流。在捕获中可以通过向不同流发出调用来公开并行性，但总体流依赖项 DAG 必须在捕获开始后从初始捕获流分支出来，并在捕获结束之前重新加入初始流： 

```python
with torch.cuda.graph(g):
   # 在上下文管理器入口处，torch.cuda.current_stream（）
    # 是初始捕获流

	# 不正确（不从初始流分支或重新加入初始流）
    with torch.cuda.stream(s):
        cuda_work()

    # 正确：
    # 从初始流分支出来
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        cuda_work()
    # 在捕获结束前重新加入初始流
    torch.cuda.current_stream().wait_stream(s)
```

**注意：**

为避免高级用户在 nsight 系统或 nvprof 中查看重放时感到困惑：与急切执行不同，该图将捕获中的非平凡流 DAG 解释为提示，而不是命令。在重播期间，图形可能会将独立的操作重新组织到不同的流上，或者以不同的顺序将它们排入队列 

## 数据并行

于分布式数据并行的用法：

#### NCCL < 2.9.6

早于 2.9.6 的 NCCL 版本不允许捕获集合体。您必须使用[部分网络捕获](https://pytorch.org/docs/stable/notes/cuda.html#partial-network-capture)，这将延迟所有减少发生在向后图形部分之外。

在用 DDP 包装网络*之前*，在可图形化的网络部分上调用 [`make_graphed_callables()`](https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html#torch.cuda.make_graphed_callables)。

#### NCCL >= 2.9.6

NCCL 版本 2.9.6 或更高版本允许在图形中使用集合体。捕获[整个向后通道](https://pytorch.org/docs/stable/notes/cuda.html#whole-network-capture)的方法是一个可行的选择，但需要三个设置步骤。

1、禁用DDP的内部异步错误处理：

```python
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
torch.distributed.init_process_group(...)
```

2、在完全向后捕获之前，必须在测流上下文中构造DDP：

```python
with torch.cuda.stream(s):
    model = DistributedDataParallel(model)
```

3、在捕获之前，必须提前进行预热，至少运行11次启用DDP的预先迭代

## 图形内存管理

每次重放时，捕获的图形都会作用于相同的虚拟地址。如果PyTorch释放了内存，则稍后的重播可能会遇到非法内存访问。如果 PyTorch 将内存重新分配给新的张量，则重放可能会损坏这些张量所看到的值。因此，必须为图形在重播期间保留图形使用的虚拟地址。PyTorch 缓存分配器通过检测捕获何时进行并从图私有内存池中满足捕获的分配来实现此目的。私人游泳池将保持活动状态，直到其 [`CUDA图形`](https://pytorch.org/docs/stable/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph)对象和捕获期间创建的所有张量超出范围。 

私有的内存池子会自动进行维护，在默认情况下，分配器会为每个捕获创建一个单独的专用池，如果捕获多个图形，这种保守的方法可确保图形重放永远不会破坏彼此的值，但优势会不必要的浪费内存。

1、在捕获之间共享内存

为了节省隐藏在私有的内存池中的记忆， [`torch.cuda.graph`](https://pytorch.org/docs/stable/generated/torch.cuda.graph.html#torch.cuda.graph) 和 [`torch.cuda.make_graphed_callables()`](https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html#torch.cuda.make_graphed_callables) 可选地允许不同的捕获共享同一个私人内存池。如果知道一组图形将始终以捕获的相同顺序重播，并且永远不会同时重播，则可以安全地共享一个私人游泳池。 

2、[`torch.cuda.graph`](https://pytorch.org/docs/stable/generated/torch.cuda.graph.html#torch.cuda.graph)参数是使用特定专用池的提示，可用于在图形之间共享内存，如下所示：`pool` 

```python
g1 = torch.cuda.CUDAGraph()
g2 = torch.cuda.CUDAGraph()

# （为 g1 和 g2 创建静态输入，运行其工作负载的预热...）

# 捕获 g1
with torch.cuda.graph(g1):
    static_out_1 = g1_workload(static_in_1)

# 捕获 g2，暗示 g2 可能与 g1 共享一个内存池
with torch.cuda.graph(g2, pool=g1.pool()):
    static_out_2 = g2_workload(static_in_2)

static_in_1.copy_(real_data_1)
static_in_2.copy_(real_data_2)
g1.replay()
g2.replay()
```

 [`torch.cuda.make_graphed_callables()`](https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html#torch.cuda.make_graphed_callables) ，如果想绘制几个可调用对象的图形，并且知道它们将始终以相同的顺序运行(并且从不同时运行)，则将它们作为元组传递，它们将在实时工作负载中运行的相同顺序， [`make_graphed_callables()`](https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html#torch.cuda.make_graphed_callables) 将使用共享的专用池捕获它们的图形。

如果在实时工作负荷中，可调用项将按偶尔更改的顺序运行，或者如果它们将同时运行，则不允许将它们作为元组传递给  [`make_graphed_callables()`](https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html#torch.cuda.make_graphed_callables) 的单个调用。相反，您必须为每个单独调用 [`make_graphed_callables()`](https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html#torch.cuda.make_graphed_callables) 

