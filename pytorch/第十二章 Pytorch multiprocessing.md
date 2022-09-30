# 第十二章 Pytorch  Multiprocessing 

## Multiprocessing简介

torch. multiprocessing是 [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing)模块包装器，它注册自定义化简器，这些化简器使用共享内存在不同进程中提供相同数据的共享视图， 一旦张量/存储移动到shared_memory([`share_memory_()`](https://pytorch.org/docs/stable/generated/torch.Tensor.share_memory_.html#torch.Tensor.share_memory_))，就可以将其发送到其他进程而无需制作任何副本。 

API与原始模块100%兼容，只需更改即可将所有张量通过队列发送或通过其他机制共享，移动到共享内存。`import multiprocessing` `import torch.multiprocessing` 

**注意：**

如果主进程突然退出(例如，由于传入的信号)，Python有时无法清理其子进程。这是一个已知的警告，所以如果在打断解释器后看到任何资源泄漏，这可能意味着这刚刚发生过。

## 战略管理

1、get_all_sharing_strategies()

返回当前系统支持的一组共享策略

```python
torch.multiprocessing.get_all_sharing_strategies()
```

2、get_sharing_strategy()

返回用于共享CPU张量的当前策略

```python
torch.multiprocessing.get_sharing_strategy()
```

3、set_sharing_strategy

设置共享CPU张量的策略

```python
torch.multiprocessing.set_sharing_strategy(new_strategy)
```

参数：

-   **new_strategy** ([*str*](https://docs.python.org/3/library/stdtypes.html#str))：所选策略的名称。应为 返回的值之[`get_all_sharing_strategies()`](https://pytorch.org/docs/stable/multiprocessing.html#torch.multiprocessing.get_all_sharing_strategies). 

source：

```python
"""
火炬多处理是围绕本机的包装：mod：“多处理”模块。它注册自定义化简器，这些化简器使用共享内存来提供共享
不同进程中相同数据的视图。一旦张量/存储被移动到shared_memory（参见：func：'~火炬。Tensor.share_memory_'），
这将是可能的将其发送到其他进程，而无需制作任何副本。

API与原始模块100%兼容 - 足以更改“导入多处理”到“导入火炬”，多处理“具有所有
张量通过队列发送或通过其他机制共享，移动到共享记忆。

由于API的相似性，我们没有记录这个软件包的大部分内容内容，我们建议参考原始模块的非常好的文档。
"""
import torch
import sys
from .reductions import init_reductions
import multiprocessing

__all__ = ['set_sharing_strategy', 'get_sharing_strategy',
           'get_all_sharing_strategies']


from multiprocessing import *  # noqa: F403


__all__ += multiprocessing.__all__  # type: ignore[attr-defined]

torch._C._multiprocessing_init()


"""添加帮助程序函数以生成 N 个进程并等待任何进程完成他们。这取决于Python 3.4中添加的“mp.get_context”。"""
from .spawn import spawn, SpawnContext, start_processes, ProcessContext, \
    ProcessRaisedException, ProcessExitedException


if sys.platform == 'darwin' or sys.platform == 'win32':
    _sharing_strategy = 'file_system'
    _all_sharing_strategies = {'file_system'}
else:
    _sharing_strategy = 'file_descriptor'
    _all_sharing_strategies = {'file_descriptor', 'file_system'}



def set_sharing_strategy(new_strategy):
    """设置共享 CPU 张量的策略。
	参数：
        new_strategy（str）：所选策略的名称。应为以下之一
        由 ：func：'get_all_sharing_strategies（）' 返回的值。
    
    """
    global _sharing_strategy
    assert new_strategy in _all_sharing_strategies
    _sharing_strategy = new_strategy


def get_sharing_strategy():
    """返回用于共享 CPU 张量的当前策略。"""
    return _sharing_strategy

def get_all_sharing_strategies():
    """返回当前系统支持的一组共享策略。"""
    return _all_sharing_strategies

init_reductions()
```

## 共享库达张量

仅在 Python 3 中支持在进程之间共享 CUDA 张量，使用 a 或 start 方法。`spawn` `forkserver`，与CPU张量不同，只要接收过程保留张量的副本，发送过程就需要保留原始张量。重新计数是在后台实现的，但要求用户遵循下一个最佳实践

**注意：**

如果使用者进程因致命信号而异常死亡，则只要发送进程正在运行，共享张量就可以永远保存在内存中 

1、尽快释放使用者中的内存

```python
## Good
x = queue.get()
# 用 x 做点什么
del x
```

```python
## Bad
x = queue.get()

# 执行其他所有操作（生产者必须将 x 保存在内存中）
```

2、保持生产者进程运行，直到所有消费者退出。这将防止生产者进程释放使用者仍在使用的内存的情况 

```python
## 生产者
# 发送张量，做点什么
event.wait()
```

```python
## 消费者
# 接收张量并使用它们
event.set()
```

3、不传递接收到的张量

```python
# 无法工作
x = queue.get()
queue_2.put(x)
```

```python
# 需要创建进程本地副本
x = queue.get()
x_clone = x.clone()
queue_2.put(x_clone)
```

```python
# 在同一进程中放置和从同一队列中获取可能会以 segfault 结束
queue.put(tensor)
x = queue.get()
```

**注意：**

它仅仅适用CPU张量，CUDA张量将始终使用CUDA API，因为这是它们可用共享的唯一方式

## 文件描述符

**注意：这是默认策略**

此策略将使用文件描述符作为共享内存句柄。每当将存储移动到共享内存时，从中获取的文件描述符都会与对象一起缓存，当它要发送到其他进程时，文件描述符将被传输（例如通过UNIX套接字）到它。接收方还将缓存文件描述符和它，以获得存储数据的共享视图。`shm_open` `mmap`

请注意，如果将共享大量张量，则此策略将在大多数时间保持大量文件描述符处于打开状态。如果系统对打开的文件描述符的数量有较低的限制，并且无法提高它们，则应使用该策略。`file_system`

## 文件系统

此策略将使用给定的文件名来标识共享内存区域。这样做的好处是不需要实现缓存从中获取的文件描述符，但同时容易发生共享内存泄漏。文件在创建后无法立即删除，因为其他进程需要访问它才能打开其视图。如果进程致命地崩溃或被终止，并且不调用存储析构函数，则文件将保留在系统中。这是非常严重的，因为它们会一直耗尽内存，直到系统重新启动，或者手动释放它们。`shm_open`

为了解决共享内存文件泄漏的问题，[`torch.multiprocessing`](https://pytorch.org/docs/stable/multiprocessing.html#module-torch.multiprocessing) 将生成一个名为的守护程序，该守护程序将自身与当前进程组隔离，并将跟踪所有共享内存分配。一旦连接到它的所有进程都退出，它将等待片刻以确保没有新连接，并将迭代组分配的所有共享内存文件。如果它发现它们中的任何一个仍然存在，它们将被解除分配。我们已经测试了这种方法，事实证明它对各种故障都是健壮的。不过，如果您的系统具有足够高的限制，并且是受支持的策略，则不建议切换到此策略。`torch_shm_manager` `file_descriptor`

## 生成子流程

生成许多子进程以执行某些功能可以通过创建实例并调用以等待其完成来完成。此方法在处理单个子流程时工作正常，但在处理多个流程时会出现潜在问题。`Process` `join`

也就是说，按顺序加入进程意味着它们将按顺序终止。如果它们不这样做，并且第一个进程没有终止，则进程终止将不被注意到。此外，没有用于错误传播的本机工具。

下面的函数解决了这些问题，并负责错误传播、无序终止，并在检测到其中一个进程中的错误时主动终止进程。`spawn`

```python
torch.multiprocessing.spawn(fn, args=(), nprocs=1, join=True, daemon=False, start_method='spawn')
```

生成使用 运行的进程。`nprocs` `fn` `args`，如果其中一个进程以非零退出状态退出，则将终止其余进程，并引发异常并显示终止原因。如果在子进程中捕获了异常，则会转发该异常，并将其回溯包含在父进程中引发的异常中。

参数：

-   **fn** (*function*)：

    函数被调用为生成进程的入口点。此函数必须在模块的顶层定义，以便可以对其进行腌制和生成。这是多处理强加的要求。

    该函数称为进程索引，并且是参数元组传递的函数。

-   **args** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple))：传递给fn的参数

-   **nprocs** ([*int*](https://docs.python.org/3/library/functions.html#int))：要生成的进程数

-   **join** ([*bool*](https://docs.python.org/3/library/functions.html#bool))：对所有进程执行阻塞联接.

-   **daemon** ([*bool*](https://docs.python.org/3/library/functions.html#bool))：生成的进程的守护程序标志。如果设置为 True，则将创建守护进程。

-   **start_method** (*string*)：（已弃用）此方法将始终用作启动方法。要使用其他启动方法，请使用生成``start_processes（)``

返回：

如果是join则为True，如果是ProcessContext join 则为False

source：

```python
# 注意： [start_processes]
# mp.start_processes处理start_method=“生成”和“分叉”。它应该是一个
# 比 mp 生成更通用的 API。目前，我们只记录 mp.spawn，因为它是
# 兼容库达start_method。但是，在像Ipython笔记本这样的环境中，“分叉”
# 比“生成”效果更好。我们为 mp.spawn 创建的每个帮助程序函数确实是
# 足够通用，像XLA这样的后端也可以在Colab笔记本中重用它们。
# 目前我们只先添加这个API，我们可以考虑把它添加到文档中
# 将来需要。
def start_processes(fn, args=(), nprocs=1, join=True, daemon=False, start_method='spawn'):
    mp = multiprocessing.get_context(start_method)
    error_queues = []
    processes = []
    for i in range(nprocs):
        error_queue = mp.SimpleQueue()
        process = mp.Process(
            target=_wrap,
            args=(fn, i, args, error_queue),
            daemon=daemon,
        )
        process.start()
        error_queues.append(error_queue)
        processes.append(process)

    context = ProcessContext(processes, error_queues)
    if not join:
        return context

   # 循环连接，直到返回 True 或引发异常。
    while not context.join():
        pass


def spawn(fn, args=(), nprocs=1, join=True, daemon=False, start_method='spawn'):
    r"""生成运行带有“参数”的“fn”的进程。
	如果其中一个进程以非零退出状态退出，则剩余的进程将被终止，并且异常将使用
    终止原因。在以下情况下，在子进程，它被转发，其回溯包含在在父进程中引发的异常。

    参数：
        fn（函数）：函数被调用为生成进程。此函数必须在顶部定义
        模块的级别，以便可以对其进行腌制和生成。这是多处理强加的要求。

        该函数称为“fn（i， *args）”，其中“i”是进程索引和“args”是传递元组的参数。

        参数（元组）：传递给“fn”的参数。
        n过程（整数）：要生成的进程数。
        加入（bool）：对所有进程执行阻止联接。
        守护进程 （bool）：生成的进程的守护程序标志。如果设置为 True，将创建守护进程。
       start_method（字符串）：（已弃用）此方法将始终使用“生成”作为启动方法。使用其他启动方法
                               使用“start_processes（）”。

    返回：
        如果“加入”是“真”，则无，
        ：类：“~进程上下文”，如果“加入”是“假”

    """
    if start_method != 'spawn':
        msg = ('This method only supports start_method=spawn (got: %s).\n'
               'To use a different start_method use:\n\t\t'
               ' torch.multiprocessing.start_processes(...)' % start_method)
        warnings.warn(msg)
    return start_processes(fn, args, nprocs, join, daemon, start_method='spawn')
```

调用时，生成[`spawn()`](https://pytorch.org/docs/stable/multiprocessing.html#torch.multiprocessing.spawn) join = Fase

```python
torch.multiprocessing.SpawnContext
```

-    `join`(*timeout=None*) 

尝试在此生成上下文中加入一个或多个进程。如果其中一个进程以非零退出状态退出，则此函数将终止其余进程，并引发异常，导致第一个进程退出。如果所有进程都已成功加入，则返回该消息，如果还有更多进程需要加入

-   参数：

     **timeout** ([*float*](https://docs.python.org/3/library/functions.html#float))：等待一段时间进行释放

source：

```python
class SpawnContext(ProcessContext):
    def __init__(self, processes, error_queues):
        warnings.warn('SpawnContext is renamed to ProcessContext since 1.4 release.')
        super(SpawnContext, self).__init__(processes, error_queues)
    pass
```

## Multiprocessing源码

### pool.py

```python
import multiprocessing.pool
import multiprocessing.util as util

from .queue import SimpleQueue


def clean_worker(*args, **kwargs):
    """
    常规的多进程工作人员不会完全清理自己，因此我们必须显式触发垃圾收集以确保调用所有析构函数......
    """
    import gc
    multiprocessing.pool.worker(*args, **kwargs)

    gc.collect()


class Pool(multiprocessing.pool.Pool):
    """
    版本的池实现。这让我们可以跨进程传递共享内存中的张量，而不是序列化底层数据。
    """

    def _setup_queues(self):
        self._inqueue = SimpleQueue()
        self._outqueue = SimpleQueue()
        self._quick_put = self._inqueue._writer.send
        self._quick_get = self._outqueue._reader.recv

    def _repopulate_pool(self):
        """
        将池进程的数量增加到指定的数量，以在收获已退出的工人后使用
        """
        for i in range(self._processes - len(self._pool)):
            # changed worker -> clean_worker
            args = (self._inqueue, self._outqueue,
                    self._initializer,
                    self._initargs, self._maxtasksperchild)
            if hasattr(self, '_wrap_exception'):
                args += (self._wrap_exception,)
            w = self.Process(target=clean_worker, args=args)
            self._pool.append(w)
            w.name = w.name.replace('Process', 'PoolWorker')
            w.daemon = True
            w.start()
            util.debug('added worker')
```

### queue.py

```python
import io
import multiprocessing.queues
from multiprocessing.reduction import ForkingPickler
import pickle


class ConnectionWrapper(object):
    """
    _multiprocessing.Connection 的代理类，它使用 ForkingPickler 序列化对象
    """

    def __init__(self, conn):
        self.conn = conn

    def send(self, obj):
        buf = io.BytesIO()
        ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(obj)
        self.send_bytes(buf.getvalue())

    def recv(self):
        buf = self.recv_bytes()
        return pickle.loads(buf)

    def __getattr__(self, name):
        if 'conn' in self.__dict__:
            return getattr(self.conn, name)
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, 'conn'))


class Queue(multiprocessing.queues.Queue):

    def __init__(self, *args, **kwargs):
        super(Queue, self).__init__(*args, **kwargs)
        self._reader: ConnectionWrapper = ConnectionWrapper(self._reader)
        self._writer: ConnectionWrapper = ConnectionWrapper(self._writer)
        self._send = self._writer.send
        self._recv = self._reader.recv


class SimpleQueue(multiprocessing.queues.SimpleQueue):

    def _make_methods(self):
        if not isinstance(self._reader, ConnectionWrapper):
            self._reader: ConnectionWrapper = ConnectionWrapper(self._reader)
            self._writer: ConnectionWrapper = ConnectionWrapper(self._writer)
        super(SimpleQueue, self)._make_methods()  # type: ignore[misc]
```

### reductions.py

```python
import torch
import torch.utils.hooks
from torch._namedtensor_internals import check_serializing_named_tensor
import os
import threading
import multiprocessing
from multiprocessing.util import register_after_fork
from multiprocessing.reduction import ForkingPickler
from typing import Union

try:
    """"
    提前加载 resource_sharer 以防止部分初始化的实例在分叉的子进程中被继承。 
    reduce_storage 方法通过 DupFd() 间接需要此模块。
    内置的 mp.Queue 类在可能与 fork 重叠的后台线程中腌制参数
    """
    import multiprocessing.resource_sharer
except ImportError:
    pass


class StorageWeakRef(object):
    """
    对存储的弱引用。 cdata 成员是一个 Python 数字，包含存储指针的整数表示
    """

    def __init__(self, storage):
        self.cdata = storage._weak_ref()
        # 保存对 _free_weak_ref 的直接引用，
        # 因为 `torch` 模块可能会在 Python 关闭期间在此模块被清除之前被清除。
        self._free_weak_ref = torch.Storage._free_weak_ref  # type: ignore[attr-defined]

    def expired(self):
        return torch.Storage._expired(self.cdata)  # type: ignore[attr-defined]

    def __del__(self):
        self._free_weak_ref(self.cdata)

    def __hash__(self):
        return self.cdata

    def __eq__(self, other):
        if id(self) == id(other):
            return True
        return self.cdata == other.cdata


class SharedCache(dict):
    """
    从多处理句柄到 StorageWeakRef 的字典
    """

    def __init__(self):
        # 如果 len 超过当前限制，则调用 free_dead_references()。该限制随剩余活动对象的数量而变化
        self.limit = 128
        # `fork` 继承了锁的状态，所以如果我们在持有锁的时候分叉，
        # 会注册一个函数来将锁重置为一个新的对象，以避免可能的死锁，遵循 python 多处理库设计
        self._after_fork()
        register_after_fork(self, SharedCache._after_fork)

    def _after_fork(self):
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            return dict.get(self, key)

    def __setitem__(self, key, storage_ref):
        with self.lock:
            dict.__setitem__(self, key, storage_ref)
            if len(self) > self.limit:
                self.free_dead_references()

    def free_dead_references(self):
        live = 0
        for key, storage_ref in list(self.items()):
            if storage_ref.expired():
                del self[key]
            else:
                live += 1
        self.limit = max(128, live * 2)


# mapping from handles to StorageWeakRef objects
shared_cache = SharedCache()


def rebuild_event(device, handle):
    return torch.cuda.Event.from_ipc_handle(device, handle)


def reduce_event(event):
    handle = event.ipc_handle()
    return (rebuild_event, (event.device, handle))


def rebuild_tensor(cls, storage, metadata):
    storage_offset, size, stride, requires_grad = metadata
    t = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
    if cls == torch.nn.parameter.Parameter:
        # 必须将 requires_grad 传递给构造函数，而不是稍后将其设置为属性，
        # 因为它是整数张量是否具有 requires_grad=False 的重要检查（否则它们会引发错误）
        t = torch.nn.parameter.Parameter(t, requires_grad=requires_grad)
    else:
        t.requires_grad = requires_grad
    return t


def rebuild_cuda_tensor(tensor_cls, tensor_size, tensor_stride, tensor_offset,
                        storage_cls, dtype, storage_device, storage_handle, storage_size_bytes, storage_offset_bytes,
                        requires_grad, ref_counter_handle, ref_counter_offset, event_handle, event_sync_required):
    # 如果 storage_handle 为 None，则 storage 指向 nullptr
    if storage_handle is None or storage_size_bytes == 0:
        storage = storage_cls(0, dtype=dtype, device=storage_device)
    else:
        storage = storage_from_cache(storage_cls, (storage_handle, storage_offset_bytes))
        if storage is None:
            torch.cuda._lazy_init()
            storage = storage_cls._new_shared_cuda(
                storage_device,
                storage_handle,
                storage_size_bytes,
                storage_offset_bytes,
                ref_counter_handle,
                ref_counter_offset,
                event_handle,
                event_sync_required)
            shared_cache[(storage_handle, storage_offset_bytes)] = StorageWeakRef(storage)
        else:
            # We already ref counting this Storage, but producer needs new ref-counters to be released.
            storage_cls._release_ipc_counter(ref_counter_handle, ref_counter_offset, device=storage_device)

    t = torch._utils._rebuild_tensor(
        torch.storage.TypedStorage(wrap_storage=storage.untyped(), dtype=dtype),
        tensor_offset, tensor_size, tensor_stride)

    if tensor_cls == torch.nn.parameter.Parameter:
        # It is crucial for integer tensors to receive
        # the requires_grad=False as an argument in the constructor
        t = torch.nn.parameter.Parameter(t, requires_grad=requires_grad)
    else:
        t.requires_grad = requires_grad

    return t


def reduce_tensor(tensor):
    storage = tensor.storage()

    if tensor.requires_grad and not tensor.is_leaf:
        raise RuntimeError("Cowardly refusing to serialize non-leaf tensor which requires_grad, "
                           "since autograd does not support crossing process boundaries.  "
                           "If you just want to transfer the data, call detach() on the tensor "
                           "before serializing (e.g., putting it on the queue).")

    check_serializing_named_tensor(tensor)
    torch.utils.hooks.warn_if_has_hooks(tensor)

    """
    注意【CUDA IPC和缓存分配器】~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~当通过 IPC 发送一个 CUDA 张量时，
    可能期望会从另一端得到相同的存储。但是，CUDA 缓存分配器很难保持这个不变性。考虑以下情况：大小为 0x100 的张量
    指向大小为 0x100 的存储在 0xA100 处的偏移量 0x20。 （为简单起见，所有这些大小都以字节为单位）。但是，使用缓
    存分配器，此存储可能是大小为 0x4000 的更大 cudaMalloc 分配 0xA000 的一部分。当想通过 IPC 发送这个 CUDA 张量
    时，必须发送整个 cudaMalloc 分配，即 0xA000 区域，而不仅仅是存储 0xA100（因为这是 CUDA 支持的）。所以，在另
    一端，根本没有办法说，“等等，你给了我一个比我想要的（0xA100）更大的区域（0xA000）”。好的，所以如果你发送了 
    cudaMalloc 分配，你能把它包装成一个存储本身吗？不，因为这个 cudaMalloc 分配可能包含混合类型的存储：float、
    bytes、double ...贮存。cudaIpcMemHandle 是在接收方访问发送方 cudaMalloc 分配的标识符。但是，给定进程中每个
    设备的 cudaIpcMemHandles 只能由每个设备每个其他进程的一个上下文打开。如果我们在一个进程中多次打开和关闭一个
    内存句柄，CUDA 可以给它一个不同的地址；同样，一旦我们关闭内存，我们就不能访问它（以及建立在它之上的 storagetensor），
    即使它仍然存在于原始进程中。由于我们不能一次性将 cudaMalloc 分配给单个存储，这需要我们在 C++ 端缓存每个 cudaIpcMemHandle 的设备指针，以重建存储类型，
    同时保持旧存储的存活。请参阅 [https:docs.nvidia.comcudacuda-runtime-apigroup__CUDART__DEVICE.html] 这很好，
    因为我们需要做的就是保存我们在分配中的位置，并从中重建存储和张量。
    """
    # and reconstruct storage and tensor from it.
    # 0xA000 ->  -------CUDA Allocation------
    #           |                            |
    #           |                            |
    #           |                            |
    #           |                            |
    # 0xA100 ->  --------storage1 begin------
    #           |                            |
    # 0xA120 ->  --------tensor1 begin ------
    #           |                            |
    #           |                            |
    #           |                            |
    #           |                            |
    #           |                            |
    # 0xA160 ->  --------tensor1 end---------
    #           |                            |
    #           |                            |
    #           |                            |
    # 0xA200 ->  --------storage1 end--------
    #           |                            |
    # 0xE000 ->  --------CUDA allocation-----
    #
    """
    要发送 tensor1，需要从发送方到接收方的以下信息用于存储重构。 1. 0xA000的cudaIpcMemHandle（可以在receiver进程中映射到一个basePtr）。 
    basePtr 可能不完全是 0xA000，因为它是一个不同的过程。 2. CUDA分配中storage1的偏移量(0xA100)。 3. storage1(0x100)的大小。
    在接收端： 
    1. 获取 MemHandle 的 devPtr 以访问内存，使用 (basePtr, offset, size) 重构相同类型的存储。 
    2. 我们可以在重构存储Tensor(size=0x040, offset=0x020, storage=Storage(data=basePtr+0xA100, size=0x0100)) 之上重构张量这个策略有几个含义： 
    1. 当我们序列化用于 IPC 的 CUDA 张量，我们不能一次性完成所有操作（非组合），这需要每个进程都有一个全局映射 memHandle -> devPtr。 
    2. 我们不能让新的 IPC 张量可调整大小。最初，超过 0x100 的存储大小调整只会导致我们进行重新分配。您并不是真的想这样做，但如果您这样做了，那么您将失去 IPC 共享。
    但是，如果您在新世界中这样做，我们将很乐意让您写出您的“分配”范围，破坏缓存分配器块中的不相关数据。坏的！顺便说一句，在旧版本的 PyTorch 中，
    我们使用“存储视图”原生支持这种情况，它允许多个存储成为彼此的视图。但这是存储视图的唯一用途，所以我们取消了它，以便我们可以使用张量视图来实现相同的东西。
    """
    #
    if storage.is_cuda:
        (device,
         handle,
         storage_size_bytes,
         storage_offset_bytes,
         ref_counter_handle,
         ref_counter_offset,
         event_handle,
         event_sync_required) = storage._share_cuda_()
        tensor_offset = tensor.storage_offset()
        shared_cache[handle] = StorageWeakRef(storage)
        # _backward_hooks purposely omitted here, see
        # Note [Don't serialize hooks]
        return (rebuild_cuda_tensor,
                (type(tensor),
                 tensor.size(),
                 tensor.stride(),
                 tensor_offset,  # tensor offset in its storage
                 type(storage),
                 tensor.dtype,
                 device,
                 handle,  # identifier which CUDA allocation is the storage in.
                 storage_size_bytes,  # size(in bytes) of the storage
                 storage_offset_bytes,  # offset(in bytes) of the storage in the CUDA allocation
                 tensor.requires_grad,
                 ref_counter_handle,
                 ref_counter_offset,
                 event_handle,
                 event_sync_required))

    # _backward_hooks purposely omitted here, see Note [Don't serialize hooks]
    metadata = (tensor.storage_offset(), tensor.size(), tensor.stride(), tensor.requires_grad)
    return (rebuild_tensor, (
        type(tensor),
        storage,
        metadata))


def fd_id(fd):
    # Returns a tuple which uniquely identifies a file descriptor. In Mac OS,
    # this doesn't work with shared memory handles, which is why we don't
    # support the "file_descriptor" sharing method on that platform.
    stat = os.fstat(fd)
    return (stat.st_ino, stat.st_dev)


def storage_from_cache(cls, key):
    storage_ref = shared_cache.get(key)
    if storage_ref is None:
        return None
    return torch.UntypedStorage._new_with_weak_ptr(storage_ref.cdata)


def rebuild_storage_fd(cls, df, size):
    fd = df.detach()
    try:
        storage = storage_from_cache(cls, fd_id(fd))
        if storage is not None:
            return storage
        storage = cls._new_shared_fd_cpu(fd, size)
        shared_cache[fd_id(fd)] = StorageWeakRef(storage)
        return storage
    finally:
        os.close(fd)


def rebuild_storage_filename(cls, manager, handle, size, dtype=None):
    storage: Union[torch.TypedStorage, torch.UntypedStorage] = storage_from_cache(cls, handle)
    if storage is not None:
        return storage._shared_decref()
    if dtype is None:
        storage = torch.UntypedStorage._new_shared_filename_cpu(manager, handle, size)
    else:
        byte_size = size * torch._utils._element_size(dtype)
        untyped_storage: torch.UntypedStorage = torch.UntypedStorage._new_shared_filename_cpu(manager, handle, byte_size)
        storage = torch.TypedStorage(
            wrap_storage=untyped_storage,
            dtype=dtype)
    shared_cache[handle] = StorageWeakRef(storage)
    return storage._shared_decref()


def rebuild_storage_empty(cls):
    return cls()

def rebuild_typed_storage(storage, dtype):
    return torch.storage.TypedStorage(wrap_storage=storage, dtype=dtype)

# Use for torch.storage.TypedStorage
def reduce_typed_storage(storage):
    return (rebuild_typed_storage, (storage._storage, storage.dtype))

def rebuild_typed_storage_child(storage, storage_type):
    return storage_type(wrap_storage=storage)

# Use for child classes of torch.storage.TypedStorage, like torch.FloatStorage
def reduce_typed_storage_child(storage):
    return (rebuild_typed_storage_child, (storage._storage, type(storage)))

def reduce_storage(storage):
    from . import get_sharing_strategy
    if storage.is_cuda:
        raise RuntimeError("Cannot pickle CUDA storage; try pickling a CUDA tensor instead")
    elif get_sharing_strategy() == 'file_system':
        metadata = storage._share_filename_cpu_()
        cache_key = metadata[1]
        rebuild = rebuild_storage_filename
        if isinstance(storage, torch.TypedStorage):
            metadata += (storage.dtype,)
        storage._shared_incref()
    elif storage.size() == 0:
        # This is special cased because Empty tensors
        # (with size 0) cannot be mmapped.
        return (rebuild_storage_empty, (type(storage),))
    else:
        fd, size = storage._share_fd_cpu_()
        df = multiprocessing.reduction.DupFd(fd)
        cache_key = fd_id(fd)
        metadata = (df, size)
        rebuild = rebuild_storage_fd  # type: ignore[assignment]

    shared_cache[cache_key] = StorageWeakRef(storage)
    return (rebuild, (type(storage),) + metadata)


def init_reductions():
    ForkingPickler.register(torch.cuda.Event, reduce_event)

    for t in torch._storage_classes:
        if t.__name__ == 'UntypedStorage':
            ForkingPickler.register(t, reduce_storage)
        else:
            ForkingPickler.register(t, reduce_typed_storage_child)

    ForkingPickler.register(torch.storage.TypedStorage, reduce_typed_storage)

    for t in torch._tensor_classes:
        ForkingPickler.register(t, reduce_tensor)

    # TODO: Maybe this should be in tensor_classes? :)
    ForkingPickler.register(torch.Tensor, reduce_tensor)
    ForkingPickler.register(torch.nn.parameter.Parameter, reduce_tensor)

```

### spawn.py

```python

from typing import Optional
import multiprocessing
import multiprocessing.connection
import signal
import sys
import warnings

from . import _prctl_pr_set_pdeathsig  # type: ignore[attr-defined]


class ProcessException(Exception):
    __slots__ = ["error_index", "error_pid"]

    def __init__(self, msg: str, error_index: int, pid: int):
        super().__init__(msg)
        self.msg = msg
        self.error_index = error_index
        self.pid = pid

    def __reduce__(self):
        return type(self), (self.msg, self.error_index, self.pid)


class ProcessRaisedException(ProcessException):
    """
    Exception is thrown when the process failed due to exception
    raised by the code.
    """
    def __init__(
        self,
        msg: str,
        error_index: int,
        error_pid: int,
    ):
        super().__init__(msg, error_index, error_pid)


class ProcessExitedException(ProcessException):
    """
    Exception is thrown when the process failed due to signal
    or exited with a specific code.
    """
    __slots__ = ["exit_code"]

    def __init__(
            self, msg: str, error_index: int, error_pid: int,
            exit_code: int, signal_name: Optional[str] = None
    ):
        super().__init__(msg, error_index, error_pid)
        self.exit_code = exit_code
        self.signal_name = signal_name

    def __reduce__(self):
        return (
            type(self),
            (self.msg, self.error_index, self.pid, self.exit_code, self.signal_name),
        )


def _wrap(fn, i, args, error_queue):
    # prctl(2) is a Linux specific system call.
    # On other systems the following function call has no effect.
    # This is set to ensure that non-daemonic child processes can
    # terminate if their parent terminates before they do.
    _prctl_pr_set_pdeathsig(signal.SIGINT)

    try:
        fn(i, *args)
    except KeyboardInterrupt:
        pass  # SIGINT; Killed by parent, do nothing
    except Exception:
        # Propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put(traceback.format_exc())
        sys.exit(1)


class ProcessContext:
    def __init__(self, processes, error_queues):
        self.error_queues = error_queues
        self.processes = processes
        self.sentinels = {
            process.sentinel: index
            for index, process in enumerate(processes)
        }

    def pids(self):
        return [int(process.pid) for process in self.processes]

    def join(self, timeout=None):
        r"""
        Tries to join one or more processes in this spawn context.
        If one of them exited with a non-zero exit status, this function
        kills the remaining processes and raises an exception with the cause
        of the first process exiting.

        Returns ``True`` if all processes have been joined successfully,
        ``False`` if there are more processes that need to be joined.

        Args:
            timeout (float): Wait this long before giving up on waiting.
        """
        # Ensure this function can be called even when we're done.
        if len(self.sentinels) == 0:
            return True

        # Wait for any process to fail or all of them to succeed.
        ready = multiprocessing.connection.wait(
            self.sentinels.keys(),
            timeout=timeout,
        )

        error_index = None
        for sentinel in ready:
            index = self.sentinels.pop(sentinel)
            process = self.processes[index]
            process.join()
            if process.exitcode != 0:
                error_index = index
                break

        # Return if there was no error.
        if error_index is None:
            # Return whether or not all processes have been joined.
            return len(self.sentinels) == 0

        # Assume failure. Terminate processes that are still alive.
        for process in self.processes:
            if process.is_alive():
                process.terminate()
            process.join()

        # There won't be an error on the queue if the process crashed.
        failed_process = self.processes[error_index]
        if self.error_queues[error_index].empty():
            exitcode = self.processes[error_index].exitcode
            if exitcode < 0:
                name = signal.Signals(-exitcode).name
                raise ProcessExitedException(
                    "process %d terminated with signal %s" %
                    (error_index, name),
                    error_index=error_index,
                    error_pid=failed_process.pid,
                    exit_code=exitcode,
                    signal_name=name
                )
            else:
                raise ProcessExitedException(
                    "process %d terminated with exit code %d" %
                    (error_index, exitcode),
                    error_index=error_index,
                    error_pid=failed_process.pid,
                    exit_code=exitcode
                )

        original_trace = self.error_queues[error_index].get()
        msg = "\n\n-- Process %d terminated with the following error:\n" % error_index
        msg += original_trace
        raise ProcessRaisedException(msg, error_index, failed_process.pid)


class SpawnContext(ProcessContext):
    def __init__(self, processes, error_queues):
        warnings.warn('SpawnContext is renamed to ProcessContext since 1.4 release.')
        super(SpawnContext, self).__init__(processes, error_queues)
    pass


# Note: [start_processes]
# mp.start_processes handles both start_method='spawn' and 'fork'. It's supposed to be a
# more generalized API than mp.spawn. Currently we only document mp.spawn as it's the
# CUDA compatible start_method. However, in environments like Ipython notebooks, 'fork'
# works better than 'spawn'. Every helper function we created for mp.spawn is indeed
# general enough, and backends like XLA can reuse them in Colab notebooks as well.
# Currently we only add this API first, we can consider adding it to documentation as
# needed in the future.
def start_processes(fn, args=(), nprocs=1, join=True, daemon=False, start_method='spawn'):
    mp = multiprocessing.get_context(start_method)
    error_queues = []
    processes = []
    for i in range(nprocs):
        error_queue = mp.SimpleQueue()
        process = mp.Process(
            target=_wrap,
            args=(fn, i, args, error_queue),
            daemon=daemon,
        )
        process.start()
        error_queues.append(error_queue)
        processes.append(process)

    context = ProcessContext(processes, error_queues)
    if not join:
        return context

    # Loop on join until it returns True or raises an exception.
    while not context.join():
        pass


def spawn(fn, args=(), nprocs=1, join=True, daemon=False, start_method='spawn'):
    r"""Spawns ``nprocs`` processes that run ``fn`` with ``args``.

    If one of the processes exits with a non-zero exit status, the
    remaining processes are killed and an exception is raised with the
    cause of termination. In the case an exception was caught in the
    child process, it is forwarded and its traceback is included in
    the exception raised in the parent process.

    Args:
        fn (function): Function is called as the entrypoint of the
            spawned process. This function must be defined at the top
            level of a module so it can be pickled and spawned. This
            is a requirement imposed by multiprocessing.

            The function is called as ``fn(i, *args)``, where ``i`` is
            the process index and ``args`` is the passed through tuple
            of arguments.

        args (tuple): Arguments passed to ``fn``.
        nprocs (int): Number of processes to spawn.
        join (bool): Perform a blocking join on all processes.
        daemon (bool): The spawned processes' daemon flag. If set to True,
                       daemonic processes will be created.
        start_method (str): (deprecated) this method will always use ``spawn``
                               as the start method. To use a different start method
                               use ``start_processes()``.

    Returns:
        None if ``join`` is ``True``,
        :class:`~ProcessContext` if ``join`` is ``False``

    """
    if start_method != 'spawn':
        msg = ('This method only supports start_method=spawn (got: %s).\n'
               'To use a different start_method use:\n\t\t'
               ' torch.multiprocessing.start_processes(...)' % start_method)
        warnings.warn(msg)
    return start_processes(fn, args, nprocs, join, daemon, start_method='spawn')
```