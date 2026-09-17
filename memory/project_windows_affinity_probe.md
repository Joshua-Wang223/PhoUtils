---
name: Windows 亲和掩码探测的伪句柄坑
description: GetCurrentProcess() 返回 -1，ctypes 默认按 c_int 处理会截断成 0xFFFFFFFF 使 GetProcessAffinityMask 静默失败；必须用 c_void_p(-1) 并声明 argtypes
type: project
---

## 事实

`photo_dedup.py` 的核数探测按 `cgroup v2 → cgroup v1 → sched_getaffinity → Windows 亲和掩码 → os.cpu_count`
逐层降级。Windows 上前三层都不可用（没有 cgroup，`os.sched_getaffinity` 也不存在），
**第四层是唯一能反映"进程实际可用核数"的途径**（可能小于物理核数），所以它不能坏。

它曾经就是坏的。实测（Python 3.12.9）：

```python
k = ctypes.windll.kernel32
m, s = ctypes.c_size_t(), ctypes.c_size_t()
k.GetCurrentProcess()                       # → -1
k.GetProcessAffinityMask(k.GetCurrentProcess(), ctypes.byref(m), ctypes.byref(s))
# → 0（调用失败），m.value = 0
```

**根因**：`GetCurrentProcess()` 返回的是伪句柄 `(HANDLE)-1`，即 64 位下的 `0xFFFFFFFFFFFFFFFF`。
ctypes 未声明 `restype` 时按 `c_int` 处理，拿到的是 `0xFFFFFFFF`（32 位截断），
这是个**无效句柄**，于是 `GetProcessAffinityMask` 返回 0，代码静默降级到 `os.cpu_count()`。

顺带一个坑：把 `-1` 换算出的 Python 整数直接当句柄传，会抛
`ctypes.ArgumentError: argument 1: OverflowError: int too long to convert`。

**正确写法**（已验证返回 `mask = 0b11111111`，即 8 核）：

```python
k32 = ctypes.windll.kernel32
k32.GetProcessAffinityMask.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
mask, sysmask = ctypes.c_size_t(), ctypes.c_size_t()
if k32.GetProcessAffinityMask(ctypes.c_void_p(-1), ctypes.byref(mask), ctypes.byref(sysmask)):
    n = bin(mask.value).count("1")     # → 8，来源标记 "Windows 亲和掩码"
```

同类问题：`GlobalMemoryStatusEx` 需要显式构造 `MEMORYSTATUSEX` 结构体并**先填 `dwLength`**，
否则同样返回 0；这一条实践里没出问题，但属于同一类"Windows API 靠返回值静默失败"的坑。

## Why

这类坑的危险在于**没有任何报错**。探测失败只是"回退到一个更保守的估计值"，
脚本照常运行、结果照常正确，只是**没有吃满机器**——如果没有"打印探测来源"的设计，
这个问题会一直藏着。这也是 `detect_cpu()` 一律返回 `(核数, 来源字符串)` 并打进运行日志的原因：
`逻辑核 8（os.cpu_count）` 与 `逻辑核 8（Windows 亲和掩码）` 数字一样，但可信度完全不同。

## How to apply

- **调用 Windows API 一律显式声明 `argtypes` / `restype`**，尤其是涉及指针/句柄/64 位整数的；
  伪句柄用 `ctypes.c_void_p(-1)`，不要传 Python 整数。
- **`returncode == 0` 必须当成失败处理**，不能只看有没有抛异常。
- **探测函数的返回值里带上"来源"**并打印出来。看到 `os.cpu_count` 字样就说明前面的层级都没命中，
  值得查一下是不是探测坏了（本次就是这样发现的：明明在 Windows 上却报 `os.cpu_count`）。
- 交叉验证手段：`GetProcessAffinityMask` 的第二个出参是**系统**掩码，两个值的位数可以互相对照；
  也可以用 `wmic cpu get NumberOfLogicalProcessors` 或任务管理器的逻辑处理器数核对。
