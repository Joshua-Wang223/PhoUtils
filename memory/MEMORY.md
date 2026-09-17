# PhoUtils · 工程记忆索引

这里放**工具背后的事实与踩坑**，不是 API 文档（用法看 `../README.md`）。

每条一行，指向同目录下的文件。文件格式沿用本机 codebuddy 自动记忆的约定：
`---` frontmatter 里带 `name` / `description` / `type`（project / user / feedback / reference），
正文对 project 与 feedback 类用「事实 → **Why:** → **How to apply:**」的结构，方便日后判断
这条记忆是不是还成立。

---

- [云挂载盘的写入不可信（RaiDrive 挂 z: → Quark）](project_cloud_mount_write_hazard.md)
  — 实测写入异步最终一致：探测文件 t+5s 在、t+25s **自己消失**，一个被删的文件**复活**；
  每次 `shutil.move` 进 `_duplicates/` 云端就补一份 `xxx(1).JPG`（两批 21→20、2→2，均字节相同）；
  副本时间戳还**早于**正本，说明是云端重放而非脚本 bug（脚本只会产生 `__1` 后缀）。
  结论：`z:/quark1/` 上做批量改文件前必须提醒用户——本地处理完再同步、或先卸载 RaiDrive；
  纯预览不写盘是安全的；遇 `(N)` 副本先比 SHA-256 再删
- [感知哈希的并行收益与直觉相反](project_phash_parallel_payoff.md)
  — 哈希打包成整数用 `int.bit_count()` 比对，单对仅约 0.7 µs，
  于是比较阶段的瓶颈是**进程池启动**（约 1 s）而非计算：1200 张时并行只有 **0.40x**、
  3000 张才 **1.42x**，门槛因此定在 `PARALLEL_PAIR_MIN = 3000`（解码阶段是 8，那一侧并行有效）；
  并记下测这类代码踩的四个坑：`ex.map` 只传一个参数（且"失败退回串行"的容错会**静默吞掉**它）、
  Windows spawn 下测试脚本缺 `__main__` 守卫会被子进程重跑、A/B 必须用同一份数据、
  测前要先记可用内存（本机 2.7–6.1 GB 浮动会让并发度随运行而变）
- [Windows 亲和掩码探测的伪句柄坑](project_windows_affinity_probe.md)
  — `GetCurrentProcess()` 返回 `-1`，ctypes 默认按 `c_int` 截断成 `0xFFFFFFFF` 成为无效句柄，
  `GetProcessAffinityMask` 静默返回 0 → 悄悄降级到 `os.cpu_count()`；必须 `argtypes` 声明
  `c_void_p` 并传 `ctypes.c_void_p(-1)`。教训：Windows API 一律显式声明签名、
  `returncode == 0` 当失败、探测结果**带上来源字符串并打印**（本次正是靠输出里那句
  `os.cpu_count` 才发现坏了）
