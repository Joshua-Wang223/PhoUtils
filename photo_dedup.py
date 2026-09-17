#!/usr/bin/env python3
"""照片去重：字节级重复 + 感知哈希近重复（含旋转匹配、EXIF 时间保护、多进程加速）。

依赖：Pillow, numpy（无其他第三方库）。

典型用法：
    python photo_dedup.py ./photos                  # 预览（只报告，不动文件）
    python photo_dedup.py ./photos --apply          # 执行：移到 _duplicates/
    python photo_dedup.py ./photos --apply --delete # 执行：直接删除
    python photo_dedup.py ./photos --apply --sheet  # 顺带生成每组对比图

判定分两阶段：
  1) 字节完全相同（SHA-256，先用文件大小预筛）—— 必然是重复。
  2) 感知哈希相近（pHash 256bit + dHash + aHash，可选 0/90/180/270 旋转）
     并用并查集聚成"连拍簇"——同一机位同姿势的多帧，属于内容冗余，
     默认每组只保留最锐利的一张（拉普拉斯方差，统一缩放到长边 1024 比较）。

两道保险：
  · EXIF 时间保护（--max-time-gap，默认 15 秒）：两人都有拍摄时间、且相隔超过
    阈值时，不做哈希比较，避免把"同一构图但不同时刻"的照片并进一组。
  · 并行加速：自动探测逻辑核数与可用内存，按 min(核数, 内存预算) 开启多进程；
    --workers 0 为自动，1 为强制串行。哈希指纹打包成整数，比对用 int.bit_count()。
"""

import argparse
import ctypes
import datetime
import hashlib
import math
import os
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

try:
    import numpy as np
    from PIL import ExifTags, Image, ImageDraw
except ImportError:
    sys.exit("需要 Pillow 和 numpy：pip install pillow numpy")

Image.MAX_IMAGE_PIXELS = None

DEFAULT_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp",
                ".heic", ".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2")

# 文件名里出现这些，说明多半是"副本"，优先保留另一边
COPY_PAT = re.compile(r"(\(\d+\)|（\d+）|\bcopy\b|副本|复件|拷贝|-coyp)", re.I)

# 并发策略常量（与 VidUtils/vidcrop_* 的取舍一致）
MAX_USABLE_MEM_RATIO = 0.85   # 最多用掉总内存的这个比例
MIN_SYSTEM_RESERVE_GB = 1.5   # 无论如何给系统留出的余量
DEFAULT_MEM_PER_WORKER = 0.25  # 单个进程预估占用 GB（解码一张大图 + 哈希）
PARALLEL_MIN_ITEMS = 8        # 解码任务数少于此时串行更快（每张 ~0.4s，值得开进程池）

# 两两比较的并行门槛。实测（1200 张 / 72 万对，4 进程）：串行 0.50s，并行 1.20s
# —— 哈希已打包成整数，单对只花约 0.7µs，进程池 1s 左右的启动开销要几千张才摊得平。
# 因此这里给一个很高的门槛：常见规模走串行，超大图库（3000 张以上）才启用并行。
PARALLEL_PAIR_MIN = 3000


# ═══════════════════════════════════════════════════════════════════
#  系统能力探测
# ═══════════════════════════════════════════════════════════════════

def detect_cpu():
    """逻辑核数。依次尝试 cgroup v2 / v1 / sched_getaffinity / Windows 亲和掩码 / cpu_count。"""
    try:
        parts = open("/sys/fs/cgroup/cpu.max", encoding="utf-8").read().split()
        if len(parts) == 2 and parts[0] != "max":
            quota, period = int(parts[0]), int(parts[1])
            if quota > 0 and period > 0:
                return max(1, round(quota / period)), f"cgroup v2 ({quota}/{period})"
    except Exception:
        pass
    try:
        quota = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", encoding="utf-8").read())
        period = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us", encoding="utf-8").read())
        if quota > 0 and period > 0:
            return max(1, round(quota / period)), f"cgroup v1 ({quota}/{period})"
    except Exception:
        pass
    try:
        n = len(os.sched_getaffinity(0))
        if n > 0:
            return n, "sched_getaffinity"
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            # 伪句柄必须按指针传：GetCurrentProcess 返回 -1，按 int 传会被截断成
            # 0xFFFFFFFF 导致调用失败（拿不到真实的亲和掩码）。
            k32 = ctypes.windll.kernel32
            k32.GetProcessAffinityMask.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                ctypes.POINTER(ctypes.c_size_t)]
            mask, sysmask = ctypes.c_size_t(), ctypes.c_size_t()
            if k32.GetProcessAffinityMask(ctypes.c_void_p(-1),
                                          ctypes.byref(mask), ctypes.byref(sysmask)):
                n = bin(mask.value).count("1")
                if n > 0:
                    return n, "Windows 亲和掩码"
        except Exception:
            pass
    return max(1, os.cpu_count() or 1), "os.cpu_count"


def detect_memory_gb():
    """返回 (总内存 GB, 可用内存 GB, 来源)；探测不到时返回 (0, 0, 原因)。"""
    try:
        v = open("/sys/fs/cgroup/memory.max", encoding="utf-8").read().strip()
        if v != "max":
            limit = int(v) / 1024 ** 3
            cur = 0.0
            try:
                cur = int(open("/sys/fs/cgroup/memory.current", encoding="utf-8").read()) / 1024 ** 3
            except Exception:
                pass
            return limit, max(0.0, limit - cur), "cgroup v2"
    except Exception:
        pass
    try:
        total = avail = 0.0
        for line in open("/proc/meminfo", encoding="utf-8"):
            if line.startswith("MemTotal:"):
                total = int(line.split()[1]) / 1024 ** 2
            elif line.startswith("MemAvailable:"):
                avail = int(line.split()[1]) / 1024 ** 2
            if total and avail:
                break
        if total:
            return total, avail, "/proc/meminfo"
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(st)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return (st.ullTotalPhys / 1024 ** 3, st.ullAvailPhys / 1024 ** 3,
                        "GlobalMemoryStatusEx")
        except Exception:
            pass
    return 0.0, 0.0, "未探测到"


def estimate_mem_per_worker(files, override):
    """单进程内存预估。未指定时按最大文件推算解码峰值（JPEG 解码约为文件的 40 倍）。"""
    if override > 0:
        return override, "--mem-per-worker 指定"
    try:
        biggest = max(os.path.getsize(f) for f in files)
    except (ValueError, OSError):
        biggest = 0
    est = max(DEFAULT_MEM_PER_WORKER, min(2.0, biggest * 40 / 1024 ** 3))
    return est, f"按最大文件 {biggest / 1024 ** 2:.1f} MB 推算"


def resolve_workers(n_tasks, mem_per_worker, override):
    """按 min(核数上限, 内存上限, 待处理数) 决定进程数，返回 (进程数, 说明, 探测信息)。"""
    cpu, cpu_src = detect_cpu()
    total_gb, avail_gb, mem_src = detect_memory_gb()

    if n_tasks <= 0:
        return 0, "", (cpu, cpu_src, total_gb, avail_gb, mem_src)
    if override > 0:
        return max(1, min(override, n_tasks)), f"--workers {override}", \
            (cpu, cpu_src, total_gb, avail_gb, mem_src)

    by_cpu = max(1, cpu)
    if total_gb > 0:
        usable = max(0.1, min(total_gb * MAX_USABLE_MEM_RATIO,
                              max(0.0, avail_gb - MIN_SYSTEM_RESERVE_GB)))
        by_mem = max(1, int(usable / max(0.05, mem_per_worker)))
        limit_txt = f"内存上限 {by_mem}"
    else:
        by_mem = by_cpu
        limit_txt = "内存未知，按核数"

    workers = max(1, min(by_cpu, by_mem, n_tasks))
    if n_tasks < PARALLEL_MIN_ITEMS and override == 0:
        workers = 1
        why = f"任务仅 {n_tasks} 个，串行更快"
    else:
        why = f"核数上限 {by_cpu} 与 {limit_txt}，待处理 {n_tasks} 取小"
    return workers, why, (cpu, cpu_src, total_gb, avail_gb, mem_src)


def profile_text(cpu, cpu_src, total_gb, avail_gb, mem_src):
    mem = f"内存 {total_gb:.1f} GB（可用 {avail_gb:.1f} GB，{mem_src}）" if total_gb > 0 \
        else f"内存未知（{mem_src}）"
    return f"逻辑核 {cpu}（{cpu_src}）｜{mem}"


# ═══════════════════════════════════════════════════════════════════
#  基础工具
# ═══════════════════════════════════════════════════════════════════

def iter_images(root, exts, recursive, skip_dirs=("_duplicates",)):
    """遍历图片。skip_dirs 里的目录名（含 --dup-dir 自定义名）递归时一律跳过。"""
    seen = set()
    walk = os.walk(root) if recursive else [(root, [], sorted(os.listdir(root)))]
    for dirpath, dirnames, filenames in walk:
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        if recursive:
            filenames = sorted(filenames)
        for name in filenames:
            if name.startswith("._") or not name.lower().endswith(exts):
                continue
            p = os.path.join(dirpath, name)
            if os.path.isfile(p) and p not in seen:
                seen.add(p)
                yield p


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def canon_rank(path):
    """越小越像"原始文件"：带 (2)/副本 后缀的排后，其次按文件名。"""
    name = os.path.basename(path)
    return (1 if COPY_PAT.search(name) else 0, name)


def fmt_time(sec):
    return f"{sec * 1000:.0f} ms" if sec < 1 else f"{sec:.1f} s"


# ═══════════════════════════════════════════════════════════════════
#  EXIF 拍摄时间
# ═══════════════════════════════════════════════════════════════════

_EXIF_TAG = {v: k for k, v in ExifTags.TAGS.items()}


def _parse_dt(text):
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.datetime.strptime(text.strip(), fmt)
        except (ValueError, AttributeError):
            continue
    return None


def capture_time(img):
    """从已打开的图片读取拍摄时间，返回 (epoch 秒 | None, 来源说明)。

    优先 DateTimeOriginal（含子秒与时区偏移），退回 DateTime。
    相机时钟按本地时间处理；有 OffsetTimeOriginal 时归一到 UTC 再比较。
    """
    try:
        exif = img.getexif()
    except Exception as e:
        return None, f"EXIF 读取失败({e})"
    if not exif:
        return None, "无 EXIF"

    sub = exif.get_ifd(0x8769)  # ExifIFD
    dt = _parse_dt(sub.get(_EXIF_TAG["DateTimeOriginal"], "")) or \
        _parse_dt(exif.get(_EXIF_TAG["DateTime"], ""))
    if dt is None:
        return None, "无拍摄时间字段"

    sec = dt.timestamp()
    frac = str(sub.get(_EXIF_TAG.get("SubsecTimeOriginal", -1), "") or "").strip()
    if frac.isdigit():
        sec += int(frac) / 10 ** len(frac)

    off = str(sub.get(_EXIF_TAG.get("OffsetTimeOriginal", -1), "") or "").strip()
    m = re.fullmatch(r"([+-])(\d{2}):?(\d{2})", off)
    if m:
        sign = 1 if m.group(1) == "+" else -1
        sec -= sign * (int(m.group(2)) * 3600 + int(m.group(3)) * 60)
    return sec, "DateTimeOriginal"


# ═══════════════════════════════════════════════════════════════════
#  感知哈希
# ═══════════════════════════════════════════════════════════════════

_DCT_BASIS = {}


def _dct_basis(n):
    if n not in _DCT_BASIS:
        k = np.arange(n, dtype=np.float64)
        b = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n))
        b *= np.sqrt(2.0 / n)
        b[0] *= 1 / np.sqrt(2)
        _DCT_BASIS[n] = b
    return _DCT_BASIS[n]


def _pack(bits):
    """bits → 整数，之后用 (a ^ b).bit_count() 求汉明距离。"""
    return int.from_bytes(np.packbits(np.asarray(bits, dtype=np.uint8)).tobytes(), "big")


def hashes(img):
    """返回打包后的 (pHash 256bit, dHash 64bit, aHash 64bit)。"""
    g = img.convert("L")

    small = np.asarray(g.resize((32, 32), Image.LANCZOS), dtype=np.float32)
    d = _dct_basis(32) @ small @ _dct_basis(32).T
    low = d[:16, :16].flatten()[1:]
    ph = (low > np.median(low)).astype(np.uint8)

    dh_img = np.asarray(g.resize((9, 8), Image.LANCZOS), dtype=np.float32)
    dh = (dh_img[:, 1:] > dh_img[:, :-1]).astype(np.uint8)

    ah_img = np.asarray(g.resize((8, 8), Image.LANCZOS), dtype=np.float32)
    ah = (ah_img > ah_img.mean()).astype(np.uint8)

    return _pack(ph), _pack(dh.flatten()), _pack(ah.flatten())


def laplacian_var(img, long_side=1024):
    """清晰度指标：拉普拉斯方差，统一缩放后比较才公平。"""
    g = img.convert("L")
    w, h = g.size
    if max(w, h) > long_side:
        s = long_side / max(w, h)
        g = g.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    a = np.asarray(g, dtype=np.float32)
    if min(a.shape) < 3:
        return 0.0
    lap = (a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]
           - 4 * a[1:-1, 1:-1])
    return float(lap.var())


def fingerprint(path, rotations=True):
    """返回 (指纹, 清晰度, 拍摄时间 | None, 时间来源)。指纹为 {角度: (p, d, a)} 整数元组。"""
    img = Image.open(path)
    img.load()
    variants = {0: hashes(img)}
    if rotations:
        for ang in (90, 180, 270):
            variants[ang] = hashes(img.rotate(ang, expand=True))
    sharp = laplacian_var(img)
    ts, src = capture_time(img)
    img.close()
    return variants, sharp, ts, src


def fingerprint_worker(job):
    """进程池入口：绝不让异常冒泡打断整个池。"""
    path, rotations = job
    try:
        fp, sharp, ts, src = fingerprint(path, rotations)
        return path, fp, sharp, ts, src, None
    except Exception as e:
        return path, None, 0.0, None, "", f"{type(e).__name__}: {e}"


# ═══════════════════════════════════════════════════════════════════
#  分组（两两比较，可多进程）
# ═══════════════════════════════════════════════════════════════════

class UnionFind:
    def __init__(self, items):
        self.p = {x: x for x in items}

    def find(self, x):
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb

    def groups(self):
        out = defaultdict(list)
        for x in self.p:
            out[self.find(x)].append(x)
        return [sorted(v) for v in out.values() if len(v) > 1]


_PAIR_CTX = {}


def pair_scan(row_range):
    """扫描行区间 [lo, hi) 的右三角：返回 (匹配索引对, 被时间保护跳过的对数)。

    入参是元组，这样单进程与进程池 ex.map 可以用同一个入口。
    """
    lo, hi = row_range
    fps = _PAIR_CTX["fps"]
    times = _PAIR_CTX["times"]
    thr_p, thr_d = _PAIR_CTX["thr_p"], _PAIR_CTX["thr_d"]
    gap = _PAIR_CTX["gap"]
    n = len(fps)

    hits = []
    skipped = 0
    for i in range(lo, hi):
        fa, ta = fps[i], times[i]
        for j in range(i + 1, n):
            tb = times[j]
            # 时间保护：双方都有拍摄时间且相隔超阈值 → 同一构图也不并簇
            if gap > 0 and ta is not None and tb is not None and abs(ta - tb) > gap:
                skipped += 1
                continue
            fb = fps[j]
            for ha in fa.values():
                for hb in fb.values():
                    if (ha[0] ^ hb[0]).bit_count() > thr_p:
                        continue
                    if (ha[1] ^ hb[1]).bit_count() > thr_d:
                        continue
                    if (ha[2] ^ hb[2]).bit_count() > thr_d:
                        continue
                    hits.append((i, j))
                    break
                else:
                    continue
                break
    return hits, skipped


def _pair_init(fps, times, thr_p, thr_d, gap):
    _PAIR_CTX.update(fps=fps, times=times, thr_p=thr_p, thr_d=thr_d, gap=gap)


def compare_all(fps, times, thr_p, thr_d, gap, workers, log):
    """两两比较，返回 (匹配对列表, 被时间保护跳过的对数, 用时, 是否并行)。"""
    n = len(fps)
    pairs = n * (n - 1) // 2
    t0 = time.perf_counter()

    if workers <= 1 or n < PARALLEL_PAIR_MIN:
        _pair_init(fps, times, thr_p, thr_d, gap)
        hits, skipped = pair_scan((0, n))
        log(f"    比较 {pairs} 对，单进程，用时 {fmt_time(time.perf_counter() - t0)}")
        return hits, skipped, time.perf_counter() - t0, False

    # 按行切块，块数取 4×进程数以便负载均衡
    chunk = max(1, math.ceil(n / (workers * 4)))
    ranges = [(s, min(s + chunk, n)) for s in range(0, n, chunk)]
    hits, skipped = [], 0
    try:
        with ProcessPoolExecutor(max_workers=workers, initializer=_pair_init,
                                 initargs=(fps, times, thr_p, thr_d, gap)) as ex:
            for h, s in ex.map(pair_scan, ranges):
                hits.extend(h)
                skipped += s
        log(f"    比较 {pairs} 对，{workers} 进程，用时 {fmt_time(time.perf_counter() - t0)}")
    except Exception as e:
        log(f"    !! 多进程比较失败（{e}），退回单进程")
        _pair_init(fps, times, thr_p, thr_d, gap)
        hits, skipped = pair_scan((0, n))
        return hits, skipped, time.perf_counter() - t0, False
    return hits, skipped, time.perf_counter() - t0, True


def pick_keeper(files, strategy, sharp):
    if strategy == "first":
        return sorted(files, key=canon_rank)[0]
    return max(sorted(files), key=lambda f: (sharp.get(f, 0.0),))


# ═══════════════════════════════════════════════════════════════════
#  执行动作
# ═══════════════════════════════════════════════════════════════════

def apply_action(files, dup_dir, action, dry_run, log):
    """把 files 移走或删掉，返回 (释放字节数, 实际处理的文件)。"""
    freed = 0
    done = []
    for f in files:
        try:
            size = os.path.getsize(f)
        except OSError:
            continue
        if dry_run:
            freed += size
            done.append(f)
            continue
        try:
            if action == "delete":
                os.remove(f)
            else:
                target = os.path.join(dup_dir, os.path.basename(f))
                stem, ext = os.path.splitext(os.path.basename(f))
                n = 1
                while os.path.exists(target):
                    target = os.path.join(dup_dir, f"{stem}__{n}{ext}")
                    n += 1
                shutil.move(f, target)
            freed += size
            done.append(f)
        except OSError as e:
            log(f"  !! 处理失败 {f}: {e}")
    return freed, done


def make_sheets(groups, sheet_prefix, log):
    """每组生成一张横向对比图，方便人工复核。"""
    ok = 0
    for i, g in enumerate(groups, 1):
        thumbs = []
        for f in g[:12]:
            try:
                im = Image.open(f)
                im.load()
                im.thumbnail((320, 460), Image.LANCZOS)
                thumbs.append((os.path.basename(f), im))
            except Exception as e:
                log(f"  !! 无法读取 {f}: {e}")
        if not thumbs:
            continue
        pad, label = 8, 22
        W = pad + sum(t[1].width + pad for t in thumbs)
        H = pad + label + max(t[1].height for t in thumbs) + pad
        sheet = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(sheet)
        x = pad
        for name, im in thumbs:
            sheet.paste(im, (x, pad + label))
            d.text((x, pad + 4), name, fill="black")
            x += im.width + pad
        sheet.save(f"{sheet_prefix}_group{i:02d}.jpg", quality=88)
        ok += 1
    return ok


# ═══════════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════════

def build_args():
    ap = argparse.ArgumentParser(
        description="照片去重：字节级重复 + 感知哈希近重复（连拍簇）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="照片目录")
    ap.add_argument("--apply", action="store_true",
                    help="真正执行（默认只预览，不动任何文件）")
    ap.add_argument("--delete", action="store_true",
                    help="直接删除而不是移动到隔离目录，需配合 --apply")
    ap.add_argument("--dup-dir", default="_duplicates",
                    help="隔离目录名，相对照片目录（默认 _duplicates）")
    ap.add_argument("-r", "--recursive", action="store_true", help="递归子目录")
    ap.add_argument("--no-rotation", action="store_true",
                    help="不做 0/90/180/270 旋转匹配")
    ap.add_argument("--phash-dist", type=int, default=20,
                    help="pHash(256bit) 汉明距离阈值，越小越严（默认 20）")
    ap.add_argument("--dhash-dist", type=int, default=10,
                    help="dHash/aHash(64bit) 汉明距离阈值（默认 10）")
    ap.add_argument("--max-time-gap", type=float, default=15.0,
                    help="EXIF 拍摄时间相隔超过该秒数就不并簇，0=关闭（默认 15）")
    ap.add_argument("--keep", choices=["sharpest", "first"], default="sharpest",
                    help="连拍簇保留哪张：sharpest=最锐利（默认），first=按文件名第一张")
    ap.add_argument("--no-burst", action="store_true",
                    help="只做字节级去重，跳过感知哈希/连拍检测")
    ap.add_argument("--sheet", action="store_true", help="生成每组对比图")
    ap.add_argument("--exclude", action="append", default=[],
                    help="排除文件名含该子串的照片，可重复指定")
    ap.add_argument("--workers", type=int, default=0,
                    help="并行进程数，0=按核数与内存自动（默认）")
    ap.add_argument("--mem-per-worker", type=float, default=0.0,
                    help="单进程内存预估 GB，0=按最大文件推算（默认）")
    return ap.parse_args()


def main():
    args = build_args()

    root = os.path.abspath(args.folder)
    if not os.path.isdir(root):
        sys.exit(f"目录不存在：{root}")
    if args.delete and not args.apply:
        sys.exit("--delete 必须配合 --apply（先预览，确认后再执行）")
    if args.workers < 0:
        sys.exit("--workers 不能为负数")

    dup_dir = os.path.join(root, args.dup_dir)
    dry_run = not args.apply

    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    exts = tuple(DEFAULT_EXTS)
    skip_dirs = (os.path.basename(args.dup_dir.rstrip("/\\")) or "_duplicates",)
    files = [f for f in iter_images(root, exts, args.recursive, skip_dirs)
             if not any(s.lower() in os.path.basename(f).lower() for s in args.exclude)]

    log(f"目录：{root}")
    log(f"扫描到 {len(files)} 张图片" + ("（已按 --exclude 过滤）" if args.exclude else ""))
    if not files:
        return
    log(f"动作：{'预览（不修改文件）' if dry_run else ('删除' if args.delete else f'移动到 {args.dup_dir}/')}")

    to_remove = []
    total_freed = 0
    t_start = time.perf_counter()

    # ---- 阶段 1：字节级重复
    by_size = defaultdict(list)
    for f in files:
        try:
            by_size[os.path.getsize(f)].append(f)
        except OSError:
            pass

    exact_groups = []
    for size, same_size in sorted(by_size.items()):
        if len(same_size) < 2:
            continue
        by_hash = defaultdict(list)
        for f in same_size:
            by_hash[sha256(f)].append(f)
        for h, group in by_hash.items():
            if len(group) > 1:
                exact_groups.append(sorted(group))

    log()
    log(f"[1/2] 字节级重复：{len(exact_groups)} 组")
    for group in exact_groups:
        # 字节完全相同，清晰度也一样，保留文件名最"正"的那份即可
        keep = sorted(group, key=canon_rank)[0]
        drop = [f for f in group if f != keep]
        size = os.path.getsize(keep)
        log(f"  保留 {os.path.relpath(keep, root)}（{size / 1024:.0f} KB，共 {len(group)} 份完全相同）")
        for f in drop:
            log(f"    冗余 {os.path.relpath(f, root)}")
        to_remove.extend(drop)
        total_freed += size * len(drop)
    if not exact_groups:
        log("  （无）")

    # ---- 阶段 2：感知哈希 / 连拍簇
    burst_groups = []
    sharp = {}
    if not args.no_burst:
        drop_set = set(to_remove)
        todo = [f for f in files if f not in drop_set]
        mem_per_worker, mem_why = estimate_mem_per_worker(todo, args.mem_per_worker)
        workers, why, prof = resolve_workers(len(todo), mem_per_worker, args.workers)
        cpu, cpu_src, total_gb, avail_gb, mem_src = prof

        log()
        log("[2/2] 感知哈希近重复（连拍簇）")
        log(f"  系统：{profile_text(cpu, cpu_src, total_gb, avail_gb, mem_src)}")
        log(f"  并行：{workers} 进程（{why}；单进程预估 {mem_per_worker:.2f} GB，{mem_why}）")

        t0 = time.perf_counter()
        records = []          # (path, 指纹, 拍摄时间)
        failed = []
        no_time = Counter()
        done = 0

        def consume(result):
            nonlocal done
            path, fp, sh, ts, src, err = result
            done += 1
            if err:
                failed.append((path, err))
                return
            records.append((path, fp, ts))
            sharp[path] = sh
            if ts is None:
                no_time[src or "未知"] += 1

        jobs = [(f, not args.no_rotation) for f in todo]
        if workers > 1 and len(jobs) >= PARALLEL_MIN_ITEMS:
            try:
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    for result in ex.map(fingerprint_worker, jobs):
                        consume(result)
                        if done % 50 == 0:
                            print(f"  ... 已解析 {done}/{len(jobs)}", file=sys.stderr)
            except Exception as e:
                log(f"  !! 多进程解析失败（{e}），退回单进程重跑")
                records, failed, no_time, done = [], [], Counter(), 0
                for job in jobs:
                    consume(fingerprint_worker(job))
        else:
            for job in jobs:
                consume(fingerprint_worker(job))

        for path, err in failed:
            log(f"  !! 无法解析 {os.path.relpath(path, root)}: {err}")
        log(f"  解析 {len(records)} 张，用时 {fmt_time(time.perf_counter() - t0)}"
            + (f"，{workers} 进程" if workers > 1 else "，单进程"))

        records.sort(key=lambda r: r[0])
        names = [r[0] for r in records]
        fps = [r[1] for r in records]
        times = [r[2] for r in records]

        if no_time:
            detail = "，".join(f"{k}×{v}" for k, v in no_time.most_common(3))
            log(f"  提示：{sum(no_time.values())} 张无拍摄时间（{detail}），不参与时间保护")

        hits, skipped, _, _ = compare_all(
            fps, times, args.phash_dist, args.dhash_dist, args.max_time_gap, workers, log)
        if args.max_time_gap > 0:
            log(f"  时间保护：--max-time-gap {args.max_time_gap:g}s，跳过 {skipped} 对超时的比较")

        uf = UnionFind(names)
        for i, j in hits:
            uf.union(names[i], names[j])
        burst_groups = uf.groups()

        log(f"  结果：{len(burst_groups)} 组，共 {sum(len(g) for g in burst_groups)} 张")
        if not burst_groups:
            log("  （无）")
        for g in sorted(burst_groups, key=lambda x: (-len(x), x[0])):
            keep = pick_keeper(g, args.keep, sharp)
            log(f"  {len(g)} 帧，保留 {os.path.relpath(keep, root)}"
                + (f"（清晰度 {sharp.get(keep, 0):.0f}）" if args.keep == "sharpest" else ""))
            for f in g:
                if f == keep:
                    continue
                log(f"    冗余 {os.path.relpath(f, root)}  清晰度 {sharp.get(f, 0):.0f}")
                to_remove.append(f)
                total_freed += os.path.getsize(f)

    if not to_remove:
        log()
        log(f"没有发现需要处理的重复照片。（总用时 {fmt_time(time.perf_counter() - t_start)}）")
        return

    # ---- 执行
    log()
    log(f"合计 {len(to_remove)} 张，约 {total_freed / 1024 / 1024:.1f} MB")
    if dry_run:
        log(f"总用时 {fmt_time(time.perf_counter() - t_start)}")
        log("这是预览。确认无误后加 --apply 执行：")
        log(f"  python {os.path.basename(__file__)} \"{root}\" --apply")
        return

    if args.sheet:
        os.makedirs(dup_dir, exist_ok=True)
        n = make_sheets(exact_groups + burst_groups, os.path.join(dup_dir, "dup"), log)
        log(f"已生成 {n} 张对比图到 {args.dup_dir}/")

    if not args.delete:
        os.makedirs(dup_dir, exist_ok=True)
    freed, done = apply_action(to_remove, dup_dir, "delete" if args.delete else "move",
                               False, log)

    report = os.path.join(root, "photo_dedup_report.txt")
    with open(report, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    left = len(list(iter_images(root, exts, args.recursive, skip_dirs)))
    log()
    log(f"完成：处理 {len(done)} 张，释放 {freed / 1024 / 1024:.1f} MB，"
        f"目录内剩余 {left} 张，总用时 {fmt_time(time.perf_counter() - t_start)}")
    if not args.delete:
        log(f"冗余文件在 {args.dup_dir}/，确认后可还原：mv \"{args.dup_dir}\"/* .  或直接删除该目录")
    log(f"明细已写入 {os.path.relpath(report, root)}")


if __name__ == "__main__":
    main()
