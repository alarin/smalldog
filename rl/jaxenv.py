#!/usr/bin/env python
"""
jaxenv.py — the environment variables this box needs, each set before the
library that reads it is imported.

    import jaxenv
    cache = jaxenv.configure(mem_fraction=0.60)   # BEFORE `import jax`
    import jax

    jaxenv.configure_gl()                         # BEFORE any GL context
    import mujoco

Two functions, one discipline, and it is the reason they share a file rather
than sitting at the top of three scripts: XLA reads the memory fraction once, at
the first device call, and Mesa reads the driver override once, when the first
GL context is created.  Set either one late and it is ignored in silence — the
run does not fail, it just runs the slow way.  The module name is older than its
second half.

They live here rather than in each script because train_ppo.py, eval.py and
replay.py all need them, and all three had the first one written out separately.
Everything below is `setdefault`, so a variable exported in the shell wins over
this file -- including `JAX_COMPILATION_CACHE_DIR=""`, which turns the cache off
without editing anything.

XLA_PYTHON_CLIENT_MEM_FRACTION
------------------------------
JAX preallocates a fraction of the card on the first device call.  The Windows
desktop on this box is already holding 0.5-1.5 GB, so the 0.75 default fails and
XLA falls back through a retry ladder, printing screens of RESOURCE_EXHAUSTED
before carrying on.  Not fatal, and not the training running out of memory --
but it buries the OOM that would matter.

The persistent compilation cache
--------------------------------
JAX ships it OFF: `jax_compilation_cache_dir` defaults to None, so every process
compiles the whole graph from scratch, and the brax PPO + MJX graph is big
enough that this is minutes before anything happens.  Measured 2026-08-31 on the
RTX 3070, `train_ppo.py --smoke`, cold cache then warm:

    time to the first progress line     148 s  ->  96 s
    cache entries written               571    ->  0
    XLA autotune blocks on stderr       32     ->  0
    whole run                           19.3 min -> 14.1 min

The first three lines are the cache doing its job and nothing else: a perfect
hit rate, and the autotuner never ran.  The last line is ONE sample against one
sample on a phase that is 85 % training, and the cold run's autotuning had
finished by its own first eval -- so whatever the remaining 4 min is, it is not
the cache, and it must not be quoted as a speedup.

Two runs with `--seed 0` also did not agree: final reward 86.53 against 84.15.
So this stack is not bit-reproducible across processes, cache or no cache, and
one smoke number is not evidence about a change.  Which of the two runs was the
cached one is not the interesting question at n=1; that it varies at all is.

Pointing a cache directory at the tree also persists the GPU autotuning, not
just the compiled executables -- jax's default for
`persistent_cache_enable_xla_caches` is
`"xla_gpu_per_fusion_autotune_cache_dir"`, and on a graph this size the
per-fusion autotune pass is a large share of the wait.

A note on reading a run that looks stuck, because it cost this session an hour.
A compiling process and a training process look ALIKE from outside: one CPU core
pegged, the worker threads asleep on a futex, flat VRAM, and `nvidia-smi`
reporting 90 %+ either way -- XLA autotunes ON the card, and brax dispatches
asynchronously, so neither the thread pattern nor GPU utilisation separates
them.  The only reliable signal is the progress line, which arrives at the first
eval.  Do not pipe this script through `tail`: the pipe buffers, the header and
that first line are invisible until the process exits, and a run that is fine
looks hung for as long as you care to watch it.  Redirect to a file and run
python with -u.

The cache is keyed on the HLO plus the jax and jaxlib versions, so a lock bump
invalidates it by construction and a stale entry cannot be served.  Deleting the
directory is always safe:

    rm -rf rl/.jax_cache

MUJOCO_GL, GALLIUM_DRIVER, MESA_LOADER_DRIVER_OVERRIDE
------------------------------------------------------
`MUJOCO_GL=egl` initialises here and renders correct frames -- on the CPU.  WSL
ships no NVIDIA EGL vendor (`/usr/share/glvnd/egl_vendor.d/` holds only
`50_mesa.json`), so Mesa answers and Mesa picks `llvmpipe`.  Measured on
`replay.py --robots 100` at 1280x960: 7740 ms/frame and 3845 MB peak RSS against
1802 ms/frame and 1377 MB through `d3d12`.

The memory is the half that matters.  llvmpipe spends ~2.5 GB of **host** RAM on
tile and per-thread buffers, and host RAM is what takes this distro down -- so
the driver override is not a performance nicety, it is the difference between a
12-minute render and a 52-minute one that may not finish.  WSL.md, "Rendering,
viewers", carries the table.

The override is applied only where the Gallium driver is actually on disk, so it
is inert on the mac and on the robot.  `gl_renderer()` asks the live context what
answered, which is the only way to know: the fast path and the slow one differ in
nothing else you can see from outside.

There is deliberately no size cap.  `jax_compilation_cache_max_size` turns on
LRU eviction, but jax raises RuntimeError from that path unless `filelock` is
installed, and it is not in this lock -- a cap set here would fail at the first
cache WRITE, which is the worst place to learn about it.  The directory is
unbounded and disposable instead; the one-line rm above is the eviction policy.
"""
from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, ".jax_cache")


def configure(mem_fraction: float = 0.60, cache_dir: str | None = None) -> str:
    """Set both, and return the cache directory actually in force.

    Must be called before anything imports jax: XLA reads the memory fraction
    once, at the first device call.
    """
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(mem_fraction))
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", cache_dir or CACHE_DIR)
    # jax's own default, restated because it is the load-bearing one: graphs
    # that compile in under a second are not worth a cache round-trip, and
    # caching them would bury the handful that cost minutes.
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "1.0")
    return os.environ["JAX_COMPILATION_CACHE_DIR"]


# Mesa's D3D12 Gallium driver.  Its presence on disk is the test for "this box
# can render on the card": /usr/lib/wsl/lib/libd3d12.so is WSL's passthrough and
# this .so is the Mesa driver that goes through it.  Absent, we are on a machine
# where forcing the override would break a GL stack that was already fine.
D3D12_DRI = "/usr/lib/x86_64-linux-gnu/dri/d3d12_dri.so"


def configure_gl(backend: str = "egl") -> str:
    """Pick a MuJoCo GL backend and point Mesa at the card.  Before any context.

    Returns the backend actually in force.  See the module docstring for why
    `MUJOCO_GL=egl` alone is the slow path here and why that costs host RAM.
    """
    os.environ.setdefault("MUJOCO_GL", backend)
    if os.environ["MUJOCO_GL"] == "egl" and os.path.exists(D3D12_DRI):
        os.environ.setdefault("GALLIUM_DRIVER", "d3d12")
        os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "d3d12")
    return os.environ["MUJOCO_GL"]


def gl_renderer() -> str:
    """What actually rendered, asked of the live context.

    Call AFTER the first Renderer exists: glGetString needs a current context,
    and before there is one the honest answer is not available.  Never infer it
    from the environment variables -- that they are set is not that they took.
    """
    try:
        from OpenGL import GL
        s = GL.glGetString(GL.GL_RENDERER)
        return s.decode() if isinstance(s, bytes) else str(s)
    except Exception as e:            # no context, no PyOpenGL — either way, unknown
        return f"unknown ({type(e).__name__}: {e})"


def cache_size(cache_dir: str | None = None) -> tuple[int, int]:
    """(entries, bytes) on disk.  Cheap enough to call before and after a run."""
    d = cache_dir or os.environ.get("JAX_COMPILATION_CACHE_DIR") or CACHE_DIR
    n = total = 0
    for root, _, files in os.walk(d):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
                n += 1
            except OSError:               # evicted or half-written; not our problem
                pass
    return n, total


def cache_line(cache_dir: str | None = None, before: tuple[int, int] | None = None) -> str:
    """One line for a run's header or footer, with the delta if given."""
    d = cache_dir or os.environ.get("JAX_COMPILATION_CACHE_DIR") or CACHE_DIR
    if not d:
        return "compilation cache OFF (JAX_COMPILATION_CACHE_DIR is empty)"
    n, b = cache_size(d)
    s = f"{d} — {n} entries, {b/1e6:.0f} MB"
    if before is not None:
        dn, db = n - before[0], b - before[1]
        s += f" ({dn:+d} entries, {db/1e6:+.0f} MB this run)"
    return s


if __name__ == "__main__":
    print(cache_line(configure()))
    print(f"MUJOCO_GL={configure_gl()} "
          f"GALLIUM_DRIVER={os.environ.get('GALLIUM_DRIVER', '(unset)')}")
