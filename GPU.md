# Running this tree on the 5070 Ti box

The runbook for the second training machine: Windows/WSL2, **RTX 5070 Ti (Blackwell,
`sm_120`, 16 GB)**, 35 GB of host RAM. It is also the gaming PC. `WSL.md` is the runbook
for the 3070 box and nearly all of it holds here unchanged — the WSL-side checks, the
d3d12 render path, `usbipd`, pushing. This file carries only what is different, and the
numbers this card was measured to do.

## Check you are where you think you are

```bash
uname -r              # must contain "microsoft"
pwd -P                # must start with /home, NEVER /mnt/c
nvidia-smi            # must list the RTX 5070 Ti — otherwise you are on the 3070 box and WSL.md applies
```

## First time on this box

Same as `WSL.md` (`uv`, the clone, an SSH key made **here**), with one difference: the
distro's system python is 3.14 and the venv was built on 3.12 on purpose —

```bash
cd ~/smalldog/rl
uv sync --extra cuda --python 3.12     # uv fetches its own 3.12; the lock has wheels for it
uv run python checks/check_model.py    # 0 FAIL, exit 0 — and --terrain the same
```

`uv.lock` did not move: the locked jax 0.9.2 + CUDA 12.9 wheels cover `sm_120` as they
stand. No `.wslconfig` exists on this box, so WSL takes the default half of the host:
15 GB RAM and 4 GB swap. That holds training and the 100-robot `replay.py` (9.4 GB
virtual) — but the OOM story in `WSL.md` is a story about a distro with 7.7 GB, and the
first sign of swap here is the moment to write one.

## The card actually runs the graph

`CLAUDE.md` asked for this to be confirmed rather than assumed, because a CUDA/arch
mismatch is a silent CPU fallback, not an import error. Confirmed:

```
jax 0.9.2  backend gpu  [CudaDevice(id=0)]  compute_capability 12.0
jit 4096x4096 f32 matmul: 20 in 72 ms  (~38 TFLOP/s — a CPU cannot do that)
```

Three lines of stderr are this box's noise, not a problem, and none of them means the
fallback happened:

| line | why |
|---|---|
| `cuda_executor.cc: Could not get kernel mode driver version: Version does not match the format X.Y.Z` | the Windows driver reports itself as `616.64`, two components; XLA wanted three. Printed twice per process. |
| `xtile_compiler.cc: Fusion: gemm_fusion_dot_general ...` followed by an HLO dump | one Triton GEMM candidate the autotuner tried and could not build for `sm_120`; it picks another. 10–30 per compile. |
| `cuda_timer.cc: Delay kernel timed out: measured time has sub-optimal accuracy` | autotuner timing noise while the card is already busy. |

The progress line, not the stderr, is the signal — `jaxenv.py` says why.

## What it does, measured

| run | 3070 (8 GB) | 5070 Ti (16 GB) |
|---|---|---|
| `train_ppo.py --smoke`, cold cache, to the first progress line | 148 s | 96 s |
| `--smoke`, whole run, cold cache | 19.3 min | 8.5 min |

VRAM, with `XLA_PYTHON_CLIENT_PREALLOCATE=false` so `nvidia-smi` reports what the run
uses rather than what it reserved (the desktop's ~0.7 GB is included):

| model | peak `memory.used` |
|---|---|
| 2048 envs, 24 boxes, episode 500, 33 min at steady state | 9689 MiB — ~9.0 GB for the process |

So the box-terrain run — the one 8 GB never allowed — fits at the full 2048 with the
default `--mem-fraction 0.6` (10.7 GB reserved of 16.3). What is not free on it is time:
24 boxes against 19 robot collision geoms is a far heavier physics step than the flat
scene, and a 2048-env iteration is minutes, not seconds. Start it in the background,
redirect to a file, and do not read anything into silence before the first eval.

## It is the gaming PC

A game and PPO sharing the card is bad for both, so a run gets paused while a game is up:

```bash
./tools/pause_for_games.sh &                       # any Steam game (gameoverlayui.exe)
./tools/pause_for_games.sh Cyberpunk2077.exe &     # or by image name
```

It is a `SIGSTOP`/`SIGCONT` on the trainer, polled through `tasklist.exe`, and it is the
only kind of pause there is — brax cannot resume a run mid-flight, and a restart from a
checkpoint drops the optimizer state. Verified on a live CUDA process: stop, resume,
100 % GPU again, nothing recompiled. The script's own exit sends `CONT` first, so a killed
watcher does not leave a frozen run that looks like a compile.

The trainer's VRAM stays allocated while it is stopped. Whether Windows pages it out to
host RAM when the game asks is **unmeasured** — check Task Manager's dedicated GPU memory
with a run stopped and a game up before trusting it. If it does not page, the run's
`--mem-fraction` is the lever, and the table above says how low it can go.
