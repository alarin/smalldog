# Running this tree in WSL2

Three machines share this repository and nothing else — no shared filesystem, no scp of
a checkpoint (`rl/CLAUDE.md`, "The machine split"):

| machine | what it is for |
|---|---|
| a mac | the CAD — `3d/`, and everything generated out of it |
| **this box**: Windows + WSL2, RTX 3070 | training — `rl/`, plus the pure-Python MuJoCo sim |
| an Orange Pi 5 Pro | the robot — `robot/runtime`, step 7 |

This file is the runbook for a Claude Code session on the middle one. It overrides
nothing: `3d/CLAUDE.md` and `rl/CLAUDE.md` still say what is source and what is output.
It only says which of their steps this machine can actually run, and what it costs to
find out the hard way.

## Check you are where you think you are

Claude Code must be the **Linux** install, running inside the WSL2 distro — not Windows
Claude Code pointed at `\\wsl$\...`. On the wrong side of the boundary the venvs are
unexecutable, `nvidia-smi` is a different program, and every path in every script is
wrong in a way that reads like a broken checkout.

```bash
uname -r              # must contain "microsoft"
pwd -P                # must start with /home, NEVER /mnt/c
nvidia-smi            # must list the RTX 3070
```

`/mnt/c` is the one that will not announce itself: it works, an order of magnitude
slower, and mesh loads and checkpoint writes are exactly the traffic that feels it. Keep
the checkout at `~/smalldog`.

## First time on this box

```bash
sudo apt update && sudo apt install -y git build-essential curl
curl -LsSf https://astral.sh/uv/install.sh | sh     # both venvs in this tree are uv venvs
git clone git@github.com:alarin/smalldog.git ~/smalldog
cd ~/smalldog && git submodule update --init       # ros2/src/mujoco_ros2_control
```

`origin` is SSH, so pushing needs a key **on this box** registered with GitHub
(`ssh-keygen -t ed25519` here; do not copy the mac's private key across). Set
`user.name` / `user.email` to the identity already in `git log`, or history splits in two.

Do not set `core.autocrlf`. `.gitattributes` forces LF for all three machines and says
why: a CRLF checkout kills `ros2/tools/*.sh` with `bash: $'\r': command not found`,
which reads like a broken install rather than a line-ending problem.

The NVIDIA driver belongs on the **Windows** side. Installing a Linux driver inside WSL
overwrites the passthrough and `nvidia-smi` stops seeing the card.

## What runs here, and what does not

| tree | on this box |
|---|---|
| `rl/` | **yes — this is what the box is for.** `uv sync --extra cuda`. |
| `ros2/tools/standalone_sim.py` | yes. It is pure Python — `mujoco`, `numpy` and the pure-Python `TrotGait`, no ROS at all — so `rl/`'s venv runs it. This is how step 6's regressions get exercised here. |
| `ros2/` proper — pixi, colcon, `tools/sim.sh` | **no, as committed.** `ros2/pixi/pixi.toml` declares `platforms = ["osx-arm64"]` and pins `clang_osx-arm64` by name. Bringing up linux-64 is a re-solve, a second lock and a fresh set of pins for a different toolchain — a piece of work, not a flag. Do not start it in passing. |
| `3d/` | possible, not the convention. CadQuery, gmsh and sfepy all have linux-64 wheels, but geometry is the mac's job; a build here that skips the mac's baselines is worse than no build. If you do change geometry here, the whole ladder in `3d/CLAUDE.md`, "Verifying a change", applies unchanged — including step 6, and including a *before* FEA run to compare against. |
| `robot/` | selftests yes, hardware only with USB passthrough (below). |

## rl/ — the venv and the card

```bash
cd ~/smalldog/rl
uv sync --extra cuda
uv run python -c "import jax; print(jax.devices())"   # must list a CudaDevice
uv run python checks/check_model.py                   # exit 0, or do not train
```

`rl/uv.lock` is **not in the tree yet** — the first `uv sync` writes it. Commit it, on its
own: a pinned resolution is the whole point when three machines share one tree
(`rl/CLAUDE.md`), and the machine that first resolves it decides for the other two.

8 GB of VRAM is the binding constraint, not the FLOPs — `XLA_PYTHON_CLIENT_MEM_FRACTION`,
`num_envs` 1024–2048, and Windows itself spending 0.5–1.5 GB on the desktop. The numbers
and the reasoning are in `rl/README.md`, "Environments"; they are not repeated here.

Training is a long job: start it in the background and let the session keep working, do
not sit inside a foreground run for an hour.

## Host RAM is the other binding constraint, and it takes the distro with it

The VRAM paragraph above is true and incomplete. 8 GB of VRAM bounds what trains; the
box's **16 GB of host RAM** bounds what can be built at all, and host RAM is the one
that has actually killed this machine.

WSL2 with no `.wslconfig` takes half the host — 7.7 GB — with 2 GB of swap. On
2026-08-31 `replay.py --robots 100` reached 9.4 GB of virtual memory inside that and
the OOM killer fired twice. The second time it took WSL down with it: an OOM in WSL2
lands in `init.scope`, systemd does not survive it, and the distro, the run and the
Claude Code session end together. There is no traceback afterwards. The evidence is
`/var/log/kern.log`, and only if you read it before the next boot rotates it:

```bash
grep -iE "oom|killed process" /var/log/kern.log
```

`C:\Users\tony\.wslconfig` now says:

```ini
[wsl2]
memory=11GB
swap=16GB
processors=12
```

It lives on the Windows side and needs `wsl --shutdown` before it takes effect. The
large swap is the load-bearing half: it makes nothing faster, it turns a kill into
thrashing, and thrashing can be interrupted where an OOM cannot.

Two habits follow.

**Check for orphans before a heavy run** — `pgrep -a python`. The second crash was not
one run exhausting the box. A python from the first crash was still alive holding
1.7 GB in swap, and the two together were what did it.

**Cap a measurement rather than trusting it.**

```bash
systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=512M <cmd>
```

bounds a suspect run to a cgroup, so a blowup kills the process and leaves the distro
standing. `ulimit -v` / `RLIMIT_AS` is cheaper and works for pure-MuJoCo and NumPy work,
but **it breaks EGL and CUDA**: both map large regions up front, and a capped address
space makes `mujoco.Renderer` fail with an `EGLError` that reads like a missing GL
backend rather than like a limit you set on yourself.

### Where the memory went, that time

`build_grid()` in `replay.py` assembles the herd with `MjSpec.attach(prefix=...)`, and
attach prefixes ASSET names along with body and joint names. A hundred copies therefore
arrived as a hundred private sets of the robot's 25 meshes — 2500 copies of the same
62k vertices, which MuJoCo has no way to know are identical. Measured here:

| copies | as attached | sharing the meshes |
|---|---|---|
| 16 | 1545 MB | 475 MB |
| 100 | ~9 GB, never finishes | 1322 MB |

`replay.py` now points each copy's geoms back at the original asset and deletes the
duplicates, which renders a pixel-for-pixel identical frame — verified at n=4. The mesh
geoms are visual only (group 2, contype 0), so nothing about collision changes, and this
scene never collides anything anyway.

Anything else in this tree that replicates a body through `attach()` has the same
problem waiting in it.

## Rendering, viewers, and what counts as looking at it

- Headless frames — `--shot`, `render.py`, `replay.py` — need a GL backend, and
  `MUJOCO_GL=egl` **is not by itself the fast path on this box.** It initialises, it
  renders correct frames, and it renders them on the CPU: there is no NVIDIA EGL vendor
  in WSL (`/usr/share/glvnd/egl_vendor.d/` holds only `50_mesa.json`, and
  `/usr/lib/wsl/lib` ships CUDA and D3D12 but no EGL), so Mesa answers and Mesa picks
  `llvmpipe`. Ask it for the card instead:

  ```bash
  export MUJOCO_GL=egl
  export GALLIUM_DRIVER=d3d12 MESA_LOADER_DRIVER_OVERRIDE=d3d12
  ```

  Mesa then goes through WSL's `libd3d12.so` and `glGetString(GL_RENDERER)` says
  `D3D12 (NVIDIA GeForce RTX 3070)` rather than `llvmpipe`. That one line is worth
  checking before believing any render timing from this machine. Measured on
  `replay.py --robots 100`, 1280x960, 4601 geoms:

  | | ms/frame | peak RSS |
  |---|---|---|
  | `llvmpipe` (what you get by default) | 7740 | 3845 MB |
  | `d3d12` | 1802 | 1377 MB |

  The memory is the part worth noticing: llvmpipe's tile and per-thread buffers across
  16 threads cost 2.5 GB of **host** RAM that the GPU path does not spend at all — and
  host RAM is what kills this distro (above). It is 4x slower as well, but a 12-minute
  render that finishes beats a 52-minute one that takes systemd with it.

- `MUJOCO_GL=osmesa` is the documented software fallback for when EGL will not
  initialise at all. It is **not installed** here (`sudo apt install libosmesa6`), and
  since EGL does initialise and merely lands on a software rasteriser, reaching for
  osmesa would swap one CPU renderer for another. Fix the driver, not the backend.
- `mujoco.viewer` works under WSLg. `mjpython` does not exist on this platform and
  `ros2/tools/view.sh` is a mac wrapper — neither is a bug here.
- In a non-interactive session a viewer window verifies nothing. Write a png and read it.

## The sim regressions, run from here

```bash
cd ~/smalldog/ros2
../rl/.venv/bin/python tools/standalone_sim.py --headless
../rl/.venv/bin/python tools/standalone_sim.py --headless --terrain
../rl/.venv/bin/python tools/standalone_sim.py --course
```

`--lidar` adds the L2 and works too: it reaches sideways into `3d/lidar.py`, which needs
only `numpy` and `mujoco`, not the CAD venv.

The distances quoted in `3d/CLAUDE.md` step 6 were measured **on the mac**. MuJoCo is
deterministic for a given model, seed and version, but not necessarily across platforms
or across versions of MuJoCo — and that step already warns that the flat trot sits near a
bifurcation, where 11 g anywhere moves it by 180 mm. So the mac's numbers are not a
baseline here. Run the *unchanged* tree on this box first and compare against that.

`ros2/smalldog_description/` is generated by the mac and is read-only from this machine
(`rl/CLAUDE.md`). Do not run `generate_model.py` here to "fix" a sim result, and never
commit a regenerated description from this box.

## robot/ — reaching the servo bus

USB serial does not cross into WSL by itself. From an **administrator PowerShell on
Windows**:

```powershell
usbipd list
usbipd bind   --busid <BUSID>      # once per adapter
usbipd attach --wsl --busid <BUSID>
```

then it is `/dev/ttyUSB0` inside WSL (`sudo usermod -aG dialout $USER`, then a new shell).
The FTDI latency timer ships at 16 ms and eats the whole 20 ms tick on its own — the fix
and the arithmetic are in `robot/README.md`, "The 50 Hz budget".

Everything in `robot/` has a `--dry-run` or a `--selftest`. Without the bench physically
attached to *this* machine, those are the only things to run.

## Pushing from here

The mac commits to the same branch, so pull before you push:

```bash
git pull --rebase && git push
```

What this box is authoritative for: `rl/` (including `uv.lock` and `params/`), `robot/`
selftest fixes, and documentation of things learned here. What it is not: the CAD, the
generated description, and anything whose only evidence is a number this platform
produced and the mac has never seen. `rl/runs/` is gitignored and stays that way — a
result worth keeping is a number in a commit message or a file in `params/`, not a 2 GB
directory.
