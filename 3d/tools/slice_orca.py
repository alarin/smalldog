#!/usr/bin/env python3
"""slice_orca.py - slice parts from out/stl/ with OrcaSlicer's CLI, using the
presets that are set up in the OrcaSlicer GUI.

    python3 tools/slice_orca.py                       # the test leg, as printed
    python3 tools/slice_orca.py --name body chassis_top lidar_mount
    python3 tools/slice_orca.py --filament "НИТ TPU @ Neptune" --infill 25% \
            --walls 3 --name feet_tpu foot --copies 4

Writes out/gcode/<name>.gcode (ready to print) and out/gcode/<name>.3mf (the
project, openable in the GUI), and prints time / filament.

Why the preset juggling below: `--load-settings` / `--load-filaments` take a
preset json but do *not* resolve its "inherits", and a GUI user preset is only a
diff against a system one - hand it over as-is and you silently slice with
built-in defaults for everything the diff does not mention (empty layer gcode,
0.4 outer wall, ...).  So the chain is flattened here first, out of the vendor
bundles inside the app, and only then handed to the CLI.
"""
import argparse, glob, json, os, re, shutil, subprocess, sys, tempfile

ORCA = "/Applications/OrcaSlicer.app/Contents/MacOS/OrcaSlicer"
RES  = "/Applications/OrcaSlicer.app/Contents/Resources/profiles"
USER = os.path.expanduser("~/Library/Application Support/OrcaSlicer/user/default")
HERE = os.path.dirname(os.path.abspath(__file__))
STL  = os.path.join(HERE, "..", "out", "stl")
STL2 = os.path.join(HERE, "..", "out", "bench", "stl")   # bench_rig.py, not the robot
GOUT = os.path.join(HERE, "..", "out", "gcode")

# what the printed BOM in README.md asks for on the leg parts
DEFAULT_PARTS    = ["hip_bracket_A", "thigh_A", "shin_A"]
DEFAULT_MACHINE  = "TOP Neptune4"
DEFAULT_PROCESS  = "0.2-0.8 Neptune 4"
DEFAULT_FILAMENT = "TOP НИТ petg черный (scaled)"

DROP = {"from", "instantiation", "renamed_from", "setting_id", "inherits", "type"}


def build_index():
    """(vendor, preset name) -> file, over every vendor bundle in the app.

    Vendor-scoped on purpose: every vendor ships its own `fdm_process_common`
    and friends, and resolving Elegoo's chain against Anycubic's namesake drops
    real values (that is how outer_wall_line_width came out 0.4, not 0.42)."""
    index = {}
    for path in (glob.glob(f"{RES}/*/*/**/*.json", recursive=True)
                 + glob.glob(f"{RES}/*/*/*.json")):
        try:
            d = json.load(open(path))
        except Exception:
            continue
        if not isinstance(d, dict) or "name" not in d:
            continue
        vendor = os.path.relpath(path, RES).split(os.sep)[0]
        ren = d.get("renamed_from", [])          # a preset the vendor renamed:
        ren = [ren] if isinstance(ren, str) else list(ren)   # old user presets
        for key in [d["name"]] + ren:                        # still inherit it
            index.setdefault((vendor, key), path)
            index.setdefault((None, key), path)
    return index


INDEX = build_index()


def resolve(name, vendor=None, seen=()):
    """a system preset -> the merged dict of its whole inherits chain"""
    if name in seen:
        sys.exit(f"inherits loop at {name}")
    path = INDEX.get((vendor, name)) or INDEX.get((None, name))
    if path is None:
        sys.exit(f"parent preset not found: {name!r}")
    vendor = os.path.relpath(path, RES).split(os.sep)[0]
    d = json.load(open(path))
    cfg = resolve(d["inherits"], vendor, seen + (name,)) if d.get("inherits") else {}
    cfg.update({k: v for k, v in d.items() if k not in DROP})
    return cfg


def flatten(kind, name, typ, overrides):
    path = os.path.join(USER, kind, name + ".json")
    if not os.path.exists(path):
        have = sorted(os.path.basename(p)[:-5] for p in glob.glob(f"{USER}/{kind}/*.json"))
        sys.exit(f"no {kind} preset {name!r} in the GUI. have:\n  " + "\n  ".join(have))
    d = json.load(open(path))
    cfg = resolve(d["inherits"]) if d.get("inherits") else {}
    cfg.update({k: v for k, v in d.items() if k not in DROP})
    cfg.update(overrides)
    cfg["type"], cfg["from"] = typ, "User"
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="*", default=DEFAULT_PARTS,
                    help=f"part names in out/stl/ (default: {' '.join(DEFAULT_PARTS)})")
    ap.add_argument("--name", default="test_leg", help="output basename in out/gcode/")
    ap.add_argument("--machine",  default=DEFAULT_MACHINE)
    ap.add_argument("--process",  default=DEFAULT_PROCESS)
    ap.add_argument("--filament", default=DEFAULT_FILAMENT)
    ap.add_argument("--infill", default="40%")
    ap.add_argument("--walls", type=int, default=5)
    ap.add_argument("--copies", type=int, default=1, help="copies of every part")
    ap.add_argument("--no-support", action="store_true")
    ap.add_argument("--stl-dir", action="append", default=None,
                    help="where to look for <part>.stl; repeatable, searched in order "
                         "(default: out/stl then out/bench/stl)")
    a = ap.parse_args()

    dirs = a.stl_dir or [STL, STL2]
    stls = []
    for p in a.parts:
        f = next((os.path.join(d, p + ".stl") for d in dirs
                  if os.path.exists(os.path.join(d, p + ".stl"))), None)
        if f is None:
            sys.exit(f"no {p}.stl in " + " or ".join(os.path.relpath(d) for d in dirs)
                     + " - run mini_dog.py (or bench_rig.py) first")
        stls += [f] * a.copies

    machine = flatten("machine", a.machine, "machine",
                      # the CLI never resolves inherits (that is why we flatten),
                      # but it does match a process's compatible_printers against
                      # the printer's inherits name, so it has to stay
                      {"inherits": json.load(open(f"{USER}/machine/{a.machine}.json")).get("inherits", "")})
    compat = {"compatible_printers": [a.machine, machine["inherits"]],
              "compatible_printers_condition": ""}
    process = flatten("process", a.process, "process", dict(compat, **{
        "name": f"{a.process} - {a.name}",
        "print_settings_id": f"{a.process} - {a.name}",
        "sparse_infill_density": a.infill,
        "wall_loops": str(a.walls),
        # README.md, "Printed BOM": normal(auto), 30 deg, 0.2 mm z-gap
        "enable_support": "0" if a.no_support else "1",
        "support_type": "normal(auto)",
        "support_threshold_angle": "30",
        "support_top_z_distance": "0.2",
        "support_bottom_z_distance": "0.2",
    }))
    filament = flatten("filament", a.filament, "filament",
                       dict(compat, compatible_prints=[], compatible_prints_condition=""))

    tmp = tempfile.mkdtemp(prefix="slice_orca.")
    try:
        cfg = os.path.join(tmp, "cfg"); os.makedirs(cfg)
        for d, fn in ((machine, "machine.json"), (process, "process.json"),
                      (filament, "filament.json")):
            json.dump(d, open(f"{cfg}/{fn}", "w"), indent=1, ensure_ascii=False)
        os.makedirs(GOUT, exist_ok=True)
        cmd = [ORCA, "--datadir", os.path.join(tmp, "data"),
               "--load-settings", f"{cfg}/machine.json;{cfg}/process.json",
               "--load-filaments", f"{cfg}/filament.json",
               "--arrange", "1", "--slice", "0",
               "--export-3mf", a.name + ".3mf", "--outputdir", tmp] + stls
        r = subprocess.run(cmd, capture_output=True, text=True)
        gcode = os.path.join(tmp, "plate_1.gcode")
        if r.returncode or not os.path.exists(gcode):
            sys.exit((r.stdout + r.stderr).strip() or "orca-slicer failed")
        shutil.move(gcode, f"{GOUT}/{a.name}.gcode")
        shutil.move(f"{tmp}/{a.name}.3mf", f"{GOUT}/{a.name}.3mf")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # the header carries the layer count, the totals are in the footer
    fh = open(f"{GOUT}/{a.name}.gcode", errors="ignore")
    head = fh.read(4000)
    fh.seek(max(0, os.path.getsize(f"{GOUT}/{a.name}.gcode") - 400000))
    head += fh.read()
    def g(pat, d="?"):
        m = re.search(pat, head, re.M)
        return m.group(1) if m else d
    time_  = g("^; estimated printing time \\(normal mode\\) = (.+)$")
    grams  = g("^; filament used \\[g\\] = (.+)$")
    layers = g("^; total layer number: (.+)$")
    tall   = g("^; max_z_height: (.+)$")
    copies = f" x{a.copies}" if a.copies > 1 else ""
    support = "no support" if a.no_support else "normal(auto) support 30 deg"
    print(f"{a.name}: {', '.join(a.parts)}{copies}")
    print(f"  {a.machine} / {a.process} / {a.filament}")
    print(f"  {a.walls} walls, {a.infill} infill, {support}")
    print(f"  {time_}, {grams} g, {layers} layers, {tall} mm tall")
    print(f"  -> out/gcode/{a.name}.gcode, out/gcode/{a.name}.3mf")


if __name__ == "__main__":
    main()
