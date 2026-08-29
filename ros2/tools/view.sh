#!/bin/bash
# Interactive standalone_sim.py, with the arguments passed straight through:
#
#     ./tools/view.sh                     # flat ground
#     ./tools/view.sh --terrain --lidar   # rough ground, with the LiDAR cloud drawn
#
# Two macOS things this exists to handle, both of which look like a broken install:
#
#   1. MuJoCo's passive viewer has to run under `mjpython`, not `python` - the window
#      needs the main thread, and plain python exits with "launch_passive requires
#      mjpython on macOS".  The headless runs (--headless, --course) do not, and are
#      fine with ../3d/.venv/bin/python.
#   2. mjpython dlopens the interpreter, and the venv's python is a uv build whose
#      libpython3.12.dylib sits outside every path its @rpath knows.  It dies with
#      "Library not loaded: @rpath/libpython3.12.dylib" before running a line of the
#      script.  sysconfig knows where that dylib actually is, so ask it.
set -e
here="$(cd "$(dirname "$0")" && pwd)"
py="$here/../../3d/.venv/bin/python"
[ -x "$py" ] || { echo "view.sh: no CAD venv at $py" >&2; exit 1; }
libdir="$("$py" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
exec env DYLD_FALLBACK_LIBRARY_PATH="$libdir" "$(dirname "$py")/mjpython" \
     "$here/standalone_sim.py" "$@"
