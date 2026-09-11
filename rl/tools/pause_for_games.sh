#!/bin/bash
# pause_for_games.sh -- freeze the trainer while a game has the card.
#
#   ./tools/pause_for_games.sh                     # default: any Steam game
#   ./tools/pause_for_games.sh Cyberpunk2077.exe eldenring.exe
#   ./tools/pause_for_games.sh & disown            # alongside a run, in the background
#
# The 5070 Ti box is also the gaming PC.  A game and PPO sharing the card is bad
# for both, and brax has no mid-run resume (a restart from a checkpoint drops the
# optimizer state and the LR schedule), so the pause is a signal, not a
# checkpoint: SIGSTOP freezes the python where it stands, SIGCONT picks it up
# there.  CUDA contexts and the compiled graph survive a stop; nothing is lost
# and nothing recompiles.
#
# Windows processes are visible from WSL through tasklist.exe, so the whole loop
# lives on this side.  `gameoverlayui.exe` is Steam's in-game overlay -- it exists
# exactly while a Steam game is running, whichever game it is -- so that is the
# default.  Non-Steam games go on the command line by image name.
#
# What a stopped trainer does NOT give back is its VRAM: at --mem-fraction 0.6
# that is ~10 GB of the 16 held by a process that is not running.  WDDM can page
# an idle process's allocations out to host RAM when a game wants the card, but
# whether it does so for this process is a measurement, not a promise -- check
# Task Manager's dedicated GPU memory with the trainer stopped and a game up.  If
# it does not, the lever is --mem-fraction on the run itself; the 2048-env
# measurements in ../GPU.md say how low it can go.
#
# The trap matters: a watcher killed while the trainer is stopped would leave the
# run frozen with no one to wake it, and a frozen run looks exactly like a
# compiling one from outside (jaxenv.py, "a run that looks stuck").  So the
# watcher's own exit -- ^C, SIGTERM, anything -- sends CONT first.

TASKLIST=/mnt/c/Windows/System32/tasklist.exe
POLL=${POLL:-5}
# Anchored on the executable on purpose: an unanchored 'train_ppo.py' also
# matches any shell whose command line MENTIONS the script -- a grep, a tail, the
# terminal that launched it -- and stopping those is how the first test of this
# file froze its own test harness.
PATTERN=${PATTERN:-'^\S*python[0-9.]*( -u)? train_ppo\.py'}
GAMES=("$@")
[ ${#GAMES[@]} -eq 0 ] && GAMES=(gameoverlayui.exe)

[ -x "$TASKLIST" ] || { echo "no $TASKLIST: this is not WSL, nothing to watch" >&2; exit 1; }

trainers() { pgrep -f "$PATTERN"; }

resume() {
    local pids; pids=$(trainers)
    [ -n "$pids" ] && kill -CONT $pids 2>/dev/null
}
trap 'resume; exit 0' INT TERM EXIT

game_running() {
    local g
    for g in "${GAMES[@]}"; do
        # tasklist exits 0 either way; the match is the image name in the CSV
        "$TASKLIST" /FI "IMAGENAME eq $g" /FO CSV /NH 2>/dev/null | grep -qi "\"$g\"" && return 0
    done
    return 1
}

echo "watching for: ${GAMES[*]}   (poll ${POLL}s, trainer pattern '$PATTERN')"
paused=0
while true; do
    pids=$(trainers)
    if game_running; then
        if [ $paused -eq 0 ] && [ -n "$pids" ]; then
            kill -STOP $pids && paused=1 && echo "$(date +%T)  game up -> STOP $pids"
        fi
    elif [ $paused -eq 1 ]; then
        kill -CONT $pids 2>/dev/null; paused=0
        echo "$(date +%T)  game gone -> CONT $pids"
    fi
    sleep "$POLL"
done
