#!/usr/bin/env bash
# run.sh - one debug run in a network-less bubblewrap sandbox, with a run folder.
#
#   tools/bringup/run.sh [label] [-- extra kyty_emulator options]
#
# Environment:
#   KYTY_GAME      game folder                 (required: the folder with eboot.bin)
#   KYTY_BUILD     build folder                (default <repo>/_Build/linux-release)
#   KYTY_RUNS      where run folders are made  (default <repo>/../runs)
#   KYTY_STATE     persistent emulator state: pipeline cache, saves (default <repo>/../state)
#   KYTY_SECONDS   stop the run after this many seconds (default 180; 0 = until it exits)
#   KYTY_PAD=host  pass game controllers in (default none: a drifting stick writes 45,000 log
#                  lines a second)
#   KYTY_KEEP_GOING=0  run without --keep-going (first failure is fatal, as upstream)
#   KYTY_FATAL_ABORT=1 abort() on a fatal error so systemd-coredump keeps a core (large)
#   KYTY_FORCE=1   start although another sandboxed run holds the GPU
#
# The guest executable runs as native x86-64 code inside the emulator process, so the whole
# emulator is treated as untrusted: new user/pid/ipc/uts/net namespaces (only `lo`), no host
# root, tmpfs home, cleared environment. Bound: /usr (ro), GPU nodes, the Wayland socket, the
# game (ro), the build (ro), the state and run folders (rw).
# After the run: triage.py writes triage.txt and triage.json into the run folder.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
label=run
if [ $# -gt 0 ] && [ "$1" != "--" ]; then label=$1; shift; fi
[ "${1:-}" = "--" ] && shift
game=${KYTY_GAME:?set KYTY_GAME to the program folder (the one with eboot.bin)}
build=${KYTY_BUILD:-$repo/_Build/linux-release}
runs=${KYTY_RUNS:-$repo/../runs}
state=${KYTY_STATE:-$repo/../state}
seconds=${KYTY_SECONDS:-180}
xdg=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
wl=${WAYLAND_DISPLAY:-wayland-1}
[ -x "$build/kyty_emulator" ] || { echo "run.sh: no emulator in $build" >&2; exit 2; }
[ -f "$game/eboot.bin" ] || { echo "run.sh: no eboot.bin in $game" >&2; exit 2; }
[ -S "$xdg/$wl" ] || { echo "run.sh: no Wayland socket $xdg/$wl" >&2; exit 2; }
if [ "${KYTY_FORCE:-0}" != 1 ] && pgrep -f 'bwrap .*--unshare-net' >/dev/null; then
    echo "run.sh: another sandboxed run is live (pgrep bwrap); not starting. KYTY_FORCE=1 overrides." >&2
    exit 3
fi
run=$runs/$(date -u +%Y%m%d-%H%M%S)-$label
mkdir -p "$run" "$state"
run=$(cd "$run" && pwd); state=$(cd "$state" && pwd)

opts=(--game "$game" --printf-direction Console)
[ "${KYTY_KEEP_GOING:-1}" = 1 ] && opts+=(--keep-going)
opts+=("$@")

args=(
    --unshare-all --die-with-parent --new-session --clearenv
    --ro-bind /usr /usr
    --symlink usr/bin /bin --symlink usr/bin /sbin --symlink usr/lib /lib --symlink usr/lib /lib64
    --ro-bind /etc/ld.so.cache /etc/ld.so.cache
    --ro-bind /etc/passwd /etc/passwd --ro-bind /etc/group /etc/group
    --proc /proc --dev /dev --tmpfs /tmp --perms 0700 --tmpfs /run --tmpfs /home/sandbox
    --dev-bind /dev/dri /dev/dri
    --ro-bind /sys/dev /sys/dev --ro-bind /sys/devices /sys/devices
    --ro-bind /sys/class/drm /sys/class/drm
    --ro-bind "$xdg/$wl" "/run/$wl"
    --ro-bind "$game" "$game"
    --ro-bind "$build" /opt/kyty
    --bind "$state" /data/state
    --bind "$run" /run/out
    --chdir /data/state
    --setenv HOME /home/sandbox --setenv USER sandbox --setenv PATH /usr/bin
    --setenv TMPDIR /tmp --setenv LANG C.UTF-8
    --setenv XDG_RUNTIME_DIR /run --setenv WAYLAND_DISPLAY "$wl" --setenv SDL_VIDEO_DRIVER wayland
    --setenv SDL_AUDIODRIVER dummy --setenv ALSOFT_DRIVERS null
    --setenv XDG_CACHE_HOME /data/state/cache --setenv MESA_SHADER_CACHE_DIR /data/state/cache/mesa
)
for f in /etc/ld.so.conf /etc/ld.so.conf.d /etc/fonts /usr/share/vulkan; do
    [ -e "$f" ] && args+=(--ro-bind "$f" "$f")
done
[ -n "${KYTY_FATAL_ABORT:-}" ] && args+=(--setenv KYTY_FATAL_ABORT "$KYTY_FATAL_ABORT")
[ -n "${RADV_DEBUG:-}" ] && args+=(--setenv RADV_DEBUG "$RADV_DEBUG")
if [ "${KYTY_PAD:-none}" = host ]; then
    # Game controllers only, by USB vendor (Valve, Microsoft, Sony, Nintendo, 8BitDo, Hori, PDP, PowerA).
    vendors=" ${KYTY_PAD_VENDORS:-28de 045e 054c 057e 2dc8 0f0d 0e6f 24c6} "
    args+=(--ro-bind /sys/class/input /sys/class/input --ro-bind /sys/class/hidraw /sys/class/hidraw)
    for n in /dev/input/event* /dev/input/js*; do
        [ -c "$n" ] || continue
        v=$(cat "/sys/class/input/$(basename "$n")/device/id/vendor" 2>/dev/null || true)
        case "$vendors" in *" ${v,,} "*) args+=(--dev-bind "$n" "$n") ;; esac
    done
    for h in /dev/hidraw*; do
        [ -c "$h" ] || continue
        id=$(sed -n 's/^HID_ID=[0-9A-Fa-f]*:0000\([0-9A-Fa-f]\{4\}\):.*/\1/p' "/sys/class/hidraw/$(basename "$h")/device/uevent" 2>/dev/null || true)
        case "$vendors" in *" ${id,,} "*) args+=(--dev-bind "$h" "$h") ;; esac
    done
    [ -d /run/udev ] && args+=(--ro-bind /run/udev /run/udev)
fi

{
    echo "label=$label"; echo "start_utc=$(date -u +%FT%TZ)"
    echo "commit=$(git -C "$repo" describe --always --dirty 2>/dev/null) branch=$(git -C "$repo" branch --show-current 2>/dev/null)"
    echo "emulator_sha256=$(sha256sum "$build/kyty_emulator" | cut -d' ' -f1)"
    echo "seconds=$seconds pad=${KYTY_PAD:-none}"
    echo "options=${opts[*]}"
    echo "kernel=$(uname -r) mesa=$(pacman -Q mesa 2>/dev/null | cut -d' ' -f2)"
} > "$run/run.txt"

# Refuse to start if the sandbox can see more than the loopback interface.
net=$(bwrap "${args[@]}" /usr/bin/cat /proc/net/dev 2>"$run/sandbox-probe.err" | sed -n "3,\$p" | cut -d: -f1 | tr -d " " | tr "\n" " " || true)
[ "$net" = "lo " ] || { echo "run.sh: sandbox network is not isolated (sees: $net)" >&2; exit 4; }

echo "run folder: $run"
set +e
if [ "$seconds" -gt 0 ]; then
    timeout --signal=TERM --kill-after=10 "$seconds" bwrap "${args[@]}" /opt/kyty/kyty_emulator "${opts[@]}" 2>&1 | grep -av "^Controller axis" > "$run/kyty.log"
else
    bwrap "${args[@]}" /opt/kyty/kyty_emulator "${opts[@]}" 2>&1 | grep -av "^Controller axis" > "$run/kyty.log"
fi
code=${PIPESTATUS[0]}
set -e
{ echo "end_utc=$(date -u +%FT%TZ)"; echo "exit_code=$code   # 124 = stopped by the time limit, 65 = emulator fatal (_Exit(321))"; } >> "$run/run.txt"
if [ -d "$state/_ShaderFailures" ]; then
    cp -a "$state/_ShaderFailures" "$run/ShaderFailures"   # cumulative over runs with this state folder
fi
python3 "$here/triage.py" "$run/kyty.log" --json "$run/triage.json" > "$run/triage.txt" 2>&1 || true
cat "$run/run.txt"; echo; cat "$run/triage.txt"
exit "$code"
