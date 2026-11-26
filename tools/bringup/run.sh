#!/usr/bin/env bash
# run.sh - one debug run in a network-less bubblewrap sandbox, with a run folder.
#
#   tools/bringup/run.sh [label] [-- extra kyty_emulator options]
#
# Environment:
#   KYTY_GAME      game folder                 (required: the folder with eboot.bin)
#   KYTY_BUILD     build folder                (default <repo>/_Build/linux-debug, with symbols)
#   KYTY_RUNS      where run folders are made  (default <repo>/../runs)
#   KYTY_STATE     persistent emulator state: pipeline cache, saves (default <repo>/../state)
#   KYTY_SECONDS   stop the run after this many seconds (default 180; 0 = until it exits)
#   KYTY_PAD=host  pass game controllers in (default none: a drifting stick writes 45,000 log
#                  lines a second)
#   KYTY_KEEP_GOING=0  run without --keep-going (first failure is fatal, as upstream)
#   KYTY_FATAL_ABORT=1 abort() on a fatal error so systemd-coredump keeps a core (large)
#   KYTY_AUDIO=none    no sound (default host: the PulseAudio/PipeWire socket is passed in)
#   KYTY_AUTO_CROSS=<s>  no controller needed: Cross is pressed every 2 s for the first <s> seconds
#   KYTY_PROFILE_AT=<s>|stall  perf-record the emulator for KYTY_PROFILE_SECONDS (20) from that second;
#                      writes perf.data, perf.txt (hot functions), perf-threads.txt
#   KYTY_SHADER_CACHE=0  do not keep translated shaders between runs (default: kept in the state
#                      folder and compiled at boot on all cores, so every run starts further)
#   KYTY_TRACE=0       run without --trace-calls
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
# Machine-local settings (KYTY_GAME and friends) live outside the repository.
[ -f "$repo/../bringup-local.env" ] && . "$repo/../bringup-local.env"
label=run
if [ $# -gt 0 ] && [ "$1" != "--" ]; then label=$1; shift; fi
[ "${1:-}" = "--" ] && shift
game=${KYTY_GAME:?set KYTY_GAME to the program folder (the one with eboot.bin)}
build=${KYTY_BUILD:-$repo/_Build/linux-debug}
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
[ "${KYTY_TRACE:-1}" = 1 ] && opts+=(--trace-calls)
[ "${KYTY_SHADER_CACHE:-1}" = 1 ] && opts+=(--shader-cache)
[ "${KYTY_AUTO_CROSS:-0}" != 0 ] && opts+=(--auto-cross "$KYTY_AUTO_CROSS")
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
    --ro-bind "$build" "$build"
    --bind "$state" /data/state
    --bind "$run" /run/out
    --chdir /data/state
    --setenv HOME /home/sandbox --setenv USER sandbox --setenv PATH /usr/bin
    --setenv TMPDIR /tmp --setenv LANG C.UTF-8
    --setenv XDG_RUNTIME_DIR /run --setenv WAYLAND_DISPLAY "$wl" --setenv SDL_VIDEO_DRIVER wayland
    --setenv XDG_CACHE_HOME /data/state/cache --setenv MESA_SHADER_CACHE_DIR /data/state/cache/mesa
)
for f in /etc/ld.so.conf /etc/ld.so.conf.d /etc/fonts /usr/share/vulkan; do
    [ -e "$f" ] && args+=(--ro-bind "$f" "$f")
done
if [ "${KYTY_AUDIO:-host}" = host ] && [ -S "$xdg/pulse/native" ]; then
    args+=(--ro-bind "$xdg/pulse/native" /run/pulse/native --setenv PULSE_SERVER unix:/run/pulse/native)
    [ -S "$xdg/pipewire-0" ] && args+=(--ro-bind "$xdg/pipewire-0" /run/pipewire-0)
else
    args+=(--setenv SDL_AUDIODRIVER dummy --setenv ALSOFT_DRIVERS null)
fi
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
start_epoch=$(date +%s)
if [ -n "${KYTY_PROFILE_AT:-}" ]; then
    # Host CPU profile of the emulator, taken from outside the sandbox.
    ( if [ "$KYTY_PROFILE_AT" = stall ]; then
          # Start when frames were being shown and then none is for 4 s.
          seen=0; prev=0
          while sleep 4; do
              pgrep -x kyty_emulator >/dev/null || exit 0
              now=$(grep -ac 'flip_arg  *= ' "$run/kyty.log" 2>/dev/null || true); now=${now:-0}
              [ "$now" -gt 300 ] && seen=1
              grep -aq '^--- Error\|^terminate called\|context is lost' "$run/kyty.log" && exit 0
              [ "$seen" = 1 ] && [ "$now" = "$prev" ] && break
              prev=$now
          done
      else
          sleep "$KYTY_PROFILE_AT"
      fi
      pid=$(pgrep -x kyty_emulator | head -1); [ -n "$pid" ] || exit 0
      perf record -q -F 499 -g -p "$pid" -o "$run/perf.data" -- sleep "${KYTY_PROFILE_SECONDS:-20}" 2>"$run/perf.err"
      perf report -i "$run/perf.data" --no-children --sort comm,dso,symbol --stdio 2>/dev/null | grep -v "^$" | head -400 > "$run/perf.txt"
      perf report -i "$run/perf.data" --no-children --sort tid --stdio -g none 2>/dev/null | grep -v "^#\|^$" | head -40 > "$run/perf-threads.txt"
      perf report -i "$run/perf.data" --no-children --sort tid,dso,symbol --stdio -g none 2>/dev/null | grep -v "^#\|^$" | head -150 > "$run/perf-flat.txt"
    ) &
fi
set +e
if [ "$seconds" -gt 0 ]; then
    timeout --signal=TERM --kill-after=10 "$seconds" bwrap "${args[@]}" "$build/kyty_emulator" "${opts[@]}" 2>&1 | gawk 'BEGIN{s=systime()} /^Controller axis/{next} {t=systime()-s; if(t!=l){print "@@T " t; l=t} print}' > "$run/kyty.log"
else
    bwrap "${args[@]}" "$build/kyty_emulator" "${opts[@]}" 2>&1 | gawk 'BEGIN{s=systime()} /^Controller axis/{next} {t=systime()-s; if(t!=l){print "@@T " t; l=t} print}' > "$run/kyty.log"
fi
code=${PIPESTATUS[0]}
set -e
{ echo "end_utc=$(date -u +%FT%TZ)"; echo "exit_code=$code   # 124 = stopped by the time limit, 65 = emulator fatal (_Exit(321))"; } >> "$run/run.txt"
# Kernel side of the run: GPU ring timeouts and resets name the guilty process.
journalctl -k --since "@$start_epoch" --no-pager 2>/dev/null | grep -i 'amdgpu\|\[drm\]' | grep -v 'No EDID' > "$run/kernel-gpu.txt" || true
[ -s "$run/kernel-gpu.txt" ] || rm -f "$run/kernel-gpu.txt"
# After a GPU ring timeout the kernel keeps a device dump for a few minutes only; it is root-readable.
if grep -qi "ring .* timeout" "$run/kernel-gpu.txt" 2>/dev/null; then
    for d in /sys/class/drm/card*/device/devcoredump/data; do
        [ -e "$d" ] && sudo -n cat "$d" > "$run/gpu-devcoredump.txt" 2>/dev/null
    done
    [ -s "$run/gpu-devcoredump.txt" ] || rm -f "$run/gpu-devcoredump.txt"
fi
if [ -d "$state/_ShaderFailures" ]; then
    cp -a "$state/_ShaderFailures" "$run/ShaderFailures"   # cumulative over runs with this state folder
fi
python3 "$here/triage.py" "$run/kyty.log" --json "$run/triage.json" > "$run/triage.txt" 2>&1 || true
cat "$run/run.txt"; echo; cat "$run/triage.txt"
[ -f "$run/kernel-gpu.txt" ] && { echo "KERNEL GPU MESSAGES (kernel-gpu.txt):"; grep -i 'timeout\|reset\|Process' "$run/kernel-gpu.txt" | cut -c1-200; }
exit "$code"
