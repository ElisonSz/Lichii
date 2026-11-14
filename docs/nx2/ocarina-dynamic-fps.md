# Ocarina Dynamic FPS add-on

`Dynamic FPS by Dark (built-in)` is a game Add-On, enabled by default for Ocarina.
It targets title `0400486024DEC000`, module `main`, build ID
`E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348` only. Max Resolution remains a
separate, default-on Add-On. Either can be disabled independently in Add-Ons;
the existing disabled-add-on configuration persists that choice.

## Credit and source

Dark, an Eglantine developer, wrote the routine. This port preserves the call and
120-byte routine from Frizti's `oot-gather-fix` commit
`765659af4ccb7201d1145c57f3b45ce84653cea0`. That commit explicitly states that
Dark gave permission to bundle it. Frizti supplied the built-in integration.
CTGL's #299 changed one instruction: the delta cap is `CNTFRQ / 10` instead of
`CNTFRQ / 20`.

The routine's SHA-256 is
`0afb4af10a78bb49372e6807fc5b71d9fa99d828b7625f0fab7429fc66a4b1e6`
(`3c15dfb57f748328ad7d1c1675941dc2ace32a5e7b10907368ceac6a9a8c5ee1` before the
cap change).
The call's SHA-256 is
`e4d3f65ad1b568b557f02b9a087a9c304240ac70d6ef474dac83a1f97737515e`.
This port adds original-byte and memory-layout guards, and
transactional validation before writing either record.

## Verified original layout

An independently extracted original `main.text` has SHA-256
`f71eb7b75b11dcc8d955f3f6e987a098573288d9ce41aa22a96272541e858c5f`, matching
the text hash in its NSO header. Its instruction at module offset `0x6E6CC4`
is `28 18 20 1E`, or `fdiv s8, s1, s0`. The patch replaces it with
`bl 0x3C46200` and executes the original division first in the new routine.
The enclosing function saves and restores its return address. The continuation
does not consume the routine's clobbered scratch registers or flags before
intervening calls or redefinition.

The original text ends at `0x3C461D0`; read-only data starts at `0x3C47000`.
The routine occupies `0x3C46200` through `0x3C46277`, in executable alignment
padding, not original text. `AppLoader_NSO::LoadModule` zero-initializes the
module image before copying segments and rounds the text mapping to a page.
The loader checks that this space is still zero-filled executable padding
before installing the routine.

Data ends at `0x4D46B80` and BSS ends at `0x4EA0000`. The routine stores its
previous timestamp and smoothed delta at `0x4E9FE00` and `0x4E9FE08`. The
loader requires both words to fit BSS and remain zero-initialized. These
checks establish addressability and initialization, not exclusive ownership
of that storage throughout gameplay.

Any changed user-mod bytes in either patch record or the timer state take
precedence over the entire built-in patch. An incompatible build, original
instruction, layout, or truncated module leaves both patch records untouched.

## Water ripple passes

The ripple simulation (compute shaders `fb4203a7d928ed42` then `e2ce39e630ce65f3`,
256x256 ping-pong textures) sets its pass count at `main+0x475F8` as
`iterations * (int)step`, with `iterations` 6. Truncation is harmless at the
original step of exactly 1.0, but any step below 1.0 runs no passes: at 120 fps
the ripples stop, and the grid origin that follows the player keeps moving with
no pass to apply the shift. Measured before this patch at the save in Zora's
Domain, frames that ran the simulation out of 10: no patch 10, FPS Unlock capped
at 60 fps 4 (step 0.998 to 1.003), FPS Unlock at about 99 fps 0.

Both built-ins replace that `fcvtzs` with `bl 0x3C46300` and drop the `mul` that
followed it. The routine adds `iterations * step` to a remainder kept per water
object, runs the rounded whole number of passes and keeps the rest, with at least
one pass whenever the step is positive so the origin shift is always applied. A
paused game (step 0) still runs none. The final pass `6a7f45ab293a93b1` was never
affected.

The remainders live in 8 slots of 16 bytes from `0x4E9FE10` (the water object
from `x21` at the patch site, then the remainder as a float), with the next slot
to reuse at `0x4E9FE90`. An object without a slot takes the next one in turn, so
two water objects drawn in the same frame don't pull from one shared fraction.
The timer state is 0x98 bytes from `0x4E9FE00`. The routine is 148 bytes at
`0x3C46300-0x3C46393`, clear of the Dynamic FPS routine at `0x3C46200` and of the
ultrawide caves at `0x3C46400` and up. It uses w8, x9-x12 and s0-s2, none read
by the game again before it writes them.

## Timing limits

The routine reads `CNTPCT_EL0` and `CNTFRQ_EL0`. After its first call, it caps
the timestamp difference at `CNTFRQ / 10`, or 100 ms, then smooths it by adding
one eighth of the signed difference from the previous smoothed value.
It calculates `s2 = (smoothed_ticks / CNTFRQ) * s1`. It replaces the original
`s8 = s1 / s0` only when `s2 > 1.03 * s8`.
The original code selects `s1` as 30 or 60 and converts the integer field at
`x21 + 0x2C8` into `s0`, so the result is a frame-step scale, not seconds.

This is not a promise of correct game speed at every frame rate. It retains
the original step as a floor, caps long frames, and smooths changes. Actual
gameplay speed and the runtime value of that frame-rate field still require
live validation.

In Lichii, `CoreTiming::GetClockTicks` scales the counter for CPU Boost and
Overclock, while Dynarmic's advertised counter frequency stays unchanged.
Boost uses a 1.7 multiplier and Overclock uses 2. Enabling core-speed
synchronization can further divide the counter by the configured speed limit.
Those settings alter the elapsed seconds this routine calculates. CPU Clock
Normal and an unscaled counter are prerequisites for evaluating its timing;
the Add-On does not change global clock behavior or automatically override
the user's settings. Correct gameplay speed with the Add-On is unverified.

## FPS Unlock: sound manager at 60 Hz

`FPS Unlock (built-in)` lowers the step below 1.0 above 60 fps (0.5 at 120 fps).
The game's sound engine expects at least one whole 60 fps step per update: its
fade and delay timers and parameter ramps count updates from the step it is
given. With less, some sounds are not released when an area unloads; they pile
up until the game's voice budget is full, and then most new sounds and the
music stop until a restart (sword swings, spin and jump attacks, swimming,
area and shop music; #716, #1092).

The sound manager job at `0x415300` (phase in `w2`: phase 0 reads the step and
updates the sound engine, phase 1 follows) is entered every frame. FPS Unlock
replaces its first instruction, `pacibsp`, with `b 0x3C46800` (124-byte gate)
and its step load at `0x415334`, `ldr s0, [x8, #0x10]`, with `bl 0x3C46280`
(16 bytes, returns `max(step, 1.0)`). In phase 0 the gate runs the job as before
when the step is at least 1.0. Below that it adds the step to an accumulator
at `0x4E9FEA0` and runs the job only once a whole step has built up, keeping
the remainder; otherwise it returns. Other phases follow that frame's decision
(flag at `0x4E9FEA4`). So above 60 fps the sound manager runs at 60 Hz with
whole steps and sound still follows real time; at 60 fps and below nothing
changes. FPS Unlock's timer state grows to `0xA8` bytes to cover both words.
Dynamic FPS never lowers the step below 1.0 and does not include the gate.

Both caves are in the executable padding after the text (`0x3C46280` follows
the FPS routine at `0x3C46200`; `0x3C46800` is past the other caves in use).

## Tests

The `[patch_manager]` tests call production `PatchManager::PatchNSO` with
synthetic module images and real IPS/IPS32 user mods. They cover default-on and
disable behavior, title/module/build gates, original-byte and layout
guards, truncated inputs, all-record user-mod precedence, and Max Resolution.
The `[addon_config]` tests round-trip the production configuration reader and
writer, including legacy configurations and repeated enable/disable cycles.

Run them from a clang/Ninja test build with:

```sh
cmake --build build --target tests --parallel 4
build/bin/tests '[patch_manager],[addon_config]'
```
