# GPU traces

A GPU trace (`.lxtrace`) holds what the emulated GPU consumed over a window, in the order it consumed it, after a snapshot of the GPU state at the window's start. It records guest inputs (command words, guest memory, engine registers, maps and presents), never host Vulkan calls, so a replay runs the shader translation, the caches and the Vulkan backend of the build that replays it.

## Replay

```sh
eden-cli --replay <trace> --out <dir> [--frames N] [--device N]
```

The replay needs no game, no window and no keys. It brings up the GPU on a bare `Core::System` (`System::InitializeForGpuReplay`: kernel, Host1x and the GPU, no process, audio or services), restores the snapshot and feeds the records through the GPU in order. It writes per present:

- `frame-NNNN.png`: the composited frame at the size of its largest layer (crop rectangle), through the renderer's screenshot path.
- `frames.json`: per frame its file, size, XXH3-128 hash of the RGBA bytes and the wall time since the previous frame, plus the program id and the device that replayed it.

The window is a `VK_EXT_headless_surface` surface (`WindowSystemType::Headless`), so any Vulkan driver with that extension works, Mesa's lavapipe included.

`eden-cli --gpu-trace-sample <trace>` writes a synthetic trace that needs no game: one Switch 2 channel whose four frames are two CPU-written images and two sets of 3D engine clears (whole and scissored). The unit tests read it back (`[gpu_trace]`).

## File format

A 64-byte header (magic `LXGTRACE`, format version, header size), then one zstd stream of records: a 16-byte header (`type`, `size`) and `size` payload bytes. `video_core/gpu_trace/trace_file.h` defines every record. In order:

1. `Config`: JSON with the program id, whether it is a Switch 2 title, the memory layout and the output size.
2. Maps: `AddressSpace` (a GMMU with its page geometry), `GmmuMap`, `SmmuMap` (device pages to physical pages, so aliasing survives).
3. `Channel` per GPU channel, then `State` records: engine registers, the pusher's method state, the Switch 2 macro RAM, QMD scheduler state, syncpoints and semaphore regions (`gpu_trace/state.h`).
4. `Memory`: physical ranges and their content. Content is a `Blob` stored once and named by its XXH3-128 hash, so identical pages cost nothing after the first.
5. `SnapshotEnd`, then the stream: `Memory` (a CPU write the GPU sees from that point on), `Submit` (GPFIFO entries, prefetch words, submit-time word copies), map changes, `Present` (the layers and fences of a composite request) and `Resume`.
6. `End`.

Physical addresses are offsets into guest memory (`Core::DeviceMemory`). The replay maps them through a process page table whose addresses are those offsets.

## Lavapipe

Lavapipe creates the device (every required feature, extension and limit is there). Its gaps:

| Gap | What the replay does |
|---|---|
| Subgroups are 8 wide and can't be forced to 32 | The guest warp size is requested only where the device's subgroup size range includes 32. Warp operations (SHFL, VOTE, ballot) then run on 8 lanes and give other results than a GPU: frames that use them differ. |
| No depth bounds test | The existing fallback skips the test. |
| No ASTC | The existing CPU decode. |
| MSAA 1x and 4x only | 2x and 8x targets fail to create. |
| CPU rasterizer | Correct, slow: a trace replays at a few frames a second. |

Frames match bit for bit only between runs of the same build on the same driver. Across drivers they differ by precision, so expected frames are kept per device class.

## What a trace can't hold

- Guest CPU timing. A bug that comes from the order between the CPU and the GPU thread replays in the recorded order and may not reproduce.
- Content that never reaches guest memory: resolution-scaled images and MSAA targets that can't be downloaded. Record at 1x for exact traces.
- Driver bugs: those need the same driver to replay.
