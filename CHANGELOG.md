## v1.1.0
- Final build for release with basic functionality
- Released in conjunction with HRfunc paper

## v1.1.1
- Exposed the preprocess_fnirs function in the library

## v1.3.0
- New: NiceGUI-based desktop GUI installed via `pip install hrfunc[gui]`
- New: `hrfunc` console script launches the GUI; `hrfunc <path>` preloads
- New: 3-path welcome screen (Open my data / Browse HRF library /
  Recent projects) with XDG-cache-backed recent-folder list
- New: BIDS-aware dataset tree with case-insensitive substring filter
- New: Inspect tab — channel list, 2D probe layout, event-annotation
  table; loads MNE Raw lazily via an in-memory LRU(3) cache
- New: Preprocess tab — full pipeline button + diagnostic stage toggles
  + before/after preview; results stored in a separate `processed_cache`
- New: HRFs tab — toeplitz deconvolution with multi-event picker,
  log-scale lambda slider, duration field, progress bar; canonical
  mode renders an SPM-style double-gamma reference HRF
- New: Activity tab — toeplitz (reuses estimated HRFs) and canonical
  (uses bundled HRF library) modes; lens.plot_nirx-style overlay
  preview with event markers
- New: Quality tab — per-scan SNR / skewness / kurtosis / SCI metrics
  across raw / preprocessed / deconvolved stages plus a dataset-wide
  aggregate that walks every scan in the manifest
- New: HRF Library page at `/library` — three-pane Browser-persona
  flow with Context Filter sidebar, plotly 3D HRtree viz, and per-HRF
  detail pane with trace preview
- New: HRF gallery — HRFs-tab preview replaced by a clickable channel
  grid (one mini-plot per channel) with a per-channel detail panel
  showing full trace + ±1 std shading
- New: Export tab — saves processed Raw (SNIRF/FIF), activity Raw
  (SNIRF/FIF), montage HRFs (JSON via montage.save), per-channel HRF
  plot PNGs to a folder, and quality metrics as a flat CSV (one row
  per scan × stage)
- New: Cross-component event bus (`scan_selected`, `scan_loaded`,
  `preprocess_done`, `hrf_estimated`, `activity_estimated`,
  `quality_computed`, `library_filter_changed`,
  `library_selection_changed`) for panel reactivity
- New: Folder-scan I/O subsystem (`hrfunc.io.scan_folder`,
  `classify_path`, `RawCache`) reusable from the Python API
- New: `progress_callback` kwarg added to `montage.estimate_hrf` and
  `montage.estimate_activity` for non-GUI progress tracking
- New: `hrfunc install-shortcut` / `hrfunc uninstall-shortcut`
  subcommands add or remove a system-level launcher (Spotlight on
  macOS, Start menu on Windows, Activities on Linux) so non-coder
  researchers can open HRfunc like any other desktop app. First GUI
  launch prompts the user to install the shortcut automatically.
- New: "Show MNI brain" toggle on the `/library` HRtree explorer —
  overlays a translucent fsaverage pial surface beneath the HRF
  scatter for spatial context. Mesh is bundled in the wheel
  (~2.5k verts / 5k tris in MNI-meter coords) so no fsaverage
  download is required.
- See [docs/external/gui_guide.md](docs/external/gui_guide.md) for the
  full GUI walkthrough and troubleshooting guide

## v1.3.1
- Fix (packaging, affects all v1.3.0 installs): the bundled HRF library
  was missing from the published wheel and sdist, so `pip install
  hrfunc` shipped an empty library. The `/library` HRtree explorer
  showed no HRFs and library-backed activity estimation had nothing to
  match against. This failed silently — `tree()` treats an absent file
  as "no HRFs loaded" rather than an error, so it only reproduced on a
  pip install, never from a source checkout. Anyone on v1.3.0 should
  upgrade.
- Fix: single-scan activity Save could write the *wrong* scan's data;
  saving is now gated on the result belonging to the selected scan,
  matching the preview's predicate.
- Fix: Activity group-HRF count ignored excluded subjects, so it could
  report "GROUP HRFs (N)" and pass the >=2 gate on a pool that excluded
  a subject.
- Fix: an empty single-scan estimate (activity or HRF) left the previous
  scan's result on screen with no error shown.
- Fix: cross-project state leak, ROI-save mixing, and an atlas lookup
  crash, plus a second review sweep across workflow / edge-case /
  scientific-guidance lenses.
- Fix: the 4 long-standing test failures are resolved at root cause; the
  suite is now fully green (1166 passing).
- New: bulk "Save all activity" lets you pick a destination folder
  instead of writing next to each source file — the escape hatch when
  the source dataset lives on a read-only or mounted volume, which
  previously dead-ended every save. Flat-folder name collisions are
  disambiguated rather than silently overwritten.
- New: the submission health pill honors HRServ `node_role` — a healthy
  node running as a replica no longer shows green while every upload
  fails. Adds a DEGRADED state and aggregates both nodes into one pill.