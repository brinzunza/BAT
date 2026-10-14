# Agent instructions for `dtfxStrategy/`

Read the repo-root `../AGENTS.md` first — it applies here too.

This folder is one research project among several siblings in the repo (`coinbaseResearch/`,
`flightDataResearch/`, etc.) — it is not itself a shared library. The shared infrastructure for
trade validation, sizing, settlement, and Monte Carlo now lives **outside** this folder, at
`../quant_engine/` (Python interface) and `../cpp_engine/` (C++ core it wraps).

If you are writing, porting, or validating a strategy in `research_scripts/`:

- Read `../quant_engine/docs/ADDING_A_STRATEGY.md` before writing any sizing, margin, or Monte
  Carlo logic yourself.
- `research_scripts/` still holds ~480 independent analysis scripts, most predating
  `quant_engine`. They keep working unchanged — nothing here requires migrating them. New work
  should go through `quant_engine.portfolio.run_portfolio()` rather than adding another
  one-off sizing/settlement implementation.
- `dtfx_engine` (the compiled C++ extension, built from `../cpp_engine/`) still lands in
  `research_scripts/` after `bash ../cpp_engine/build.sh`, so `import dtfx_engine` keeps working
  from any script here exactly as before.
- Already-published numbers in `research_scripts/beid_goal_hunt/REPORT.md` and elsewhere are
  the ground truth to reproduce, not to casually recompute differently.
