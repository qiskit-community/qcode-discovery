"""Entry point for running OpenEvolve-based evolutionary search.

This script connects to a LiteLLM-compatible proxy and uses OpenEvolve
to evolve the ``generate_candidates`` function in
``evolve/seed_solution.py``.  It does **not** evolve individual polynomial
pairs -- it evolves the *strategy* for generating them.

Prerequisites
-------------
* A running LiteLLM proxy (or any OpenAI-compatible endpoint).  Pass the
  URL via ``--api-base``, ``$OPENAI_API_BASE``, ``$LITELLM_API_BASE``, or
  let it default to ``http://localhost:4000/v1``.
* The ``evolve`` dependency group: ``uv sync --group dev --group evolve``.

How it works
------------
1. Loads ``evolve/config.yaml`` and applies CLI overrides (model list,
   temperature, api_base, iterations).
2. Passes ``evolve/seed_solution.py`` (initial program) and
   ``evolve/openevolve_evaluator.py`` (fitness function) to OpenEvolve.
3. OpenEvolve mutates the code between ``# EVOLVE-BLOCK-START`` and
   ``# EVOLVE-BLOCK-END`` markers using LLM-generated diffs.
4. Each mutation is evaluated via the two-stage cascade in
   ``openevolve_evaluator.py`` (see that module's docstring for details).
5. MAP-Elites with 5 islands and periodic migration maintains diversity
   across ``lattices_with_high_k`` and ``num_high_k`` feature dimensions.
6. The LLM receives structured evaluation artifacts (best code found,
   per-lattice breakdown, errors) as feedback for the next mutation.

W&B integration
---------------
Pass ``--wandb`` to enable live dashboards.  A background ``WandbSyncer``
thread tails the metrics JSONL file written by evaluator subprocesses
(where ``wandb.run`` is ``None``) and forwards each record to W&B with
running-best tracking.

Output
------
* Best evolved program: ``results/evolution/<run_name>/best_generate_candidates.py``
* OpenEvolve checkpoints: ``results/evolution/<run_name>/checkpoints/``
* Metrics JSONL: ``results/evolution_metrics.jsonl``
* Discovered codes: ``results/discovered_codes.json``

Usage::

    # Single model
    uv run python evolve/run_evolution.py \\
        --model anthropic/claude-sonnet-4-5-20250514 --iterations 100

    # Ensemble of models
    uv run python evolve/run_evolution.py \\
        --models gemini/gemini-2.5-flash anthropic/claude-sonnet-4-5-20250514 \\
        --iterations 200 --run-name ensemble_v1

    # Resume from checkpoint
    uv run python evolve/run_evolution.py \\
        --resume results/evolution/run_20260218/checkpoints/checkpoint_100 \\
        --iterations 200

    # With W&B tracking
    uv run python evolve/run_evolution.py \\
        --model anthropic/claude-sonnet-4-5-20250514 --iterations 100 --wandb
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
from datetime import datetime
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SEED_SOLUTION = str(Path(__file__).parent / "seed_solution.py")
SEED_SOLUTION_MILP = str(Path(__file__).parent / "seed_solution_milp.py")
SEED_SOLUTION_NONCSS = str(Path(__file__).parent / "seed_solution_noncss.py")
EVALUATOR = str(Path(__file__).parent / "openevolve_evaluator.py")
EVALUATOR_NONCSS = str(Path(__file__).parent / "openevolve_evaluator_noncss.py")
DEFAULT_CONFIG = str(Path(__file__).parent / "config.yaml")
DEFAULT_CONFIG_NONCSS = str(Path(__file__).parent / "config_noncss.yaml")
EVOLUTION_BASE = str(Path(PROJECT_ROOT) / "results" / "evolution")
METRICS_FILE = str(Path(PROJECT_ROOT) / "results" / "evolution_metrics.jsonl")


def _resolve_api_base(args) -> str:
    """Resolve the API base URL from args or environment."""
    if args.api_base:
        return args.api_base
    for var in ("OPENAI_API_BASE", "LITELLM_API_BASE"):
        val = os.environ.get(var)
        if val:
            return val
    return "http://localhost:4000/v1"


def _resolve_output_dir(args) -> str:
    """Compute the output directory from --output or --run-name."""
    if args.output:
        return args.output
    run_name = args.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return str(Path(EVOLUTION_BASE) / run_name)


def _build_config(args, api_base: str, model_names: list[str] | None):
    """Load config YAML and apply CLI overrides for models, temperature, etc."""
    import yaml
    from openevolve import Config
    from openevolve.config import LLMModelConfig

    config = Config.from_yaml(args.config)
    config.max_iterations = args.iterations

    # Propagate api_base to all model configs.
    # Setting config.llm.api_base alone does NOT propagate because
    # __post_init__ already ran during from_yaml().
    config.llm.api_base = api_base
    config.llm.update_model_params({"api_base": api_base}, overwrite=True)

    # Override model list if CLI specifies models; otherwise keep config as-is
    if model_names is not None:
        config.llm.models = [
            LLMModelConfig(name=name, weight=1.0)
            for name in model_names
        ]
        # Propagate ALL shared params to the new models -- not just api_base.
        # Fresh LLMModelConfig objects have None for max_tokens, timeout, retries,
        # etc.  Without propagation the retry loop crashes (range(None + 1)).
        shared = {
            "api_base": api_base,
            "api_key": config.llm.api_key,
            "temperature": config.llm.temperature,
            "top_p": config.llm.top_p,
            "max_tokens": config.llm.max_tokens,
            "timeout": config.llm.timeout,
            "retries": config.llm.retries,
            "retry_delay": config.llm.retry_delay,
            "system_message": config.llm.system_message,
        }
        config.llm.update_model_params(shared, overwrite=False)
        # Force api_base from CLI (overwrite any config default)
        config.llm.update_model_params({"api_base": api_base}, overwrite=True)
        # Reset evaluator_models to match
        config.llm.evaluator_models = config.llm.models.copy()
        config.llm.update_model_params({"api_base": api_base}, overwrite=True)
    else:
        # Using config YAML as-is. Re-apply per-model temperature overrides
        # that were lost during Config.from_yaml() -- OpenEvolve's __post_init__
        # calls update_model_params(overwrite=False), which overwrites explicit
        # null values with the global temperature.
        with open(args.config) as f:
            raw = yaml.safe_load(f)
        for i, model_raw in enumerate(raw.get("llm", {}).get("models", [])):
            if i < len(config.llm.models) and "temperature" in model_raw:
                config.llm.models[i].temperature = model_raw["temperature"]
                if i < len(config.llm.evaluator_models):
                    config.llm.evaluator_models[i].temperature = model_raw["temperature"]

    # Handle temperature: some models (e.g. GPT-5.1 Codex) reject it
    if args.no_temperature:
        config.llm.temperature = None
        config.llm.update_model_params({"temperature": None}, overwrite=True)
    elif args.temperature is not None:
        config.llm.temperature = args.temperature
        config.llm.update_model_params(
            {"temperature": args.temperature}, overwrite=True
        )

    return config


# ---------------------------------------------------------------------------
# W&B sync: background thread reads JSONL written by evaluator subprocesses
# ---------------------------------------------------------------------------

class WandbSyncer:
    """Tails the metrics JSONL file and logs each new line to W&B.

    The evaluator writes one JSON record per stage-2 evaluation from
    subprocess workers (where wandb.run is None). This thread runs in
    the main process and streams those records to W&B with proper step
    indices, plus running-best tracking.
    """

    def __init__(self, metrics_file: str, poll_interval: float = 5.0):
        self.metrics_file = metrics_file
        self.poll_interval = poll_interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._byte_offset = 0
        self._step = 0
        self._best_fom = 0.0

    def start(self):
        # Truncate any stale metrics from a previous run
        Path(self.metrics_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_file, "w"):
            pass
        self._byte_offset = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    def _run(self):
        while not self._stop.is_set():
            self._drain()
            self._stop.wait(self.poll_interval)
        # Final drain to catch any records written during shutdown
        self._drain()

    def _drain(self):
        import wandb

        try:
            with open(self.metrics_file, "r") as f:
                f.seek(self._byte_offset)
                new_data = f.read()
                self._byte_offset = f.tell()
        except FileNotFoundError:
            return

        if not new_data:
            return

        for line in new_data.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            self._step += 1
            fom = record.get("best_fom", 0)
            self._best_fom = max(self._best_fom, fom)

            wandb.log({
                # Per-evaluation metrics
                "eval/best_fom": fom,
                "eval/mean_fom": record.get("mean_fom", 0),
                "eval/num_valid_codes": record.get("num_valid", 0),
                "eval/num_high_k_codes": record.get("num_high_k", 0),
                "eval/lattices_with_high_k": record.get("lattices_with_high_k", 0),
                "eval/best_encoding_rate": record.get("best_encoding_rate", 0),
                "eval/codes_above_fom6": record.get("num_above_6", 0),
                "eval/codes_above_fom12": record.get("num_above_12", 0),
                "eval/total_candidates": record.get("total_candidates", 0),
                # Running best across all evaluations
                "progress/running_best_fom": self._best_fom,
            }, step=self._step)


# ---------------------------------------------------------------------------
# Evolution runners
# ---------------------------------------------------------------------------

def _run_fresh(config, output_dir: str, iterations: int,
               seed: str = SEED_SOLUTION, evaluator: str = EVALUATOR):
    """Run a fresh evolution using the high-level API."""
    from openevolve import run_evolution

    return run_evolution(
        initial_program=seed,
        evaluator=evaluator,
        config=config,
        iterations=iterations,
        output_dir=output_dir,
        cleanup=False,
    )


def _run_resume(config, output_dir: str, iterations: int, checkpoint_path: str,
                seed: str = SEED_SOLUTION, evaluator: str = EVALUATOR):
    """Resume evolution from a checkpoint using the controller directly.

    The high-level run_evolution() API doesn't expose checkpoint_path,
    so we instantiate the OpenEvolve controller ourselves.
    """
    from openevolve.controller import OpenEvolve

    os.makedirs(output_dir, exist_ok=True)
    controller = OpenEvolve(
        initial_program_path=seed,
        evaluation_file=evaluator,
        config=config,
        output_dir=output_dir,
    )
    best_program = asyncio.run(
        controller.run(iterations=iterations, checkpoint_path=checkpoint_path)
    )
    return best_program


def main():
    parser = argparse.ArgumentParser(
        description="Run OpenEvolve evolutionary search for BB codes."
    )
    # --- Model selection ---
    parser.add_argument(
        "--model", type=str, default=None,
        help="Single LiteLLM model identifier (e.g. anthropic/claude-sonnet-4-5-20250514). "
             "Use --models for ensemble.",
    )
    parser.add_argument(
        "--models", type=str, nargs="+", default=None,
        help="Multiple LiteLLM model identifiers for ensemble evolution "
             "(e.g. --models gemini/gemini-2.5-flash anthropic/claude-sonnet-4-5-20250514). "
             "Takes precedence over --model.",
    )
    # --- Run management ---
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Run name (output goes to results/evolution/<run-name>/). "
             "Default: auto-generated as run_YYYYMMDD_HHMMSS.",
    )
    parser.add_argument(
        "--resume", type=str, default=None, metavar="CHECKPOINT_PATH",
        help="Resume from a checkpoint directory "
             "(e.g. results/evolution/run_xyz/checkpoints/checkpoint_100).",
    )
    # --- Standard options ---
    parser.add_argument(
        "--iterations", type=int, default=100,
        help="Number of evolutionary iterations.",
    )
    parser.add_argument(
        "--api-base", type=str, default=None,
        help="LiteLLM proxy base URL (default: $OPENAI_API_BASE or "
             "$LITELLM_API_BASE or http://localhost:4000/v1).",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to OpenEvolve config YAML (default: config.yaml, "
             "or config_noncss.yaml when --noncss is set).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Explicit output directory (overrides --run-name).",
    )
    parser.add_argument(
        "--wandb", action="store_true",
        help="Enable Weights & Biases tracking.",
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="LLM temperature (default: from config). Use 0 or omit for models "
             "that don't support it (e.g. GPT-5.1 Codex).",
    )
    parser.add_argument(
        "--no-temperature", action="store_true",
        help="Disable sending temperature parameter (for models that reject it).",
    )
    parser.add_argument(
        "--wandb-project", type=str, default="qcode-discovery",
        help="W&B project name.",
    )
    # --- MILP evolution (Campaign 4+) ---
    parser.add_argument(
        "--milp", action="store_true",
        help="Use MILP exact distance instead of BP-OSD. Requires "
             "evaluate_stage2_milp in the evaluator.",
    )
    # --- Non-CSS PBB evolution ---
    parser.add_argument(
        "--noncss", action="store_true",
        help="Evolve non-CSS PBB codes instead of CSS BB codes. "
             "Uses seed_solution_noncss.py, openevolve_evaluator_noncss.py, "
             "and config_noncss.yaml by default.",
    )
    parser.add_argument(
        "--seed", type=str, default=None,
        help="Path to seed solution file (default: evolve/seed_solution.py, "
             "or evolve/seed_solution_milp.py when --milp is set, "
             "or evolve/seed_solution_noncss.py when --noncss is set).",
    )
    args = parser.parse_args()

    # Resolve model list (None means "use config as-is")
    if args.models:
        model_names = args.models
    elif args.model:
        model_names = [args.model]
    else:
        model_names = None  # use whatever is in the config YAML

    # Resolve config path
    if args.config is None:
        args.config = DEFAULT_CONFIG_NONCSS if args.noncss else DEFAULT_CONFIG

    api_base = _resolve_api_base(args)
    output_dir = _resolve_output_dir(args)

    # Propagate run name to evaluator subprocesses via environment variable.
    # OpenEvolve calls evaluate_stage2(program_path) with no way to pass
    # extra args.  _log_code_jsonl reads QCODE_RUN_NAME to route JSONL
    # to the correct run directory.
    run_name = Path(output_dir).name
    os.environ["QCODE_RUN_NAME"] = run_name

    # Per-model attribution: tag each evolved program with the model that
    # produced it (worker side, via the evaluator import) and log every accepted
    # program to <output_dir>/model_attribution.jsonl (main side, here).
    try:
        from evolve.model_attribution import install as _install_attribution
        os.makedirs(output_dir, exist_ok=True)
        _install_attribution(str(Path(output_dir) / "model_attribution.jsonl"))
    except Exception as exc:
        print(f"Warning: model attribution not installed: {exc}")

    # Resolve seed solution
    if args.seed:
        seed_path = args.seed
    elif args.noncss:
        seed_path = SEED_SOLUTION_NONCSS
    elif args.milp:
        seed_path = SEED_SOLUTION_MILP
    else:
        seed_path = SEED_SOLUTION
    if not Path(seed_path).exists():
        print(f"Error: seed solution not found: {seed_path}")
        sys.exit(1)

    # When --noncss is set, use the non-CSS evaluator directly (no patching needed).
    if args.noncss:
        EVALUATOR_ACTIVE = EVALUATOR_NONCSS
        if not Path(EVALUATOR_ACTIVE).exists():
            print(f"Error: non-CSS evaluator not found: {EVALUATOR_ACTIVE}")
            sys.exit(1)
        print(f"Non-CSS mode: using openevolve_evaluator_noncss.py")
    # When --milp is set, monkey-patch the evaluator to use MILP stage 2.
    # OpenEvolve calls evaluate_stage2() by name from the evaluator module,
    # so we replace it at module level after import.
    elif args.milp:
        import importlib.util
        spec = importlib.util.spec_from_file_location("oe_evaluator", EVALUATOR)
        _ev_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_ev_mod)
        if hasattr(_ev_mod, "evaluate_stage2_milp"):
            _ev_mod.evaluate_stage2 = _ev_mod.evaluate_stage2_milp
            _ev_mod.evaluate = _ev_mod.evaluate_stage2_milp
            # Write a patched copy the evaluator will use
            _milp_eval_path = str(Path(output_dir) / "_evaluator_milp.py")
            os.makedirs(output_dir, exist_ok=True)
            eval_source = Path(EVALUATOR).read_text()
            # Append the alias at the end of the file
            patched = eval_source + (
                "\n\n# --- MILP mode: override stage 2 ---\n"
                "evaluate_stage2 = evaluate_stage2_milp\n"
                "evaluate = evaluate_stage2_milp\n"
            )
            Path(_milp_eval_path).write_text(patched)
            # Use the patched evaluator
            EVALUATOR_ACTIVE = _milp_eval_path
            print(f"MILP mode: using evaluate_stage2_milp (patched evaluator)")
        else:
            print("Warning: evaluate_stage2_milp not found in evaluator, "
                  "falling back to BP-OSD")
            EVALUATOR_ACTIVE = EVALUATOR
    else:
        EVALUATOR_ACTIVE = EVALUATOR

    # Validate resume path
    if args.resume and not Path(args.resume).exists():
        print(f"Error: checkpoint path does not exist: {args.resume}")
        sys.exit(1)

    # Start W&B if requested
    wandb_syncer: WandbSyncer | None = None
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                config={
                    "models": model_names or "config-default",
                    "iterations": args.iterations,
                    "config": args.config,
                    "api_base": api_base,
                    "resume": args.resume,
                    "output_dir": output_dir,
                },
            )
            print(f"W&B run: {wandb.run.url}")
            # Start background sync from JSONL → W&B
            wandb_syncer = WandbSyncer(METRICS_FILE)
            wandb_syncer.start()
        except ImportError:
            print("wandb not installed. Run: uv sync --group tracking")
            sys.exit(1)

    # Run OpenEvolve
    try:
        config = _build_config(args, api_base, model_names)

        # Startup banner
        active_models = [m.name for m in config.llm.models]
        print(f"\nStarting evolution:")
        if len(active_models) == 1:
            print(f"  Model: {active_models[0]}")
        else:
            print(f"  Ensemble ({len(active_models)} models):")
            for name in active_models:
                print(f"    - {name}")
        print(f"  Iterations: {args.iterations}")
        print(f"  API base: {api_base}")
        print(f"  Seed: {seed_path}")
        if args.noncss:
            print(f"  Mode: Non-CSS PBB codes")
            print(f"  Distance: BP-OSD multi-channel (non-CSS)")
        elif args.milp:
            print(f"  Distance: MILP (exact or upper bound)")
        else:
            print(f"  Distance: BP-OSD (estimate)")
        print(f"  Output: {output_dir}")
        if args.resume:
            print(f"  Resuming from: {args.resume}")
        print()

        if args.resume:
            best_program = _run_resume(
                config, output_dir, args.iterations, args.resume,
                seed=seed_path, evaluator=EVALUATOR_ACTIVE,
            )
            print(f"\nEvolution complete!")
            if best_program:
                score = (best_program.metrics or {}).get("combined_score", 0)
                print(f"  Best score: {score:.4f}")
                # Save best program
                best_path = Path(output_dir) / "best_generate_candidates.py"
                best_path.parent.mkdir(parents=True, exist_ok=True)
                best_path.write_text(best_program.code)
                print(f"  Best program: {best_path}")
            print(f"  Output: {output_dir}")
        else:
            result = _run_fresh(config, output_dir, args.iterations,
                                seed=seed_path, evaluator=EVALUATOR_ACTIVE)

            print(f"\nEvolution complete!")
            print(f"  Best score: {result.best_score:.4f}")
            print(f"  Output: {result.output_dir}")

            # Save best program
            if result.best_code:
                best_path = Path(output_dir) / "best_generate_candidates.py"
                best_path.parent.mkdir(parents=True, exist_ok=True)
                best_path.write_text(result.best_code)
                print(f"  Best program: {best_path}")

                if args.wandb:
                    try:
                        import wandb
                        artifact = wandb.Artifact("best-program", type="code")
                        artifact.add_file(str(best_path))
                        wandb.log_artifact(artifact)
                    except Exception:
                        pass

    except ImportError:
        print("openevolve not installed. Run: uv sync --group evolve")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nEvolution interrupted by user.")
    finally:
        if wandb_syncer:
            wandb_syncer.stop()
        if args.wandb:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass


if __name__ == "__main__":
    main()
