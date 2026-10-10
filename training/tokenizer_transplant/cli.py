"""CLI for tokenizer transplant.

    uv run python -m tokenizer_transplant full     --config configs/olmo3_7b__deepseek_v4_flash.yaml
    uv run python -m tokenizer_transplant selftest --config configs/olmo3_7b__deepseek_v4_flash.yaml
"""

from __future__ import annotations

import typer

from .selftest import selftest as run_selftest
from .transplant import TransplantConfig, run

app = typer.Typer(add_completion=False, help="OMP tokenizer-embedding transplant.")


@app.command()
def full(
    config: str = typer.Option(..., help="Path to a transplant YAML config."),
    device: str = typer.Option(None, help="cuda / cpu (default: auto-detect)."),
    k: int = typer.Option(None, help="Override OMP sparsity k."),
):
    """Build and save a transplanted model."""
    cfg = TransplantConfig.from_yaml(config)
    if k is not None:
        cfg.k = k
    run(cfg, device=device)


@app.command()
def selftest(
    config: str = typer.Option(..., help="Path to a transplant YAML config."),
    device: str = typer.Option(None, help="cuda / cpu (default: auto-detect)."),
    which: str = typer.Option("embed", help="embed or lm_head."),
    hold: int = typer.Option(500, help="Number of held-out anchors."),
):
    """Held-out reconstruction fidelity check (needs the real weights)."""
    cfg = TransplantConfig.from_yaml(config)
    run_selftest(cfg, device=device, hold=hold, which=which)


if __name__ == "__main__":
    app()
