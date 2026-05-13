"""Tests for the poster-results command cache behavior."""

from __future__ import annotations

from typing import Any

from src.commands import poster_results as pr


def _fake_trial(task: tuple[int, int, int | None]) -> tuple[int, int, dict[str, Any]]:
    n, trial, seed = task
    value = float(n + trial + (seed or 0))
    return n, trial, {
        "raw_sa": {"apsp": value, "flow": value + 1},
        "global": {
            "apsp": value + 2,
            "flow": value + 3,
            "qvars": value + 4,
            "sg": value + 5,
            "pt": value + 6,
        },
        "mr2s": {
            "apsp": value + 7,
            "flow": value + 8,
            "qvars": value + 9,
            "sg": value + 10,
            "phys_total": value + 11,
            "phys_max": value + 12,
            "phys_mean": value + 13,
            "phys_min": value + 14,
        },
        "random": {"apsp": value + 15, "flow": value + 16},
        "timings": {
            "raw_sa": 0.0,
            "global_solve": 0.0,
            "global_embed": 0.0,
            "clustered_solve": 0.0,
            "clustered_embed": 0.0,
            "random": 0.0,
        },
    }


def test_poster_trial_cache_key_is_stable() -> None:
    key = pr._poster_trial_cache_key(n=20, trial=3, seed=42)
    assert key == (
        'poster-results-trial:{"n": 20, "seed": 42, '
        '"trial": 3, "version": 1}'
    )


def test_run_reuses_poster_trial_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pr, "_plot_results", lambda results, output_dir: None)

    call_count = 0

    def counted_trial(task):
        nonlocal call_count
        call_count += 1
        return _fake_trial(task)

    monkeypatch.setattr(pr, "_run_trial", counted_trial)

    kwargs = dict(
        sizes=[8],
        num_graphs=2,
        seed=0,
        output_dir=str(tmp_path),
        num_workers=0,
    )

    pr.run(**kwargs)
    assert call_count == 2
    assert (tmp_path / "poster_results.json").exists()
    assert len(list((tmp_path / "poster_trial_cache").glob("*.pkl"))) == 2

    call_count = 0
    pr.run(**kwargs)
    assert call_count == 0


def test_run_can_disable_poster_trial_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pr, "_plot_results", lambda results, output_dir: None)
    monkeypatch.setattr(pr, "_run_trial", _fake_trial)

    pr.run(
        sizes=[8],
        num_graphs=1,
        seed=0,
        output_dir=str(tmp_path),
        num_workers=0,
        use_cache=False,
    )

    assert not (tmp_path / "poster_trial_cache").exists()
