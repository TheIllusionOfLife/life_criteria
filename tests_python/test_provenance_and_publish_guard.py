from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from scripts import check_manuscript_consistency as cmc
from scripts import upload_zenodo


def test_upload_zenodo_requires_confirm_publish(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        upload_zenodo,
        "parse_args",
        lambda: Namespace(
            new_version=None,
            edit=None,
            fetch_bibtex=None,
            metadata=Path("zenodo_metadata.json"),
            title=None,
            description=None,
            creator=[],
            version=None,
            keyword=[],
            github_url=None,
            conference_title=None,
            conference_url=None,
            language=None,
            publish=True,
            confirm_publish=False,
            sandbox=False,
            no_verify_checksums=False,
        ),
    )

    rc = upload_zenodo.main()
    out = capsys.readouterr()
    assert rc == 1
    assert "requires --confirm-publish" in out.err


def test_upload_zenodo_confirm_publish_still_checks_token(monkeypatch, capsys) -> None:
    monkeypatch.delenv("ZENODO_TOKEN", raising=False)
    monkeypatch.setattr(
        upload_zenodo,
        "parse_args",
        lambda: Namespace(
            new_version=None,
            edit=None,
            fetch_bibtex=None,
            metadata=Path("zenodo_metadata.json"),
            title=None,
            description=None,
            creator=[],
            version=None,
            keyword=[],
            github_url=None,
            conference_title=None,
            conference_url=None,
            language=None,
            publish=True,
            confirm_publish=True,
            sandbox=False,
            no_verify_checksums=False,
        ),
    )

    rc = upload_zenodo.main()
    out = capsys.readouterr()
    assert rc == 1
    assert "ZENODO_TOKEN not set" in out.err


def test_check_manuscript_consistency_run_checks_happy_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cmc, "EXPERIMENT_SCRIPTS", [])
    paper = tmp_path / "main.tex"
    paper.write_text(
        r"""
        This system runs for 10000 timesteps with population sampled every 100.
        \label{tab:params}
        \label{tab:results}
        """
    )

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "steps": 10000,
                "sample_every": 100,
                "base_config": {
                    "mutation_point_rate": 0.02,
                    "mutation_scale": 0.15,
                },
                "source_git_commit": "abc123",
                "source_generated_at_utc": "2026-03-04T00:00:00+00:00",
            }
        )
    )

    bindings = tmp_path / "bindings.json"
    bindings.write_text(
        json.dumps(
            {
                "bindings": [
                    {"paper_ref": "tab:params", "manifest": "experiments/a.json"},
                    {"paper_ref": "tab:results", "manifest": "experiments/b.json"},
                ]
            }
        )
    )

    report = cmc.run_checks(paper, manifest, bindings)
    assert report["ok"] is True
    assert not report["issues"]
