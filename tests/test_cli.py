"""Public command routing must work without loading model dependencies."""

import sys
from pathlib import Path

import pytest

from denmark import __main__ as cli


def commands(tree, prefix=()):
    for name, target in tree.items():
        path = (*prefix, name)
        if isinstance(target, dict):
            yield from commands(target, path)
        else:
            yield path, target


@pytest.mark.parametrize("path,target", list(commands(cli.COMMANDS)))
def test_routes_forward_arguments(monkeypatch, path, target):
    calls = []
    monkeypatch.setattr(sys, "argv", ["denmark", *path, "--output", "some path", "--help"])
    monkeypatch.setattr(cli.runpy, "run_module", lambda module, **kw: calls.append(
        (module, kw, sys.argv[1:])
    ))
    cli.main()
    assert calls == [(target, {"run_name": "__main__"}, ["--output", "some path", "--help"])]
    root = Path(__file__).resolve().parents[1]
    assert root.joinpath(*target.split(".")).with_suffix(".py").is_file()


def test_nested_help_does_not_load_models(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["denmark", "baseline"])
    monkeypatch.setattr(cli.runpy, "run_module", lambda *a, **kw: pytest.fail("unexpected dispatch"))
    cli.main()
    assert "dlm_kgw" in capsys.readouterr().out


def test_invalid_command(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["denmark", "nonexistent"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


@pytest.mark.parametrize("package", ["denmark.core", "denmark.baselines.clean"])
@pytest.mark.parametrize("family_args", [["--generator_family", "dream"], ["--generator_family=dream"]])
def test_unified_dream_dispatch(monkeypatch, package, family_args):
    import importlib

    generate = importlib.import_module(package + ".generate")
    model = importlib.import_module(package + ".model")
    calls = []
    monkeypatch.setattr(sys, "argv", ["generate", *family_args, "--help"])
    monkeypatch.setattr(model, "generate_dream", lambda: calls.append(sys.argv[1:]))
    generate.main()
    assert calls == [[*family_args, "--help"]]
