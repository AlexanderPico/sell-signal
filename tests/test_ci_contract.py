from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / '.github' / 'workflows' / 'ci.yml'
README_PATH = REPO_ROOT / 'README.md'


def test_ci_workflow_exists_and_runs_documented_checks() -> None:
    workflow = WORKFLOW_PATH.read_text()
    assert 'pytest -q' in workflow
    assert 'ruff check src tests' in workflow
    assert "python -m pip install -e '.[dev]'" in workflow
    assert 'pull_request:' in workflow


def test_readme_documents_ci_validation_commands() -> None:
    readme = README_PATH.read_text()
    assert 'pytest -q' in readme
    assert 'ruff check src tests' in readme
