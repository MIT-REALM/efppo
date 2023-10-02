import pathlib


def mkdir(path: pathlib.Path) -> pathlib.Path:
    """Helper function to reduce number of lines of code."""
    path.mkdir(exist_ok=True, parents=True)
    return path


def get_root_dir():
    return pathlib.Path(__file__).parent.parent.parent.parent


def get_runs_dir():
    return mkdir(get_root_dir() / "runs")
