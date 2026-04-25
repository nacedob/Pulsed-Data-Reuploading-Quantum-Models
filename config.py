import os

DATASETS = ['iris', 'digits_17', 'digits_56']

def get_root_path(start_path: str | None = None) -> str:
    path = os.path.abspath(start_path or os.getcwd())

    while True:
        git_path = os.path.join(path, ".git")
        if os.path.exists(git_path):
            return path

        parent = os.path.dirname(path)

        if parent == path:
            break

        path = parent

    raise FileNotFoundError("Could not find .git directory")


