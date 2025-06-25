import os


def shelve_path(name: str) -> str:
    os.makedirs("app/data", exist_ok=True)
    return os.path.join("app/data", name)
