from pathlib import Path
import pandas as pd

<<<<<<< HEAD
def load_behaviors(data_root: str | Path, split: str) -> pd.DataFrame:
    data_root = Path(data_root)
    return pd.read_parquet(data_root / split / "behaviors.parquet")

def load_history(data_root: str | Path, split: str = "train") -> pd.DataFrame:
    data_root = Path(data_root)
    return pd.read_parquet(data_root / split / "history.parquet")

def load_articles(data_root: str | Path) -> pd.DataFrame:
    data_root = Path(data_root)
    return pd.read_parquet(data_root / "articles.parquet")

=======

def load_behaviors(data_root: Path, split: str) -> pd.DataFrame:
    return pd.read_parquet(data_root / split / "behaviors.parquet")


def load_articles(data_root: Path) -> pd.DataFrame:
    return pd.read_parquet(data_root / "articles.parquet")


def load_history(data_root, split):
    path = data_root / split / "history.parquet"
    return pd.read_parquet(path)
>>>>>>> main
