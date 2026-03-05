from pathlib import Path
import pandas as pd

def load_behaviors(data_root: Path, split: str) -> pd.DataFrame:
    return pd.read_parquet(data_root / split / "behaviors.parquet")

def load_articles(data_root: Path) -> pd.DataFrame:
    return pd.read_parquet(data_root / "articles.parquet")