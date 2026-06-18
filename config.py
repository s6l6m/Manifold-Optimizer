import os

from dataclasses import dataclass
from pathlib import Path


@dataclass()
class Config:
    """Configuration class for the experiment."""
    DB_NAME: str = "skfp_atompair_fp4096_folded_dense_count"
    DATABASE_DESCRIPTION: str = "Database for MS2Query project"

    COLLECTION_NAME: str = "molecular_similarity_search"
    COLLECTION_DESCRIPTION: str = "Collection for molecular similarity search"

    RESET_DATABASE: bool = False
    RESET_DATA: bool = False

    MILVUS_PROTOCOL: str = "tcp"
    MILVUS_HOST: str = "172.18.0.4"
    MILVUS_PORT: int = 19530
    MILVUS_USER: str = "root"
    MILVUS_PASSWORD: str = "Milvus"
    MILVUS_TIMEOUT: float = 5.0

    @property
    def MILVUS_URI(self) -> str:
        return f"{self.MILVUS_PROTOCOL}://{self.MILVUS_HOST}:{self.MILVUS_PORT}"

    @property
    def MILVUS_TOKEN(self) -> str:
        return f"{self.MILVUS_USER}:{self.MILVUS_PASSWORD}"

    BINARY_EMBEDDING: bool = False

    RANDOM_SEED: int = 42

    SUBSET_SIZE: int = 1_000

    DASHBOARD_PROTOCOL: str = "http"
    DASHBOARD_HOST: str = "0.0.0.0"
    DASHBOARD_DOMAIN: str = "localhost"
    DASHBOARD_PORT: int = 8080

    @property
    def OPTUNA_SQLITE(self) -> str:
        return f"sqlite:///{self.DATA_DIR_PATH / 'optuna_study.sqlite3'}"

    @property
    def DATA_DIR_PATH(self) -> Path:
        data_dir = Path(os.getcwd()) / "data"
        if not data_dir.exists():
            os.makedirs(data_dir)
        return data_dir
