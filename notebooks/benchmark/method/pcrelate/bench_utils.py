from pathlib import Path
from dataclasses import dataclass
from contextlib import AbstractContextManager
from typing import Optional

@dataclass
class Measurment:
    samples: int
    variants: int
    population: int
    pca_time: float
    pc_relate_time: float
    num_rows: Optional[int] = None

    def to_csv(self) -> str:
        return f"{self.samples},{self.variants},{self.population},{self.pca_time:.2f},{self.pc_relate_time:.2f},{self.num_rows}"

class Experiment(AbstractContextManager):
    def __init__(self, name: str):
        self.name = name
        self.measurment_dir = Path(__file__).resolve().parent.joinpath("measurments")
        self.measurment_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_file = self.measurment_dir.joinpath(f"write_{name}")
        if not self.experiment_file.exists():
            self.experiment_fd = self.experiment_file.open("w")
            self.experiment_fd.write("samples,variants,population,pca,pc_relate,result_num_rows\n")
            self.experiment_fd.flush()
        else:
            self.experiment_fd = self.experiment_file.open("a")

    def add_measurment(self, m: Measurment) -> "Experiment":
        self.experiment_fd.write(f"{m.to_csv()}\n")
        self.experiment_fd.flush()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.experiment_fd:
            self.experiment_fd.close()