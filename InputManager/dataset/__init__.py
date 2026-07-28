from typing import Any

from .iran5_preprocessing import Iran5Preprocessing
from .mh_preprocessing import MhPreprocessing
from .moltbook_preprocessing import MoltbookPreprocessing
from .russia1_preprocessing import Russia1Preprocessing
from .russia2_preprocessing import Russia2Preprocessing
from .uk2019_preprocessing import Uk2019Preprocessing
from .venezuela2_preprocessing import Venezuela2Preprocessing
from .base_preprocessing import DatasetPreprocessing


DATASET_PREPROCESSORS = {
    "iran5": Iran5Preprocessing,
    "mh": MhPreprocessing,
    "moltbook": MoltbookPreprocessing,
    "russia1": Russia1Preprocessing,
    "russia2": Russia2Preprocessing,
    "uk2019": Uk2019Preprocessing,
    "venezuela2": Venezuela2Preprocessing,
}


def build_dataset_preprocessor(
    dataset_name: str,
    checkpoint: Any,
    directory_manager: Any,
    logger: Any,
    file_name: str
) -> DatasetPreprocessing:
    """
    Create the dataset-specific preprocessor registered for a dataset name.

    :param dataset_name: Dataset key used to select the preprocessor class.
    :param checkpoint: Checkpoint instance used to read and save artifacts.
    :param directory_manager: DirectoryManager instance with dataset paths.
    :param logger: LogManager instance used for progress logging.
    :param file_name: Name of the caller module used in logs.
    :return: DatasetPreprocessing instance for the requested dataset.
    """
    preprocessor_class = DATASET_PREPROCESSORS.get(dataset_name)

    if preprocessor_class is None:
        available_datasets = ", ".join(sorted(DATASET_PREPROCESSORS))
        raise ValueError(
            f"No data preprocessor configured for dataset '{dataset_name}'. "
            f"Available datasets: {available_datasets}."
        )

    return preprocessor_class(checkpoint, directory_manager, logger, file_name)
