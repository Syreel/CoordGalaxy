import pandas as pd

from .information_operation_preprocessing import InformationOperationPreprocessing


class Russia2Preprocessing(InformationOperationPreprocessing):
    """Preprocess the Russia2 Information Operations dataset into the framework schema."""

    def normalize_data(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """
        Normalize Russia2 post data and save it under temp_data.

        Russia2 uses the shared Information Operations schema: post/account ids,
        reply and repost account ids, URL/hashtag/mention list columns, and the
        optional is_control label.

        :param df: Raw Russia2 post dataframe.
        :param filename: Name of the normalized CSV file to save under temp_data.
        :return: Normalized Russia2 dataframe.
        """
        return super().normalize_data(df, filename)
