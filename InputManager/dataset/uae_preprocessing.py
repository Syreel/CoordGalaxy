import pandas as pd

from .information_operation_preprocessing import InformationOperationPreprocessing


class UaePreprocessing(InformationOperationPreprocessing):
    """Preprocess the UaE (Middle East) Information Operations dataset into the framework schema."""

    def normalize_data(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """
        Normalize UaE post data and save it under temp_data.

        UaE uses the shared Information Operations schema: post/account ids,
        reply and repost account ids, URL/hashtag/mention list columns, and the
        optional is_control label.

        :param df: Raw UaE post dataframe.
        :param filename: Name of the normalized CSV file to save under temp_data.
        :return: Normalized UaE dataframe.
        """
        return super().normalize_data(df, filename)
