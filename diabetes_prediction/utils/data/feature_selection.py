"""Feature Selection module
"""

from diabetes_prediction._utils import *
from diabetes_prediction.utils.data.metrics import *

from abc import ABCMeta, abstractmethod


class FeatureSelector(metaclass=ABCMeta):
    def __init__(self, metadata: pd.DataFrame, data: pd.DataFrame) -> None:
        """Feature selector.

        Args:
            metadata: Metadata.
            data: Data records.
        """
        self.metadata = metadata
        self.data     = data

    @abstractmethod
    def select(self, threshold: float) -> pd.Index:
        """Select features.
        Args:
            threshold: Threshold for correlation.

        Returns:
            Selected features.
        """
        raise NotImplementedError

class CorrelationYFeatureSelector(FeatureSelector):
    def __init__(self, metadata: pd.DataFrame, data: pd.DataFrame) -> None:
        """Correlation feature selector.

        Args:
            metadata: Metadata.
            data: Data records.
        """
        super().__init__(metadata, data)
        self.corr       = get_corr(metadata, data)
        self.thresholds = (self.corr['corr_abs'] / self.corr['corr_abs'].sum()).cumsum()

    def select(self, threshold: float) -> pd.Index:
        """Select features.
        Args:
            threshold: Threshold for correlation.

        Returns:
            Selected features.
        """
        return self.corr.index[self.thresholds <= threshold]


class CorrelationXFeatureSelector(FeatureSelector):
    def __init__(self, metadata: pd.DataFrame, data: pd.DataFrame) -> None:
        """Correlation feature selector.

        Args:
            metadata: Metadata.
            data: Data records.
        """
        super().__init__(metadata, data)
        self.corr = data.select_dtypes('number').drop(columns=PARAMS.target).corr()
        self.corr.values[np.triu_indices_from(self.corr.values)] = None
        np.fill_diagonal(self.corr.values, 0)
        self.thresholds = self.corr.abs().max().sort_values(ascending=False)

    def select(self, threshold: float) -> pd.Index:
        """Select features.
        Args:
            threshold: Threshold for correlation.

        Returns:
            Selected features.
        """
        return self.thresholds.index[self.thresholds <= threshold]


class CorrelationXYFeatureSelector(FeatureSelector):
    def __init__(self, metadata: pd.DataFrame, data: pd.DataFrame) -> None:
        """Correlation feature selector.

        Args:
            metadata: Metadata.
            data: Data records.
        """
        super().__init__(metadata, data)
        self.corr_target = get_corr(metadata, data)['corr_abs']
        self.corr = data.select_dtypes('number').drop(columns=PARAMS.target).corr().abs()
        self.corr.values[np.triu_indices_from(self.corr.values)] = 0
        self.sorted_indices = np.array(np.unravel_index(np.argsort(self.corr.values.ravel()), self.corr.shape)).T[::-1]

    def select(self, threshold: float) -> pd.Index:
        """Select features.
        Args:
            threshold: Threshold for correlation.

        Returns:
            Selected features.
        """
        features_flag = np.ones_like(self.corr.columns, dtype=bool)
        cols = self.corr.columns
        for row, col in self.sorted_indices:
            f_row, f_col, c = cols[row], cols[col], self.corr.iloc[row, col]
            if c <= threshold:
                break
            corr_target_row = self.corr_target[f_row]
            corr_target_col = self.corr_target[f_col]
            if corr_target_row >= corr_target_col:  # select smaller one
                if features_flag[row]:
                    features_flag[col] = False
            else:
                if features_flag[col]:
                    features_flag[row] = False
        selected_features = cols[features_flag]
        return self.corr_target[self.corr_target.index.isin(selected_features)].index  # sort by corr_target
