import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict

class DataLoader:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    def load_data(self) -> pd.DataFrame:
        """Load the structured dataset"""
        data = pd.read_csv(self.dataset_path)
        return data

    def split_data(self, data: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split the data into training and testing sets"""
        X = data[['education', 'experience', 'projects', 'skills']]
        y = data['score']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test