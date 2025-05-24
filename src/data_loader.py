import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict

class DataLoader:
    def __init__(self, resume_path: str, job_desc_path: str):
        self.resume_path = resume_path
        self.job_desc_path = job_desc_path
    
    def load_data(self) -> Tuple[pd.DataFrame, str]:
        """Load resume data and job description"""
        # Read CSV with the correct columns
        resumes = pd.read_csv(self.resume_path)
        
        # Verify required columns exist
        required_columns = ['ID', 'Resume_str', 'Resume_html', 'Category']
        if not all(col in resumes.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
        
        # Read job description
        with open(self.job_desc_path, 'r') as f:
            job_description = f.read()
            
        return resumes, job_description
    
    def split_data(self, data: pd.DataFrame, test_size: float = 0.2) -> Dict[str, pd.DataFrame]:
        """Split data into train and test sets"""
        train_data, test_data = train_test_split(
            data, 
            test_size=test_size, 
            random_state=42,
            stratify=data['Category'] if len(data['Category'].unique()) > 1 else None
        )
        return {
            'train': train_data,
            'test': test_data
        }