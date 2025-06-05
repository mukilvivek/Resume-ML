from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import torch
import pickle

class HybridResumeTrainer:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        # Initialize components
        self.bert_model = SentenceTransformer(model_name)
        self.education_encoder = LabelEncoder()
        self.tfidf = TfidfVectorizer(max_features=100)
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            random_state=42
        )
        self.education_categories = None
        
    def prepare_features(self, X: pd.DataFrame, job_description: str, fit_tfidf: bool = False):
        """Prepare both content-based and similarity features"""
        # 1. Process education
        if fit_tfidf:
            # During training, fit the encoder with categories including 'Unknown'
            self.education_categories = list(X['education'].unique()) + ['Unknown']
            education_encoded = self.education_encoder.fit_transform(self.education_categories)
            # Map actual education values
            education_idx = pd.Series(X['education']).map(
                {cat: idx for idx, cat in enumerate(self.education_categories[:-1])}
            ).fillna(len(self.education_categories) - 1)  # Use last index for Unknown
        else:
            # During prediction, map unseen categories to 'Unknown'
            education_idx = pd.Series(X['education']).map(
                {cat: idx for idx, cat in enumerate(self.education_categories[:-1])}
            ).fillna(len(self.education_categories) - 1)
        
        # 2. Process text fields with TF-IDF
        experience_text = X['experience'].fillna('')
        projects_text = X['projects'].fillna('')
        skills_text = X['skills'].fillna('')
        
        # Combine text features
        combined_text = experience_text + ' ' + projects_text + ' ' + skills_text
        if fit_tfidf:
            content_features = self.tfidf.fit_transform(combined_text).toarray()
        else:
            content_features = self.tfidf.transform(combined_text).toarray()
        
        # 3. Calculate similarity scores with job description
        similarity_scores = []
        for text in combined_text:
            resume_embedding = self.bert_model.encode(text, convert_to_tensor=True)
            job_embedding = self.bert_model.encode(job_description, convert_to_tensor=True)
            similarity = torch.nn.functional.cosine_similarity(resume_embedding, job_embedding.unsqueeze(0), dim=1)
            similarity_scores.append(float(similarity[0]))
        
        # 4. Combine all features
        features = np.column_stack((
            education_idx.values.reshape(-1, 1),
            content_features,
            np.array(similarity_scores).reshape(-1, 1)
        ))
        
        return features
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, job_description: str, fit_tfidf: bool = True):
        """Train the hybrid model"""
        print("Preparing features...")
        # Ensure education categories are captured during training
        self.education_categories = X_train['education'].unique()
        X_processed = self.prepare_features(X_train, job_description, fit_tfidf=fit_tfidf)
        
        print("Training Random Forest model...")
        self.rf_model.fit(X_processed, y_train)
        
        self._print_feature_importance()
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, job_description: str):
        """Evaluate the model"""
        X_processed = self.prepare_features(X_test, job_description)
        predictions = self.rf_model.predict(X_processed)
        mse = ((predictions - y_test) ** 2).mean()
        
        print(f"\nMean Squared Error: {mse:.4f}")
        return mse, predictions
    
    def _print_feature_importance(self):
        """Print the most important features"""
        feature_names = (
            ['education'] + 
            self.tfidf.get_feature_names_out().tolist() +
            ['job_similarity']
        )
        
        importances = self.rf_model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        
        print("\nTop 10 most important features:")
        for idx in indices:
            print(f"{feature_names[idx]}: {importances[idx]:.4f}")
    
    def save_model(self, output_path: str):
        """Save all components"""
        model_data = {
            'bert_model': self.bert_model,
            'education_encoder': self.education_encoder,
            'tfidf': self.tfidf,
            'rf_model': self.rf_model,
            'education_categories': self.education_categories
        }
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)