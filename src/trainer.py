from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd
from typing import List
import torch

class ResumeTrainer:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.threshold = 0.7
    
    def prepare_training_data(self, 
                            train_df: pd.DataFrame, 
                            job_description: str) -> List[InputExample]:
        """Prepare training examples using Resume_str"""
        train_examples = []
        
        for _, row in train_df.iterrows():
            # Create training pair with resume text and job description
            train_examples.append(
                InputExample(
                    texts=[row['Resume_str'], job_description],
                    # For now, use 1.0 for matching categories, 0.0 for non-matching
                    label=float(row['Category'] == 'Software Engineer')
                )
            )
        
        return train_examples
    
    def train(self, 
              train_examples: List[InputExample],
              batch_size: int = 16,
              epochs: int = 4,
              evaluation_steps: int = 50):
        """Train the model"""
        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=batch_size
        )
        
        # Use cosine similarity loss
        train_loss = losses.CosineSimilarityLoss(self.model)
        
        # Train the model
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            evaluation_steps=evaluation_steps,
            show_progress_bar=True
        )
    
    def score_resume(self, resume_text: str, job_description: str) -> float:
        """Score a resume against a job description"""
        # Generate embeddings
        resume_embedding = self.model.encode(resume_text, convert_to_tensor=True)
        job_embedding = self.model.encode(job_description, convert_to_tensor=True)
        
        # Calculate cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(resume_embedding, job_embedding, dim=0)
        
        return float(cos_sim)
    
    def save_model(self, output_path: str):
        """Save the trained model"""
        self.model.save(output_path)