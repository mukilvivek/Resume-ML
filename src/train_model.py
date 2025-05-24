from data_loader import DataLoader
from trainer import ResumeTrainer
import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

def evaluate_model(trainer: ResumeTrainer, test_data: pd.DataFrame, job_description: str, threshold: float = 0.7):
    """Evaluate model performance on test set"""
    predictions = []
    scores = []
    
    for _, row in test_data.iterrows():
        score = trainer.score_resume(row['Resume_str'], job_description)
        prediction = 1 if score >= threshold else 0
        predictions.append(prediction)
        scores.append(score)
    
    # Add predictions and scores to test data
    test_data['predicted'] = predictions
    test_data['score'] = scores
    
    # Calculate and print metrics
    true_labels = [1 if cat == 'Software Engineer' else 0 for cat in test_data['Category']]
    print("\nModel Evaluation:")
    print(classification_report(true_labels, predictions))
    
    return test_data

def main():
    # Initialize data loader
    data_loader = DataLoader(
        resume_path='data/resumes/Resume.csv',
        job_desc_path='data/jobs/software_engineer.txt'
    )
    
    # Load and split data
    print("Loading data...")
    resumes, job_description = data_loader.load_data()
    split_data = data_loader.split_data(resumes)
    
    print(f"Total resumes: {len(resumes)}")
    print(f"Training set: {len(split_data['train'])}")
    print(f"Test set: {len(split_data['test'])}")
    
    # Initialize trainer
    trainer = ResumeTrainer()
    
    # Prepare training data
    print("\nPreparing training data...")
    train_examples = trainer.prepare_training_data(
        split_data['train'],
        job_description
    )
    
    # Train the model
    print("\nTraining model...")
    trainer.train(train_examples)
    
    # Evaluate model
    print("\nEvaluating model...")
    results = evaluate_model(trainer, split_data['test'], job_description)
    
    # Save results
    print("\nSaving results...")
    os.makedirs('results', exist_ok=True)
    results.to_csv('results/evaluation_results.csv', index=False)
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    trainer.save_model('models/resume_scorer')
    
    print("\nDone! Model and results have been saved.")

if __name__ == "__main__":
    main()