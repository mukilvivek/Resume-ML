from data_loader import DataLoader
from trainer import HybridResumeTrainer
import os

def main():
    # Initialize data loader
    data_loader = DataLoader(dataset_path='data/resumes/synthetic_resumes.csv')
    
    # Load and split data
    print("Loading data...")
    data = data_loader.load_data()
    X_train, X_test, y_train, y_test = data_loader.split_data(data)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Define job description
    job_description = """
    Looking for a Software Engineer with strong experience in Python, machine learning, 
    and web development. The ideal candidate should have experience with REST APIs, 
    data processing, and cloud technologies. Must be proficient in Git and have 
    demonstrated experience building scalable applications.
    """
    
    # Initialize trainer
    trainer = HybridResumeTrainer()
    
    # Train the model
    print("\nTraining model...")
    trainer.train(X_train, y_train, job_description)
    
    # Evaluate the model
    print("\nEvaluating model...")
    mse, predictions = trainer.evaluate(X_test, y_test, job_description)
    
    # Save results
    print("\nSaving results...")
    os.makedirs('results', exist_ok=True)
    results = X_test.copy()
    results['actual_score'] = y_test
    results['predicted_score'] = predictions
    results.to_csv('results/evaluation_results.csv', index=False)
    
    # Save the model
    print("\nSaving model...")
    os.makedirs('models', exist_ok=True)
    trainer.save_model('models/resume_scorer.pkl')
    
    print("\nDone! Model and results have been saved.")

if __name__ == "__main__":
    main()