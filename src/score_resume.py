from trainer import HybridResumeTrainer
import pickle
import pandas as pd

def load_model(model_path: str = 'models/resume_scorer.pkl'):
    """Load the trained model"""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data

def score_resume(education: str, experience: str, projects: str, skills: str, 
                model_data: dict, job_description: str) -> float:
    """Score a single resume"""
    # Create a dataframe with the resume data
    resume_data = pd.DataFrame({
        'education': [education],
        'experience': [experience],
        'projects': [projects],
        'skills': [skills]
    })
    
    # Prepare features using the loaded model components
    trainer = HybridResumeTrainer()
    trainer.bert_model = model_data['bert_model']
    trainer.education_encoder = model_data['education_encoder']
    trainer.tfidf = model_data['tfidf']
    trainer.rf_model = model_data['rf_model']
    trainer.education_categories = model_data['education_categories']
    
    # Get prediction
    features = trainer.prepare_features(resume_data, job_description)
    score = trainer.rf_model.predict(features)[0]
    
    return score

def main():
    # Load the trained model
    try:
        model_data = load_model()
    except FileNotFoundError:
        print("Error: Model file not found. Please train the model first.")
        return

    # Job description
    job_description = """
    Looking for a Software Engineer with strong experience in Python, machine learning, 
    and web development. The ideal candidate should have experience with REST APIs, 
    data processing, and cloud technologies. Must be proficient in Git and have 
    demonstrated experience building scalable applications.
    """
    
    print("\nResume Scoring System")
    print("===================")
    
    # Get resume information
    print("\nPlease enter the resume details:")
    education = input("Education (e.g., 'B.Sc. Computer Science, Stanford'): ")
    experience = input("Experience (e.g., 'Software Engineer at Google (3 yrs)'): ")
    projects = input("Projects (e.g., 'Built an ML model; Developed a REST API'): ")
    skills = input("Skills (comma-separated, e.g., 'Python, Git, AWS, Docker'): ")
    
    # Get score
    score = score_resume(education, experience, projects, skills, model_data, job_description)
    
    # Print results
    print("\nResults:")
    print(f"Predicted Score: {score:.2f} / 1.00")
    
    # Provide interpretation
    if score >= 0.8:
        print("Interpretation: Excellent match for the position")
    elif score >= 0.6:
        print("Interpretation: Good match for the position")
    elif score >= 0.4:
        print("Interpretation: Moderate match for the position")
    else:
        print("Interpretation: May not be the best match for the position")

if __name__ == "__main__":
    main()