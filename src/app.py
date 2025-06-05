import streamlit as st
import pandas as pd
from score_resume import load_model, score_resume

# Must be the first Streamlit command
st.set_page_config(
    page_title="Resume Scorer",
    page_icon="📝",
    layout="wide"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 15px 30px;
        font-size: 18px;
        border-radius: 10px;
        border: none;
        width: 50%;
        margin: 0 auto;
        display: block;
        transition: background-color 0.3s;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextInput>div>div>input {
        background-color: white;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        border: 2px solid #e0e0e0;
        transition: border-color 0.3s;
    }
    .stTextInput>div>div>input:focus {
        border-color: #4CAF50;
        box-shadow: 0 0 5px rgba(76, 175, 80, 0.3);
    }
    .stTextArea>div>div>textarea {
        background-color: white;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        border: 2px solid #e0e0e0;
        transition: border-color 0.3s;
    }
    .stTextArea>div>div>textarea:focus {
        border-color: #4CAF50;
        box-shadow: 0 0 5px rgba(76, 175, 80, 0.3);
    }
    h1 {
        color: #2196F3;
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        border-radius: 10px;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    h2 {
        color: #333;
        padding: 15px;
        background: linear-gradient(135deg, #fff 0%, #f5f5f5 100%);
        border-radius: 10px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%);
    }
    div[data-testid="column"] {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        padding: 10px;
        margin: 5px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    div[class="stTextArea"] label {
        text-align: center;
        width: 100%;
        color: #333;
        font-weight: 500;
    }
    div[class="stTextInput"] label {
        text-align: center;
        width: 100%;
        color: #333;
        font-weight: 500;
    }
    .element-container {
        background-color: white;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .stTable {
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    # Title with emoji
    st.title("📝 Resume Scoring System")
    
    # Load the trained model
    try:
        model_data = load_model()
    except FileNotFoundError:
        st.error("❌ Error: Model file not found. Please train the model first.")
        return
    
    # Job Description Section with custom styling
    st.header("🎯 Job Description")
    job_description = st.text_area(
        "Current Job Description",
        """Looking for a Software Engineer with strong experience in Python, machine learning, 
        and web development. The ideal candidate should have experience with REST APIs, 
        data processing, and cloud technologies. Must be proficient in Git and have 
        demonstrated experience building scalable applications.""",
        height=150
    )
    
    # Resume Information Section
    st.header("📄 Resume Information")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        education = st.text_input(
            "🎓 Education",
            placeholder="e.g., B.Sc. Computer Science, Stanford"
        )
        experience = st.text_area(
            "💼 Experience",
            placeholder="e.g., Software Engineer at Google (3 yrs)",
            height=100
        )
    
    with col2:
        projects = st.text_area(
            "🚀 Projects",
            placeholder="e.g., Built an ML model; Developed a REST API",
            height=100
        )
        skills = st.text_input(
            "🛠️ Skills",
            placeholder="e.g., Python, Git, AWS, Docker"
        )
    
    # Add spacing before the button
    st.markdown("<br>" * 2, unsafe_allow_html=True)
    
    # Centered Score Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📊 Score Resume"):
            if not all([education, experience, projects, skills]):
                st.warning("⚠️ Please fill in all fields before scoring.")
                return
            
            # Get score
            score = score_resume(
                education=education,
                experience=experience,
                projects=projects,
                skills=skills,
                model_data=model_data,
                job_description=job_description
            )
            
            # Display results with enhanced styling
            st.markdown("---")
            st.header("📈 Results")
            
            # Create a progress bar for the score with custom color
            st.progress(score)
            st.metric("Resume Score", f"{score:.2f} / 1.00")
            
            # Display interpretation with icons
            if score >= 0.8:
                st.success("🌟 Interpretation: Excellent match for the position")
            elif score >= 0.6:
                st.info("✨ Interpretation: Good match for the position")
            elif score >= 0.4:
                st.warning("⚡ Interpretation: Moderate match for the position")
            else:
                st.error("⚠️ Interpretation: May not be the best match for the position")
            
            # Display detailed breakdown with styling
            st.subheader("📊 Detailed Breakdown")
            breakdown = pd.DataFrame({
                'Category': ['🎓 Education', '💼 Experience', '🚀 Projects', '🛠️ Skills'],
                'Input': [education, experience, projects, skills]
            })
            st.table(breakdown)

if __name__ == "__main__":
    main()