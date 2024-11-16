
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import PyPDF2
from docx import Document

# Load the lightweight MiniLM model for cosine similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the Flan-T5 model for prompt-based interaction
flan_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
flan_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

# Text extraction functions
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    return text

def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# Function to match job description to CVs
def match_cv_to_job_description(cvs, job_description):
    job_embedding = model.encode(job_description, convert_to_tensor=True)
    scores = []
    for cv in cvs:
        cv_embedding = model.encode(cv, convert_to_tensor=True)
        score = util.pytorch_cos_sim(cv_embedding, job_embedding)
        scores.append(score.item())
    return scores

# Function to match job description to CVs with explicit skill comparison, returning a score from 0 to 100
def match_cv_to_job_description_explicit(cvs, job_description):
    scores = []

    # Define a prompt template for comparison with scoring from 0 to 100
    prompt_template = """
    Job Description:
    {job_description}

    CV:
    {cv}

    Instructions:
    Compare the skills, qualifications, and experiences listed in the CV with those required in the job description.
    Identify specific skills or qualifications that align with the job requirements and provide a summary of the match quality.

    Rate the alignment on a scale from 0 to 100, where:
    100 = Excellent match with most or all key skills and experiences
    80 = Good match with many key skills and experiences
    60 = Moderate match, with some relevant skills and experiences
    40 = Limited match, lacking key skills but with some minor relevance
    20 = Poor match, with little relevance
    0 = No relevance or match

    Only return the rating as an integer value.
    """

    for cv in cvs:
        # Fill in the prompt with the job description and CV content
        prompt = prompt_template.format(job_description=job_description, cv=cv)

        # Tokenize and generate response using Flan-T5
        inputs = flan_tokenizer(prompt, return_tensors="pt")
        outputs = flan_model.generate(**inputs, max_length=10)
        response = flan_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the rating from the model response
        try:
            rating = int(response.strip())  # Convert the response to an integer
        except ValueError:
            rating = 0  # Default to 0 if parsing fails

        # Append the rating to the scores list
        scores.append(rating)

    return scores

# Streamlit UI
st.title("CV Matching to Job Description")

# Job description input
job_description = st.text_area("Job Description", "Enter the job description here")

# Upload multiple CVs
uploaded_files = st.file_uploader("Upload CVs (PDF or DOCX)", type=['pdf', 'docx'], accept_multiple_files=True)

if st.button("Match CVs"):
    if job_description and uploaded_files:
        cvs = []
        for uploaded_file in uploaded_files:
            # Extract text from each uploaded file
            if uploaded_file.name.endswith('.pdf'):
                cvs.append(extract_text_from_pdf(uploaded_file))
            elif uploaded_file.name.endswith('.docx'):
                cvs.append(extract_text_from_docx(uploaded_file))

        # Get similarity scores
        scores = match_cv_to_job_description(cvs, job_description)
        scores_explicit = match_cv_to_job_description_explicit(cvs, job_description)

        # Display the scores
        st.write("Cosine Similarity Scores:")
        for i, score in enumerate(scores):
            st.write(f"CV {i+1}: {score:.4f}")

        # Display the explicit scores
        st.write("Prompt-Based Matching Scores (0-100):")
        for i, score in enumerate(scores_explicit):
            st.write(f"CV {i+1}: {score}")
    else:
        st.error("Please upload at least one CV and provide a job description.")
