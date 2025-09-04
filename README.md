# Job-Matching 

A **Streamlit-based web app** that compares resumes against job descriptions using **DistilBERT embeddings** and **cosine similarity**.  

The app extracts text from PDFs, generates embeddings, and outputs a **match score (%)** for each resume, helping recruiters and hiring managers quickly identify the best candidates.  

---

## 🚀 Features  
✅ Upload a **Job Description (PDF)**  
✅ Upload **Multiple Candidate Resumes (PDFs)**  
✅ Automatic text extraction with **PyPDF2**  
✅ Embedding generation using **DistilBERT**  
✅ Cosine similarity scoring with **scikit-learn**  
✅ Results displayed in an interactive **pandas DataFrame**  
✅ Highlights the **Best Matching Resume**  

---

## 🛠️ Tech Stack  
- **Frontend:** [Streamlit](https://streamlit.io/)  
- **NLP Model:** [DistilBERT](https://huggingface.co/distilbert-base-uncased) (Hugging Face Transformers)  
- **Data Processing:** PyTorch, pandas, scikit-learn  
- **PDF Parsing:** PyPDF2  

---

## 📦 Installation  

1. Clone the repository:  
```bash
git clone https://github.com/your-username/job-matching-app.git
cd job-matching-app
