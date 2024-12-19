# Pathology Q&A Models: Setup Guide and README

This project demonstrates different implementations of Q&A models leveraging Groq and Ollama APIs, with support for both simple models and Retrieval-Augmented Generation (RAG) workflows. The code includes:

1. **Simple Model using Groq**
2. **RAG Model using Groq**
3. **Simple Model using Ollama**
4. **RAG Model using Ollama**

---

### Key Dependencies

Install all dependencies using the `requirements.txt` file provided in the project directory:
```bash
pip install -r requirements.txt
```
---

## Setup Guide

### Step 1: Obtain Groq API Key
1. Visit [Groq](https://groq.com) and sign up for an account.
2. Navigate to the API section in your account settings.
3. Generate an API key and save it securely.
4. Update your code to include the API key:
   ```python
   api_key = "Your API Key"
   groq_client = Groq(api_key=api_key)
   ```

### Step 2: Install Ollama and Download Models
1. Visit [Ollama](https://ollama.com) and download the application for your operating system.
2. Install Ollama by following the on-screen instructions.
3. Open the Ollama app and download the required models:
   - **OpenBio Llama3 (7B or 8B)**: Use the app interface to search for `koesn/llama3-openbiollm-8b` and download the model.
   
   > **Note**: The models are large and may require significant disk space. Ensure you have adequate storage.

4. Update your code to reference the model:
   ```python
   ollama_model = Ollama(model="koesn/llama3-openbiollm-8b", temperature=0)
   ```

---

## Running the Code

### Step 1: Load PDFs and Websites
The script includes a scraping function to retrieve data from biorxiv.org and other sources. Be aware that scraping may fail due to server restrictions or missing URLs. Preloaded sources include:

- Example PDFs: Pathology-related publications
- Example Websites: Pathology definitions and introductions

If you wish to load additional URLs, update the `pdf_urls` and `website_urls` lists in the `load_data` function.
bio_urls.txt contains pre-scraped urls from biorx.org

### Step 2: Build Vectorstore
The `load_data` function processes the text content, splits it into smaller chunks, and embeds it using Sentence-Transformers. Run this function to build the Chroma vector database:

```python
vectorstore = load_data()
```

### Step 3: Query with RAG Workflow
Use the `query_rag` function to retrieve and answer pathology-related questions. The model will first retrieve relevant documents and generate answers based on the retrieved context.

```python
def query_rag(question):
    # Implementation as provided in the code
```

Call the function with a question:
```python
answer = query_rag("What is the role of a pathologist in cancer diagnosis?")
print(answer)
```

### Step 4: Simple Model Query
For direct Q&A without context retrieval, use the following implementations:

#### Groq Simple Model
```python
response = groq_client.chat.completions.create(
    messages=[
        {"role": "user", "content": question}
    ],
    model="llama3-8b-8192"
)
```

#### Ollama Simple Model
```python
response = ollama_model.invoke(question)
```

### Step 5: Save Results
The Q&A pairs are saved to JSON files for analysis:
- `llama3.json` (Groq Simple Model)
- `llama3-Rag.json` (Groq Rag Model)
- `open-bio.json` (Ollama Simple Model)
- `open-bio-Rag.json` (Ollama RAG Model)

---

## Notes
- Ensure adequate storage for models and data.
- Be mindful of scraping ethics and adhere to website policies.
- For debugging, add logs to capture intermediate outputs.

---
