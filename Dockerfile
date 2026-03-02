FROM python:3.11-slim

WORKDIR /app

COPY requirement.txt .
RUN pip install -r requirement.txt

# Pre-download the model during build
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

EXPOSE 7860

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]