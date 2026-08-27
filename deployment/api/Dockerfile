# Image de deploiement de l'API GetAround (Hugging Face Spaces, SDK docker).
# Python 3.13 : meme version mineure que l'environnement d'entrainement, afin
# que le depicklage de model.pkl soit garanti reproductible.
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY main.py model.pkl ./

# Hugging Face Spaces expose le port 7860.
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:7860/health').status==200 else 1)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
