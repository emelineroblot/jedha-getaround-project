# Dockerfile pour déploiement sur Hugging Face Spaces
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de requirements
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copier tous les fichiers de l'API
COPY . .

# Exposer le port 7860 (requis par Hugging Face)
EXPOSE 7860

# Commande pour lancer l'API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
