# Dashboard Streamlit GetAround (Hugging Face Spaces, SDK docker).
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY analysis.py app.py ./

# Les datasets sont pre-telecharges au build : le conteneur demarre alors sans
# dependance reseau, et le premier affichage n'attend pas un telechargement.
RUN python -c "import analysis; analysis.load_delay_data(); analysis.load_pricing_data()"

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:7860/_stcore/health').status==200 else 1)"

CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
