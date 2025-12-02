FROM python:3.11-slim
ENV DEBIAN_FRONTEND=noninteractive

ENV MPLBACKEND=Agg

WORKDIR /app

RUN apt-get update && apt-get upgrade -y && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY python_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader punkt stopwords wordnet omw-1.4 punkt_tab

COPY python_app/ .

CMD ["python", "Detector.py"]
#docker build --no-cache -t python_app .
