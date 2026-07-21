FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r StackOverflow-Tag-Prediction/requirements.txt
EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "StackOverflow-Tag-Prediction/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
