FROM python:3.11-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Train inside the image so the model matches the installed scikit-learn version
RUN python train_model.py
EXPOSE 5000
# One worker (with threads): prometheus_client's default registry is per-process,
# so multiple workers would each expose different counters on /metrics.
CMD ["gunicorn", "-w", "1", "--threads", "8", "-b", "0.0.0.0:5000", "app:create_app()"]
