FROM python:3.11-slim

WORKDIR /app

COPY webapp/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY webapp/ ./webapp/
COPY vault/ ./vault/

EXPOSE 7860

CMD ["python3", "webapp/app.py"]
