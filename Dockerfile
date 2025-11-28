FROM python:3.9-slim

# REMOVED: The user creation lines
# RUN useradd -m -u 1000 user 

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# REMOVED: Switching to "user"
# USER user

# Hugging runs app on port 7860
EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]