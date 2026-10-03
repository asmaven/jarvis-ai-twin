# Jarvis (Ayaz's AI twin) on Hugging Face Spaces. Built from the folder deploy/huggingface/push.sh stages.
FROM python:3.11-slim

RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
RUN mkdir -p /app/logs && chown user /app /app/logs  # the app writes chat logs here
USER user

# Secrets come from the Space's settings, never from files:
#   ANTHROPIC_API_KEY, TWIN_PRIVATE_PASSCODE, TWIN_ACCESS_KEY (the ?key= in the shared link), TWIN_KB_TEXT (knowledge base)
ENV PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
