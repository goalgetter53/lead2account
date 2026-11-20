FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .
COPY data ./data
COPY models ./models
COPY labels ./labels
EXPOSE 8765
ENV PYTHONPATH=src
# default: run the labeller; override with e.g. `python -m l2a.experiment`
CMD ["python", "-m", "l2a.labeller.app", "--host", "0.0.0.0", "--port", "8765"]
