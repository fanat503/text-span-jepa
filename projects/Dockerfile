FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONHASHSEED=42
ENV TORCH_DETERMINISTIC=1
# V29 FINAL IDEAL - use IDEAL versions for Oral 6 Strong Accept, ULTIMATE V11 for graphs
RUN python3 frontier-01-bilinearity-break-ULTIMATE-V11.py && python3 frontier-01-bag-of-words-test.py && python3 frontier-01-graphs-ULTIMATE-V11.py && python3 frontier-01-eval-numpy-ideal.py
CMD ["python3", "frontier-01-eval-numpy-ideal.py"]
