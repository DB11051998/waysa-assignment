FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel
RUN pip install pandas Transformers nltk scikit-learn openpyxl fastapi uvicorn
EXPOSE 7600
# Copy the application code into the container
COPY . /workspace

# Set the working directory
WORKDIR /workspace

# Run FastAPI using Uvicorn, binding to host 0.0.0.0 and port 7600
CMD ["uvicorn", "inference_api:app", "--host", "0.0.0.0", "--port", "7600"]

