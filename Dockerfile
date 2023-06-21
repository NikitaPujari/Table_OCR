FROM python:3.9-slim-buster

COPY . /app

WORKDIR /app

#RUN mkdir -p /app/Cropped_image_folder

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt --use-pep517

# Run the command to install Tesseract OCR and its language data
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-deu \
    && rm -rf /var/lib/apt/lists/*

CMD ["python","main.py"]

Expose 5000