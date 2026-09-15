FROM python:3.12-slim

# ffmpeg does the video decode/encode; numpy + opencv do the pixel work
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY wmremove_app.py .

# where uploads / cleaned videos / the sqlite db live (mount a volume here)
ENV DATA_DIR=/data
RUN mkdir -p /data
VOLUME /data

ENV HOST=0.0.0.0
EXPOSE 8080

CMD ["python", "wmremove_app.py"]
