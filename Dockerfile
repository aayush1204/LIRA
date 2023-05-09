FROM python:3.10.8
ENV pythonunbuffered 1

WORKDIR /lira-docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN chmod +x entrypoint.sh
RUN ./entrypoint.sh
RUN python -m nltk.downloader all -d /usr/local/nltk_data
# Run the web service on container startup. Here we use the gunicorn webserver, with one worker process and 8 threads.
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 Lira.wsgi:application