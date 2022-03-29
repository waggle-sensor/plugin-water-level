FROM waggle/plugin-base:1.1.1-ml-cuda10.2-l4t

COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r /app/requirements.txt

COPY waterlevel.py /app/app.py

WORKDIR /app
ENTRYPOINT ["python3", "-u", "/app/app.py"]
