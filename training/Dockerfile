FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-12.py310:latest
WORKDIR /root

RUN pip3 install pytorch-forecasting pytorch-lightning scipy google-cloud-storage 


COPY ./ ./

ENTRYPOINT ["python3", "train.py"]