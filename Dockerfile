FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime
RUN pip install --upgrade transformers==4.41.2 diffusers==0.28.2 timm==1.0.7 accelerate==0.32.0 datasets==2.20.0 scipy==1.14.0 bitsandbytes==0.43.1
