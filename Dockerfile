FROM nvcr.io/nvidia/pytorch:22.11-py3
WORKDIR /workspace
USER root
EXPOSE 8800
COPY ./requirements.txt /workspace
RUN apt-get update && apt-get upgrade -y && apt-get install git -y
RUN export FORCE_CUDA="1" && export TORCH_CUDA_ARCH_LIST=8.6 && pip install -r /workspace/requirements.txt
# RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && pip install -r /workspace/requirements.txt
# RUN pip install git+https://github.com/openai/CLIP.git
# RUN git clone https://github.com/pharmapsychotic/clip-interrogator.git
# RUN git clone https://github.com/salesforce/BLIP
# COPY ./app.py /workspace/BLIP/app.py
# COPY ./app.ipynb /workspace/BLIP/app.ipynb
ENTRYPOINT ["jupyter-lab","--no-browser","--allow-root","--port=8800","--ip=0.0.0.0","--ServerApp.token=123456"]
