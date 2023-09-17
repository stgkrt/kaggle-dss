ARG CONTAIER_VER=134

FROM gcr.io/kaggle-gpu-images/python:v${CONTAIER_VER}

ENV lang="ja_jp.utf-8" language="ja_jp:ja" lc_all="ja_jp.utf-8"

RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
WORKDIR /kaggle

# if you cannot use ssh
ENV LD_LIBRARY_PATH=/usr/local/lib