FROM pure/python:3.7-cuda10.2-base

RUN mkdir workdir
COPY . /workdir/household_speaker_identification



ENV CMAKE_VERSION=3.21.0
ENV HTTP_PROXY=${PROXY_URL:-""}
ENV PROXY_URL=http://proxy.tcsbank.ru:8080

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        git \
        curl \
        vim \
        tmux \
        wget \
        autoconf \
        automake \
        libtool \
        dpkg-dev \
        git-lfs \
        pkg-config

RUN locale-gen en_US.UTF-8 ru_RU.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

#RUN #ln -sfn /usr/bin/python${PYTHON_VERSION} /usr/bin/python

#RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
#    python get-pip.py --proxy="${HTTP_PROXY}" --index-url="https://registry.tcsbank.ru/repository/pypi-all/simple/" && \
#    rm get-pip.py

WORKDIR workdir

RUN pip install -r household_speaker_identification/requirements.txt


