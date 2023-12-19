####################################
# EXAMPLE IMAGE FOR PYTHON SUBMISSIONS
####################################
FROM ubuntu:22.04

# Use this one for submissions that require GPU processing
#FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-pip build-essential wget && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
# ENV CONDA_DIR /opt/conda
# RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
#     /bin/bash ~/miniconda.sh -b -p /opt/conda
# ENV PATH=$CONDA_DIR/bin:$PATH

# ADITIONAL PYTHON DEPENDENCIES (if you have them)
 COPY requirements.txt ./
# COPY environment.yml ./
RUN python -m pip install --no-cache-dir -r requirements.txt
# RUN conda config --set offline false
# RUN conda update conda
# RUN conda env create -f environment.yml
# RUN conda activate splid

WORKDIR /app

# COPY WARMUP DATA
#ADD dataset/test/* /dataset/test/

# COPY WHATEVER OTHER SCRIPTS YOU MAY NEED
COPY trained_models/dense_model_ew.h5 /trained_models/dense_model_ew.h5
COPY trained_models/dense_model_ns.h5 /trained_models/dense_model_ns.h5
COPY base/datahandler.py base/datahandler.py
COPY base/utils.py base/utils.py
#ADD base/ ./base
COPY submission.py ./

# SPECIFY THE ENTRYPOINT SCRIPT
CMD ["python", "-u", "submission.py"]


# Build with: docker build -t my_submission .
# Execute with: docker run -v "C:/path/to/dataset":/dataset "C:/path/to/submissionfolder":/submission my_submission