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
COPY submission/submission_requirements.txt ./
# COPY environment.yml ./
RUN python -m pip install --no-cache-dir -r submission_requirements.txt
# RUN conda config --set offline false
# RUN conda update conda
# RUN conda env create -f environment.yml
# RUN conda activate splid

WORKDIR /app

# COPY WARMUP DATA
#ADD dataset/test/* /dataset/test/

# COPY WHATEVER OTHER SCRIPTS YOU MAY NEED
COPY submission/models/ew_localizer.hdf5 /models/ew_localizer.hdf5
COPY submission/models/EW_localizer_scaler.pkl /models/EW_localizer_scaler.pkl
COPY submission/models/ns_localizer.hdf5 /models/ns_localizer.hdf5
COPY submission/models/NS_localizer_scaler.pkl /models/NS_localizer_scaler.pkl

COPY submission/models/ew_ns_classifier_oneshot.hdf5 /models/ew_ns_classifier_oneshot.hdf5
COPY submission/models/ew_ns_classifier_scaler_oneshot.pkl /models/ew_ns_classifier_scaler_oneshot.pkl

COPY base/datahandler.py base/datahandler.py
COPY base/utils.py base/utils.py
COPY base/classifier.py base/classifier.py
COPY base/localizer.py base/localizer.py
COPY base/evaluation.py base/evaluation.py
#ADD base/ ./base
COPY submission/submission_combined_new.py ./submission.py

# SPECIFY THE ENTRYPOINT SCRIPT
CMD python submission.py


# From main directory, build with: docker build -t my_submission -f submission/Dockerfile .
# Execute with:  docker run -v "/home/david/Code/splid-challenge/submission/dataset/test":/dataset/test -v "/home/david/Code/splid-challenge/submission/submission":/submission my_submission
# Evaluate with: python base/evaluation.py --participant "submission/submission/submission.csv" --ground_truth "submission/dataset/test_labels.csv"
