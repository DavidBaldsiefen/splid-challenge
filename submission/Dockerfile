###################################################################################################################
# Image for Challenge submissions                                                                                 #
# Partially taken from https://github.com/ARCLab-MIT/splid-devkit, Copyright (c) 2023 Peng Mun Siew (MIT License) #
###################################################################################################################
FROM ubuntu:22.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-pip build-essential wget && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Intall dependencies
COPY submission/submission_requirements.txt ./
RUN python -m pip install --no-cache-dir -r submission_requirements.txt

WORKDIR /app

# Copy Scripts and Models
COPY submission/models/ /models/

COPY base/datahandler.py base/datahandler.py
COPY base/utils.py base/utils.py
COPY base/classifier.py base/classifier.py
COPY base/localizer.py base/localizer.py
COPY submission/submission_new.py ./submission.py

# Specify entrypoint script
CMD python submission.py

# From main directory, build with: docker build -t my_submission -f submission/Dockerfile .
# Execute with:  docker run -v "/path/to/submission/dataset/test":/dataset/test -v "/path/to/submission/submission":/submission my_submission
# Evaluate with: python base/evaluation.py --participant "submission/submission/submission.csv" --ground_truth "submission/dataset/test_labels.csv"