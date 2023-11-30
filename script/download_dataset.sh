#!/bin/bash

echo "# ---------------------------------------------------------------------------------------------------- #"
echo "  This script should be run in root directory of project."
echo "  Current working directory: $(pwd)"
echo "# ---------------------------------------------------------------------------------------------------- #"


ROOT_DIR="$(pwd)"
DATA_ROOT_DIR="${ROOT_DIR}/data"
DOC_BASE_URL="https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHIS/2018"
DATASET_BASE_URL="https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/2018"


# Download Readme
mkdir -p ${DATA_ROOT_DIR} && cd ${DATA_ROOT_DIR}
curl -sO ${DOC_BASE_URL}/readme.pdf
echo "[SUCCESS] Download Readme"


# Download components
for data_id in "familyxx" "househld" "personsx" "samchild" "samadult"
do
  DATA_DIR="${DATA_ROOT_DIR}/${data_id}"
  {
    mkdir -p ${DATA_DIR} && cd ${DATA_DIR} && \
    curl -sO ${DOC_BASE_URL}/${data_id}_summary.pdf && \
    curl -sO ${DOC_BASE_URL}/${data_id}_layout.pdf && \
    curl -sO ${DOC_BASE_URL}/${data_id}_freq.pdf && \
    curl -sO ${DATASET_BASE_URL}/${data_id}csv.zip && \
    unzip -qqo *.zip && rm *.zip && \
    echo "[SUCCESS] Download ${data_id}"
  } || {
    cd .. && rm -rf ${DATA_DIR} && \
    echo "[FAILURE] Download ${data_id}"
  }
done

echo "[SUCCESS] Download Complete"
