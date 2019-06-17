FROM gcr.io/ml-pipeline/ml-pipeline-dataproc-transform:6554e133dd453c62aea05ebb57a04f897c11d070

RUN apt-get update
RUN apt-get install bash

ENTRYPOINT ["bash"]
