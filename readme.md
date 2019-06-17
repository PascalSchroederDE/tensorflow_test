# Tensorflow test pipeline for Kubeflow

## Use case

This repository stores all files for an example Kubeflow pipeline for ML. First stage will train a model, second stage will load and evaluate it. In this readme file you will also find an installation tutorial for setting up the Kubeflow pipeline toolkit on a Minikube cluster.

## Set up Kubeflow pipeline on Minikube cluster

### Prerequisites

A working version of CUDA 10.0 is necessary to run a ML workflow on Kubeflow with GPU support.

Installation:

[docs.nvidia.com/cuda/cuda-installation-guide-linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

Important: Export paths
``` export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export PATH=/usr/local/cuda-10.0/bin:/usr/local/cuda-10.0/NsightCompute-2019.1${PATH:+:${PATH}}``` 
(alternatively persist exports in ~/.bashrc)

### Set up Minikube

1. Install Docker

``` apt-get install docker-ce``` 

2. Install Minikube 

``` curl -Lo minikube https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64 \
  && chmod +x minikube
sudo cp minikube /usr/local/bin && rm minikube```  
3. Install nvidia-runtime for Docker

[github.com/NVIDIA/nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

4. Set nvidia docker runtime as default runtime for Docker:

Add following line as first entry of `/etc/docker/daemon.json`

``` "default-runtime": "nvidia",```  
and restart docker:

``` service docker restart``` 

(before starting minikube)

5. Delete last line "search [...]ibm[...]" out of `run/systemd/resolve/resolv.conf` (for DNS resolution) 

6. ``` iptables -P FORWARD ACCEPT``` 

7. Delete existing minikube data

``` minikube stop
minikube delete
rm -r ~/.minikube
rm -rf /var/lib/minikube/certs/``` 

8. Start minikube without VM

``` minikube start --vm-driver=none --apiserver-ips=127.0.0.1 --apiserver-name localhost --cpus 4 --memory 4096 --disk-size=15g --extra-config=kubelet.resolv-conf=/run/systemd/resolve/resolv.conf``` 

(Ensure resources are sufficient and resolv.conf is set correctly for DNS resolution)

9. Clear Docker images:

``` docker system prune --all``` 

10. Install NVIDIA device plugin:

``` kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v1.10/nvidia-device-plugin.yml``` 

### Install kubeflow pipelines

1. ``` export PIPELINE_VERSION=master``` 

2. Delete existing Pipeline installation:

``` kubectl delete -f https://raw.githubusercontent.com/kubeflow/pipelines/$PIPELINE_VERSION/manifests/namespaced-install.yaml``` 

3. Install kubeflow pipelines:

``` kubectl apply -f https://raw.githubusercontent.com/kubeflow/pipelines/$PIPELINE_VERSION/manifests/namespaced-install.yaml``` 

4. Check via `kubectl get pods -n kubeflow` until every pod is running

5. ``` kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80``` 

6. Access [localhost:8080]

(a few reloads may be necessary to load all existing pipelines)

### Run a ML workflow pipeline

1. Create persistent volume claim:

``` kubectl create -f pvc.yaml -n kubeflow``` 


pvc.yaml should look similar to:

``` 
kind: PersistentVolumeClaim
apiVersion: v1
metadata:
  name: train-vol
spec:
  storageClassName: standard
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
``` 

Hostpath will be located in `/tmp/hostpath-provisioner/`

ID can be grabbed via 
``` kubectl get pvc -n kubeflow``` 

2. Copy necessary files to hostpath.
Files will be mounted to `/mnt/`  of Docker container / K8s pod

3. Mount pvc to stages, which need access to those files, in ml_pipeline.py:

Add
``` .apply(onprem.mount_pvc("train-vol", 'local-storage', "/mnt"))``` 
to definition of stage

4. Upload pipeline

5. Run pipeline 




