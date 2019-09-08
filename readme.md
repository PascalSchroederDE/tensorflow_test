# Setup Kubeflow 

## Prerequisites

1. A working version of CUDA 10.0 is necessary to run a ML workflow on Kubeflow with GPU support.

Installation:

[docs.nvidia.com/cuda/cuda-installation-guide-linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

Important: Export paths
```
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export PATH=/usr/local/cuda-10.0/bin:/usr/local/cuda-10.0/NsightCompute-2019.1${PATH:+:${PATH}}
```
(alternatively persist exports in ~/.bashrc)

2. Docker

```apt-get install docker-ce```

## Install Minikube

1. Install nvidia-runtime for Docker

[github.com/NVIDIA/nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

2. Set nvidia docker runtime as default runtime for Docker:

Add following line as first entry of `/etc/docker/daemon.json`

```
"default-runtime": "nvidia",
```

and restart docker:

```
service docker restart
```

(before starting minikube)

3. Download Minikube

```
curl -Lo minikube https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64 \
  && chmod +x minikube
sudo cp minikube /usr/local/bin && rm minikube
```

4. Delete last line "search aag-de.ibmmobiledemo.com" out of `run/systemd/resolve/resolv.conf` (for DNS resolution)

5. `iptables -P FORWARD ACCEPT`

6. Start minikube without VM

```
minikube start --vm-driver=none --apiserver-ips=127.0.0.1 --apiserver-name localhost --cpus 4 --memory 4096 --disk-size=15g --extra-config=kubelet.resolv-conf=/run/systemd/resolve/resolv.conf
```

## Install kubeflow pipelines

1. `export PIPELINE_VERSION=master`

2. Install kubeflow pipelines:

```
kubectl apply -f https://raw.githubusercontent.com/kubeflow/pipelines/$PIPELINE_VERSION/manifests/namespaced-install.yaml
```

3. Check via `kubectl get pods -n kubeflow` until every pod is running

4. `kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80`

5. Access [localhost:8080](localhost:8080)

(a few reloads may be necessary to load all existing pipelines)

# Start Minikube and kubeflow with existing installation


1. Check if minikube is up and running:

`kubectl get nodes`

2. Check if kubeflow is up and running:

`kubectl get pods -n kubeflow`

3. Forward port:

`kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80`

4. Access localhost:8080


## Use local data on Kubeflow

1. Create a yaml file looking similar to this:
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

2. Create pvc with yaml file (e.g. pvc.yaml)

`kubectl create -f pvc.yaml -n kubeflow`

3. Mounted path will be located in /tmp/hostpath-provisioner/

ID can be grabbed via 
`kubectl get pvc -n kubeflow`

4. Copy necessary files to hostpath.
Files will be mounted to /mnt/  of Docker container / K8s pod

5. Mount pvc to stages, which need access to those files:

Add
`.apply(onprem.mount_pvc("volume-name", 'local-storage', "/mnt"))`
to definition of stage

(e.g. 

```
def train_pipeline([...]):
  train = train_op([...]).apply(onprem.mount_pvc("train-vol", 'local-storage', "/mnt"))
  load = load_op([...]).apply(onprem.mount_pvc("train-vol", 'local-storage', "/mnt"))

if __name__ == '__main__':
  kfp.compiler.Compiler().compile(train_pipeline, __file__ + '.tar.gz')
```
)

## Restart Minikube

1. Export pipeline version

```
export PIPELINE_VERSION=master
```

2. Delete existing Pipeline installation:

```
kubectl delete -f https://raw.githubusercontent.com/kubeflow/pipelines/$PIPELINE_VERSION/manifests/namespaced-install.yaml
```

3. Stop Minikube 

`minikube stop`

4. If necessary delete all files 

```
rm -r ~/.minikube
rm -r /var/lib/minikube/
rm /var/lib/kubeadm. 
rm -r /var/lib/kubelet/
rm -r /etc/kubernetes/
rm -r /etc/kubernetes/
rm -rf ~/.kube/
```

5. If necessary prune Docker images and containers

`docker system prune --all`

6. Delete last line "search aag-de.ibmmobiledemo.com" out of `run/systemd/resolve/resolv.conf`

7. Start Minikube

```
minikube start --vm-driver=none --apiserver-ips=127.0.0.1 --apiserver-name localhost --cpus 4 --memory 4096 --disk-size=15g --extra-config=kubelet.resolv-conf=/run/systemd/resolve/resolv.conf
```

8. Install kubeflow pipelines:

```
kubectl apply -f https://raw.githubusercontent.com/kubeflow/pipelines/$PIPELINE_VERSION/manifests/namespaced-install.yaml
```


## Troubleshooting

### Ports are used

1. Ensure microk8s is stopped

`microk8s.stop` 

2. If that doesnt help stop kubelet:

`systemctl stop kubelet`

3. If it is still not working kill processes - first get the processes, which are using the ports:

`netstat -nltp | grep 10250`

and then kill the process:

`kill 7564`


### Pods won't come up

Reinstall docker, nvidia-docker2 and change default runtime of docker to nvidia as described above

### DNS resolution problems

Delete last line "search aag-de.ibmmobiledemo.com" out of `run/systemd/resolve/resolv.conf` (for DNS resolution)

If necessary, restart Minikube and Docker daemon and redeploy kubeflow as described above.


