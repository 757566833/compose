# single node

## swap

```shell
swapoff -a


/etc/fstab

remove line
/dev/mapper/rl-swap     none                    swap    defaults        0 0

```

## selinux

```shell
sudo setenforce 0
sudo sed -i 's/^SELINUX=enforcing$/SELINUX=permissive/' /etc/selinux/config
```

## firewalld

```shell

systemctl stop firewalld

systemctl disable firewalld

systemctl mask firewalld


dnf install iptables-services iptables-utils -y


systemctl enable --now iptables

```

## iptables module

```shell
modprobe ip_tables
modprobe ip_conntrack
modprobe iptable_filter
modprobe ipt_state

vi /etc/modules-load.d/iptables.conf

ip_tables
ip_conntrack
iptable_filter
ipt_state
```

## k3s

<https://ranchermanager.docs.rancher.com/zh/how-to-guides/new-user-guides/kubernetes-cluster-setup/k3s-for-rancher>

```shell
curl -sfL https://get.k3s.io | K3S_TOKEN=SECRET sh -s - server  --cluster-init
```

## k3s registry

```shell
// vi /etc/rancher/k3s/registries.yaml


mirrors:
  docker.io:
    endpoint:
      - "registry.cn-hangzhou.aliyuncs.com"


// systemctl restart k3s
```

## k3s forward

<https://forums.rancher.cn/t/rancher-lb-ssl-tls/1707>

```shell
// vi /var/lib/rancher/k3s/server/manifests/traefik-config.yaml

apiVersion: helm.cattle.io/v1
kind: HelmChartConfig
metadata:
  name: traefik
  namespace: kube-system
spec:
  valuesContent: |-
    additionalArguments:
      - "--entryPoints.web.proxyProtocol.insecure"
      - "--entryPoints.web.forwardedHeaders.insecure"
```

## config

<https://ranchermanager.docs.rancher.com/zh/how-to-guides/new-user-guides/kubernetes-cluster-setup/k3s-for-rancher>

```shell
cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
```

## tar git

```shell
dnf install tar git -y
```

## helm

<https://helm.sh/docs/intro/install/>

```shell
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

## add repo

<https://ranchermanager.docs.rancher.com/zh/getting-started/installation-and-upgrade/install-upgrade-on-a-kubernetes-cluster>

```shell
helm repo add rancher-stable https://releases.rancher.com/server-charts/stable
```

## name space

<https://ranchermanager.docs.rancher.com/zh/getting-started/installation-and-upgrade/install-upgrade-on-a-kubernetes-cluster>

```shell
kubectl create namespace cattle-system
```

## install rancher

<https://ranchermanager.docs.rancher.com/zh/getting-started/installation-and-upgrade/install-upgrade-on-a-kubernetes-cluster>

```shell
helm install rancher rancher-stable/rancher  --namespace cattle-system --set hostname=<xxx.xxx.com>  --set tls=external --set bootstrapPassword=zxc_123456 --set rancherImage=registry.cn-hangzhou.aliyuncs.com/rancher/rancher   --set systemDefaultRegistry=registry.cn-hangzhou.aliyuncs.com --set replicas=1
```
