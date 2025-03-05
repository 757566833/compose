# Rancher

## version

stable verison is 2.10.3

## desc

os is rocky linux@9

As of Rocky Linux 9.0, iptables and all of the utilities associated with it, are deprecated. This means that future releases of the OS will be removing iptables. For that reason, it is highly recommended that you do not use this process. If you are familiar with iptables, we recommend using iptables Guide To firewalld. If you are new to firewall concepts, then we recommend firewalld For Beginners.

<https://docs.rockylinux.org/pt/guides/security/enabling_iptables_firewall/>

## k3s must use ip_tables

disable firewalled and install iptables

```shell
Stop firewalld:

systemctl stop firewalld

Disable firewalld so it won't start on boot:

systemctl disable firewalld

Mask the service so that it can't be found:

systemctl mask firewalld


Installing And Enabling iptables Services¶
Next, we need to install the old iptables services and utilities. This is done with the following:

dnf install iptables-services iptables-utils

This will install everything that is needed to run a straight iptables rule set.

Now we need to enable the iptables service to make sure that it starts on boot:

systemctl enable iptables

```

## can‘t initialize iptables table `filter‘

<https://blog.csdn.net/qq_48391148/article/details/134972833>

get list by keyword ip

```shell
 lsmod | grep ip
```

if not has  iptable_filter

```shell
[root@localhost ~]##  modprobe iptable_filter
[root@localhost ~]##  lsmod | grep ip
iptable_filter         16384  0
ip_tables              28672  1 iptable_filter
...

```

## start

```shell
modprobe ip_tables
modprobe ip_conntrack
modprobe iptable_filter
modprobe ipt_state
```

## start on boot

### add custom service

```shell
sudo vim /etc/systemd/system/iptable_modules.service
```

```shell

[Unit]
Description=Load kernel ip_table  modules

[Service]
Type=oneshot
ExecStart=/sbin/modprobe ip_tables
ExecStart=/sbin/modprobe ip_conntrack
ExecStart=/sbin/modprobe iptable_filter
ExecStart=/sbin/modprobe ipt_state

[Install]
WantedBy=multi-user.target
```

```shell
systemctl enable iptable_modules.service
```

### add modules

```shell
vi /etc/modules-load.d/iptables.conf

ip_tables
ip_conntrack
iptable_filter
ipt_state
```

## working without iptable

rancher version set 2.4.9

## other

<https://github.com/rancher/rancher/issues/37958>

## init rancher with docker or podman volume

<https://forums.rancher.cn/t/docker-run-rancher-rancher-mirrored-pause/3546>

if podman

```shell
sudo podman run --rm --user 0 --entrypoint "" -v $(pwd):/output:Z rancher/rancher:v2.10.3 cp /var/lib/rancher/k3s/agent/images/k3s-airgap-images.tar /output/k3s-airgap-images.tar
```

## use in china

<https://forums.rancher.cn/t/rancher/3366>

<https://forums.rancher.cn/t/rancher/3317>

## podman quadlet

```shell
/etc/containers/systemd/rancher.container

[Unit]
Description=Rancher Container
Wants=network.target
After=network.target

[Container]
Image=rancher/rancher:v2.10.3
ContainerName=rancher
Volume=/home/rancher/rancher:/var/lib/rancher:Z
Volume=/home/rancher/log:/var/log:Z
Environment=AUDIT_LEVEL=1
Environment=CATTLE_SYSTEM_DEFAULT_REGISTRY=registry.cn-hangzhou.aliyuncs.com
PublishPort=4000:80
PublishPort=4443:443
PodmanArgs=--privileged
Exec=--no-cacerts

[Service]
Restart=always

[Install]
WantedBy=multi-user.target

```

-----------------
2025.3.4
