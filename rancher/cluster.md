#

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

## tar git

```shell
dnf install tar git -y
```

## firewalld

```shell

systemctl stop firewalld

systemctl disable firewalld

systemctl mask firewalld

```

## network manager

```shell
vi /etc/NetworkManager/conf.d/rke2-canal.conf

[keyfile]

unmanaged-devices=interface-name:cali*;interface-name:flannel*


```

```shell
systemctl reload NetworkManager
```
