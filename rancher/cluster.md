#

## swap

```shell
swapoff -a


vi /etc/fstab

// red hat remove line
/dev/mapper/rl-swap     none                    swap    defaults        0 0

// ubuntu remove line 
/swap.img      none    swap    sw      0       0
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

## redhat firewalld

```shell

systemctl stop firewalld

systemctl disable firewalld

systemctl mask firewalld

```

## ubuntu UFW

```shell
sudo ufw disable 

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
