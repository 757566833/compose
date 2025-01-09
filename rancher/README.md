# version

stable verison is 2.7.9


# desc

os is rocky linux@9

As of Rocky Linux 9.0, iptables and all of the utilities associated with it, are deprecated. This means that future releases of the OS will be removing iptables. For that reason, it is highly recommended that you do not use this process. If you are familiar with iptables, we recommend using iptables Guide To firewalld. If you are new to firewall concepts, then we recommend firewalld For Beginners.

https://docs.rockylinux.org/pt/guides/security/enabling_iptables_firewall/


## k3s must use ip_tables

disable firewalled and install iptables

```
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

https://blog.csdn.net/qq_48391148/article/details/134972833

get list by keyword ip
```
 lsmod | grep ip
```

if not has  iptable_filter


```
[root@localhost ~]# modprobe iptable_filter
[root@localhost ~]# lsmod | grep ip
iptable_filter         16384  0
ip_tables              28672  1 iptable_filter
...

```

# start on boot


```
sudo vim /etc/systemd/system/iptable_modules.service
```

```



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

```
systemctl enable iptable_modules.service
```

# working without iptable

version set 2.4.9


# other

https://github.com/rancher/rancher/issues/37958




-----------------   
2023.12.22 