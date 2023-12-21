# version

stable verison is 2.7.9

docker cant up 

rocky default cgroup is 2 

just only run rke1 (2.4.9)


grep cgroup /proc/filesystems

docker info
```
...
Cgroup Version 2
```

setting 

```
# vim /etc/default/grub
GRUB_CMDLINE_LINUX="systemd.unified_cgroup_hierarchy=0"
```

rocky
Type the following command as root user:
sudo grub2-mkconfig -o /boot/grub2/grub.cfg
Reboot your Linux box
sudo reboot