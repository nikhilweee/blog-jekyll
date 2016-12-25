---
layout: post
title: Persistent Mounts
subtitle: Automatically mount partitions in linux via the terminal
tags: [linux]
category: [hacks]
---
After installing your linux distro, though you'll be able to access your Windows/NTFS partitions easily, things can get annoying sometimes with you having to mount the partition manually everytime. Here I describe how to get away with this mess by automatically mounting partitions on boot.

You could also do the same from Unity using the **Disks** utility, but here I describe the so called _hard_ way. The way we do it is by editing the `/etc/fstab` file. This file gets executed everytime you boot up your system. We just instruct this file to also mount our desired partitions apart from what it normally does.

### Get the UUID

Before modifying `fstab`, we need the `UUID` of the partition which acts as a unique identifier for the partition to be mounted. To show a list of available partitions and their UUID, use

```
$ sudo blkid -o list
```

and you'll get an output similar to this:

```
device       fs_type label      mount point       UUID
------------------------------------------------------------------
/dev/sda1    ntfs    WINRE      (not mounted)     ACEAEE02EAEDC8A2
/dev/sda2    vfat               /boot/efi         6E82-B10E
/dev/sda4    ntfs    Windows    (not mounted)     00A8EFDAA8EFCBEA
/dev/sda6    ntfs    Downloads  /media/downloads  5CFA5B52FA5B2792
/dev/sda10   ntfs    Documents  /media/documents  86585E63585E5253
/dev/sda11   ntfs    RECOVERY   (not mounted)     3E60EB0960EAC6AD
```
This should help you in deciding which partition to mount. If you're still confused, you may use `fdisk -l` instead. This should give you the type of the partition listed as well so that you don't accidentally damage an important partition.

### Modify fstab


Once you get the UUID, you can now modify the fstab file as a root user and add an entry similar to the following at the end, ensuring that you change the `UUID` accordingly.

```
$ sudo nano /etc/fstab
```
```
<UUID>                  <mount point>           <type>  <options>   <dump>  <pass>

UUID=5CFA5B52FA5B2792	/media/downloads	ntfs-3g	rw,auto,user,exec,uid=1000         0	0

UUID=86585E63585E5253	/media/documents	ntfs-3g	rw,auto,user,exec,uid=1000         0	0
```
Once modified, save the file and run the mount command to re-execute the fstab file:

```
$ sudo mount -a
```
This should mount your new partition which would also be mounted automatically every time you boot up. Enjoy!
