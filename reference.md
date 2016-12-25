---
layout: page
title: Sticky Notes
subtitle: Stuff I'd like to keep note of
tags: [hacks, tutorial, linux]
category: [hacks]
date: 2016-08-21
published: true
comments: true
---

This page hosts little ideas, solutions to problems I've faced in the past or mistakes that I don't want to repeat again - stored here as a reference in case they bother me again. Read on, maybe even you might find something relevant!

## Contents

- [Shortcuts](#some-useful-aliases)
- [Docker and DC++](#docker-and-dc)
- [Test outgoing ports](#test-outgoing-ports)
- [Use Google Fonts offline](#use-google-fonts-offline)
- [Installing Node](#installing-node)
- [Installing OpenCV](#installing-opencv)
- [Raspberry Pi](#raspberry-pi)
- [Configure Mutt with Gmail](#configure-mutt-with-gmail)

### Some useful aliases
* Timesync!

```
alias timesync='sudo /etc/cron.daily/timesync'
```

```
# /etc/cron.daily/timesync
sudo date -s "$(wget -qSO- --max-redirect=0 google.com
              2>&1 | grep Date: | cut -d' ' -f5-8)Z"
```


* Short for activating virtual environments

```
alias sv='source venv/bin/activate; \
          export PS1="(${PWD##*/}-venv)$_OLD_VIRTUAL_PS1"'
```

* I use these all the time

```
alias log='tail -f /var/log/syslog'
alias nmr='sudo service network-manager restart; \
           sudo service networking restart; log'
```

* Show folder sizes in descending order

```
alias duinfo='sudo du --all --block-size=MB \
              --max-depth=1 | sort -n'
```

* I always forget the color scheme of my terminal. It's `base16_isotope`. Oh! [base16](https://chriskempson.github.io/base16/)!

### Docker and DC++
Docker and DC++ don't go hand in hand. Docker creates `iptables` for private ip addresses like 172.17.x.x and DC++ uses the same. You may keep on getting _No route to host_ when you try to connect to a hub, which is really annoying!
[@kamermans](https://github.com/kamermans) has a gist [here](https://gist.github.com/kamermans/94b1c41086de0204750b) that helped me out.

### Test outgoing ports
At times you may be behind a restricted network. Here is a [bash script](http://superuser.com/a/815481/537144) to check all outgoing ports that aren't blocked by your firewall.

I tested the open ports (upto 4000) in my university and here is the list as of October 2016:

```
22   SSH
53   DNS
80   C'mon!
143  IMAP
443  HTTPS
465  SMTP
993  IMAPS
2082 cPanel. Strange!
```

### Use Google Fonts offline
Sounds simple, but no easy way to do it. I found this [tool](https://google-webfonts-helper.herokuapp.com/fonts) by  [@majodev](http://twitter.com/majodev) to be really helpful.

### Installing Node
Again, there is an overload of methods used to install node and npm. Here is what works for me - the easy way.

```
$ sudo apt-get install nodejs npm
$ sudo npm install -g n npm
$ sudo ln -s "$(which nodejs)" /usr/bin/node
```

### Installing OpenCV

Installing OpenCV has always been a mammoth task. I seriously hate the whole process of downloading a bulky git repository and building it from the source.  And also the fact that there's no easy _pip install opencv_ way out.
But as long as you don't have any constraints like using it on an embedded device, the folks at menpo have an easy way out using anaconda.

```
conda install -c menpo opencv
conda install -c menpo opencv3
```

Of course, if you have specific requirements, it's better to build from source.

### Raspberry Pi
* [Peter Legierski](https://twitter.com/peterlegierski)'s [blog post](http://blog.self.li/post/63281257339/raspberry-pi-part-1-basic-setup-without-cables) helped me out with a headless start when I had no idea how to start my _pi_!  
* Setting up a VNC Server was easy too, thanks to the [official docs](https://www.raspberrypi.org/documentation/remote-access/vnc/README.md).  
* The forums have a good [article](https://www.raspberrypi.org/forums/viewtopic.php?t=133691&p=1025366) in case you are interested to install a GUI.  
* Use `ssh -Y pi@raspberry` to view GUI outputs from the _pi_ to the host machine.
* In case you want to access the webcam and view images, use `fswebcam image.jpg` and `feh image.jpg`.  
* Install OpenCV for raspberry with this [gist](https://gist.github.com/willprice/c216fcbeba8d14ad1138). This can make your day indeed!

### Configure Mutt with Gmail
[Nick's post](http://nickdesaulniers.github.io/blog/2016/06/18/mutt-gmail-ubuntu/) describes how to use your own gmail account in case you want to set up something like automated email alerts or regular database backups.
