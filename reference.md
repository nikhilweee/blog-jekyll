---
layout: page
title: Self-ly
subtitle: Stuff I'd like to keep note of
tags: [hacks, tutorial, linux]
category: [hacks]
date: 2016-08-21
published: true
comments: true
---

This page hosts little ideas, solutions to problems I've faced in the past or mistakes that I don't want to repeat again - stored here as a reference in case they bother me again. Go ahead, even you may find something relevant!

## Contents

- [Docker and DC++](#docker-vs-dc)
- [Test outgoing ports](#test-outgoing-ports)
- [Use Google Fonts offline](#use-google-fonts-offline)
- [Installing Node](#installing-node)
- [Headless start for Raspberry](#headless-start-for-raspberry)


### Docker and DC++
Docker and DC++ don't go hand in hand. Docker creates `iptables` for private ip addresses like 172.17.x.x and DC++ uses the same. You may keep on getting _No route to host_ when you try to connect to a hub, which is really annoying!
[@kamermans](https://github.com/kamermans) has a gist [here](https://gist.github.com/kamermans/94b1c41086de0204750b) that helped me out.

### Test outgoing ports
At times you may be behind a restricted network. Here is a bash script to check all outgoing ports that aren't blocked by your firewall, courtesy of [superuser](http://superuser.com/a/815481/537144)

### Use Google Fonts offline
Sounds simple, but no easy way to do it. I found this [tool](https://google-webfonts-helper.herokuapp.com/fonts) by  [@majodev](http://twitter.com/majodev) to be really helpful.

### Installing Node
Again, there is an overload of methods used to install node and npm. Here is what works for me - the easy way.

```
$ sudo apt-get install nodejs npm
$ sudo npm install -g n npm
$ sudo ln -s "$(which nodejs)" /usr/bin/node
```

### Headless start for Raspberry
[Peter Legierski](https://twitter.com/peterlegierski)'s blog post [here](http://blog.self.li/post/63281257339/raspberry-pi-part-1-basic-setup-without-cables) helped me out when I had no idea how to start my _pi_!
Setting up a VNC Server was easy too, thanks to the official docs  [here](https://www.raspberrypi.org/documentation/remote-access/vnc/README.md)
The forums have a good article in case you are interested to install a GUI [here](https://www.raspberrypi.org/forums/viewtopic.php?t=133691&p=1025366)
`ssh -Y pi@raspberry` and `fswebcam image.jpg` and `feh image.jpg`
Install OpenCV for raspberry [here](https://gist.github.com/willprice/c216fcbeba8d14ad1138)

### Configure Mutt and Gmail
http://nickdesaulniers.github.io/blog/2016/06/18/mutt-gmail-ubuntu/
