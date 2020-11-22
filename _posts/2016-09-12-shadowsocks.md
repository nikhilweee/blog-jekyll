---
layout: post
title: Shadowsocks
subtitle: Dont let your ISP block you
tags: [shadowsocks, tunnel, android, blocked, supervisor, linux, digitalocean]
category: Tutorials
published: true
---
[Shadowsocks](https://shadowsocks.org/)! Sounds like the new superhero after _Deadpool_. Well, in a way it _is_ a superhero! It let's you bypass censorship, even the GFW! [^gfw] Okay, so what's this fuss about? Before going further, I think it needs some introduction.

[^gfw]: The Great Firewall of China

>Shadowsocks is a SOCKS5 compatible proxy server/client suite which excels at being undetectable. It leaves no fingerprint even when you use deep packet inspection. [^quora]

[^quora]: Inspired by James Swineson's [answer](https://www.quora.com/What-is-shadowsocks/answer/James-Swineson) on Quora

Whoa! Wait, What does that mean? In simple terms, this means that shadowsocks lets you access blocked resources and is hard to detect by even the best firewalls.

_Seems Interesting. I would definitely want one!_  
Okay, but before moving further, it's important to know how this stuff works.

### Working

Shadowsocks is some software that has to be set up individually on two systems. One is your device, or the _client_, which wants to bypass firewalls, and the other is a _server_ - quite simply a device which has unrestricted access to the internet.  
The client then connects to the server (which isn't blocked by the firewall), and tunnels all its traffic through the server via shadowsocks. The firewall thinks this is legitimate traffic, and so doesn't block you.

![image](/img/shadowsocks/arch.svg)
<em><center>The big picture</center><em>

### Why Shadowsocks
Now you might think that if the client is already allowed to connect to the server, then why do we need shadowsocks in the first place?
There are a couple of reasons why I recommend shadowsocks.

* One, although you may have unrestricted access to the server, the firewall can still find out whether you're using it to tunnel blocked resources. Shadowsocks excels at deceiving. It fools the firewall into believing that your communication is legitimate. Also, it's fast. Trust me. You're not limited by some crappy VPN that only lets through a few kilobits per second.

* Two, an interesting advantage with shadowsocks is that it is available for most ecosystems. Ofcourse it's available on the usual Linux, Mac and Windows, but there's also android phones and wifi routers! [^openwrt] This means you don't need a PC to tunnel anymore!

[^openwrt]: Shadowsocks is also available for routers running OpenWRT

### Setup
Enough of the talk. Let's get some hands dirty.  

I'll be using an android phone as the client and an ubuntu machine as the server. If you cannot arrange for a server, I'd recommend renting one online. DigitalOcean lets you buy a decent server for as low as $5 per month. You can use [this link](https://m.do.co/c/ad1b7e083b2e) while signing up to get $10 in free credit.

#### Ubuntu
I'll be setting up the server now. [^vpndada] First, a little housekeeping.

[^vpndada]: VPNDada has a good [step by step tutorial](https://www.vpndada.com/how-to-setup-a-shadowsocks-server-on-digitalocean/) on setting up shadowsocks on a [digitalocean droplet](https://m.do.co/c/ad1b7e083b2e) from scratch.

```sh
$ apt-get update
$ apt-get install python-pip
$ pip install shadowsocks
```

Next, create a config file `/etc/shadowsocks.json` and paste this

```json
{
 "server": "0.0.0.0",
 "server_port": 465,
 "local_port": 1080,
 "password": "secret",
 "timeout": 300,
 "method": "aes-256-cfb"
}
```

There are some things worth looking out here.

* Setting `server` to `0.0.0.0` essentially means that you're making shadowsocks accessible from outside the machine as well. `server_port` simply specifies the port over which shadowsocks would be accessible. So, if the server's public IP is `11.22.33.44`, shadowsocks is accessible on `11.22.33.44:465` according to the configuration above.
* Also, make sure that `server_port`, which is 465 in this case, isn't blocked on your home network. [^testports] Else you wouldn't be able to connect to the server in the first place.
* It turns out that you shouldn't use the common HTTP ports like 80 and 443 for shadowsocks. Atleast they didn't work out for me.

[^testports]: Here's a nice resource to help you [test open ports](http://nikhilweee.me/reference/#test-outgoing-ports).

You don't need to bother about other options for now. Just keep note of the password, we'll be using that later. Next, you need to run shadowsocks in the background.

```sh
$ ssserver -c /etc/shadowsocks.json -d start
```

Once a client is connected, you can check the logs in `/var/log/shadowsocks.log` to confirm that it's actually working. There! Halfway done.

#### Android

Coming to the client, setting up shadowsocks for android is a piece of cake. Just install the [shadowsocks android app](https://play.google.com/store/apps/details?id=com.github.shadowsocks), and enter the `remote port`, `local port`, `encryption method` and `password` according to the config file you just created on the server. Enter your server's public IP [^domain] in place of the `server` option, and you're ready to hit the connect button.

[^domain]: You could also enter a domain instead of an IP if you have one set up.

An interesting option here is to selectively use shadowsocks for only some applications instead of a global proxy using the `per app proxy` setting. This works well if you only have a few apps that need tunnelling.

There! you can now enjoy unrestricted internet, even on your phone!
Visit the [shadowsocks wiki](https://github.com/shadowsocks/shadowsocks/wiki) page to find more about configuration options, optimization, and more advanced features.
