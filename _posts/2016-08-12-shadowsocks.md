---
layout: post
title: Shadowsocks
subtitle: Dont let your ISP block you
tags: [hacks, tutorial, linux]
category: [hacks]
published: false
---
Shadowsocks! Sounds like the new superhero after _Deadpool_. Well, in a way it _is_ a superhero! It let's you bypass censorship, even the GFW! [^gfw] Okay, so what's this fuss about? Before going further, I think it needs some introduction.

Shadowsocks is a SOCKS5 compatible proxy server/client suite which excels at being undetectable. It leaves no fingerprint even when you inspect its encrypted data. This makes Shadowsocks hard to detect by even the best firewalls. [^quora]

It will create a secured tunnel to the remote ShadowSocks server and set up a SOCKS proxy to bypass network blocks, which works like an SSH tunnel but more fast and efficient.


### Setup
If you explore the shadowsocks [site](https://shadowsocks.org), it's likely that you get lost in the technical description.

Shadowsocks has to be set up on two systems

- one is a server that would encrypt and forward packets thereby act as a proxy server
- and the other is the client that would benefit from the proxy.

Shadowsocks offers some unique advantages over other VPN or proxy solutions.

- It's open source - fellow developers rejoice!
- It's fast! Trust me!
- Cross - Platform. It has clients for almost every operating system - Linux, Mac, Windows, Android, iOS - you name it! It even supports OpenWRT so you could configure the whole of your router to use shadowsocks instead of individual devices
- You can choose what apps would use shadowsocks.

Here is the final version of the blog post on [Medium](https://medium.com/@wall2flower/shadowsocks-on-ubuntu-f2dace6870d4#.yfoc6jva3)

[^gfw]: The Great Firewall of China
[^quora]: Inspired by James Swineson's [answer](https://www.quora.com/What-is-shadowsocks/answer/James-Swineson) on Quora
