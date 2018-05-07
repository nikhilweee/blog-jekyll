---
layout: post
title: Using TexStudio with TexLive from WSL
subtitle: For those who want the best of both worlds.
tags: [texlive, texstudio, wsl]
category: [learn]
published: false
mathjax: false
---

Bash (Ubuntu) coming to Windows comes with a great deal of exciting use cases, but at the same interoperability still remains a painful issue. Here's a similar issue that took up most of by day before I finally figured out how to get it right. So I happen to use LaTeX every now and then for projects, reports and what not. Before switching back to Windows I already had my toolkit set up on Ubuntu. I used the TeXLive distribution for LaTeX and VSCode with the Latex Workshop plugin for editing. Everything was going on smoothly. But since I was exploring Bash on Windows (or WSL), I wanted to try out how LaTeX editing works out. Of course I could have installed TeXLive natively on Windows and used TeXStudio for editing, but since I was already used to Linux (apt install texlive <3), I was reluctant to change.

So I installed TeXLive on WSL as usual, and to my surprise I found out that WSL still uses Ubuntu 16.04, which means `apt-install texlive-base` would install the 2015 version of TeXLive. This posed some problems, so I tried upgrading to Ubuntu 17.10 before finally installing the 2017 version of TeXLive. Step 1, complete. Next, I installed TeXStudio through the usual install procedure.

Next, I knew that the Windows folks have done a really good job on interoperability, so you could actually run linux commands from the windows command line simply by prefixing `wsl`. So to setup TexStudio to use WSL binaries, I just prefixed `wsl` to all the commands listed in the settings, hoping that they would work. The solution, however, seems to be the following. 