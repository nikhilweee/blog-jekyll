---
layout: post
title: Set up a Ghost blog
subtitle: from scratch, in under 30 minutes!
tags: [supervisor, nginx, node, digitalocean, linux]
categories: [hacks, tutorial]
---
#### Before you read
This is *not* an in-depth tutorial for a total newbie. Something like that would make this a really long post. Though I've given one-liners explaining each command,
It would be better if you had some experience working with the linux terminal and a basic understanding of how web content is served.

### Things you'll need
Just the two of them will do

* A **Server**
: This is where your site contents will be stored. It's just an internet connected computer that'll stay awake 24x7 for your blog and will serve users with your blog as and when asked to so.

* A **domain**
: No one likes remembering IP addresses. You'll need this to give your blog a name.

### 1. Buy a server
You can buy a server online from your favourite provider. I suggest using [DigitalOcean](https://www.digitalocean.com/?refcode=ad1b7e083b2e), where you can spin up a server for just $5 per month.
To get $10 in free credits, use [this link](https://www.digitalocean.com/?refcode=ad1b7e083b2e):

    https://www.digitalocean.com/?refcode=ad1b7e083b2e

- [Sign up](https://www.digitalocean.com/?refcode=ad1b7e083b2e) for an account and enter your credit card details or pay via PayPal.
- A `$5` droplet with an `Ubuntu` image at your nearest datacenter would be enough for a blog. No need for any `additional options` or `SSH Keys` for now.

### 2. Set up your server
Now that you've bought your own server, its time to set it up. Use the following commands from your terminal.
Consult these [community tutorials](https://www.digitalocean.com/community/tutorial_series/new-ubuntu-14-04-server-checklist) for more details.

* [**Connect to your server via ssh**](https://www.digitalocean.com/community/tutorials/how-to-connect-to-your-droplet-with-ssh)

The server IP and root password can be found in your email when you set up your droplet.

    $ ssh root@<server IP>

### 3. Secure your server
These steps are not necessary, but *strictly recommended*.
I have skipped explaining them for brevity. Consult the DigitalOcean [community tutorials](https://www.digitalocean.com/community/tutorial_series/new-ubuntu-14-04-server-checklist) which explains all these steps in detail.

* **Create a new user and restrict it**. Why risk root access in case your server gets hacked?
* **Configure passwordless ssh**. So that only your computer can login to your server.
* **Disable root login**. You wouldn't want someone gaining root access to your server. Why not disable it in the first place?
* **Set up firewall** so that only specified ports remain open.

### 4. Set up your domain
I'll cover how to buy and point your domain to your server here.

* **Buy a domain**. You can use the [free ones](http://freenom.com/), or spend money for the popular ones. GoDaddy and NameCheap are reliable providers.
* **Use DigitalOcean nameservers**. This is optional, but lets you manage your server and domain from the same place. Edit your domain details to configure the nameservers to the following

```
ns1.digitalocean.com
ns2.digitalocean.com
ns3.digitalocean.com
```
* **Connect domain to droplet**. Go to `Networking > Domains` tab in DigitalOcean settings. Connect your domain with the droplet by entering your domain, say `nikhilweee.me`, selecting your droplet from the dropdown and hitting *Create Record*. Your domain should now point to your droplet IP.

### 5. Set up Ghost
Now that we've connected the domain and the droplet, it's time to set up your Ghost blog.

* **Install and update `npm` and `node`**.

Ghost is written in javascript and needs node and npm to build and run on the server.

The first command installs node and npm, the second one upgrades npm, and the third one is required for ubuntu, so that `node` is equivalent to `nodejs` in the terminal.

```
$ apt-get install nodejs npm
$ npm install -g npm
$ ln -s "$(which nodejs)" /usr/bin/node

```
* **Install Ghost**.

We'll download the latest version, extract and then install it to the recommended install directory.

We'll first install the tools `zip` and `wget` needed to download and extract the archive, then create the recommended directory `/var/www/`, and then download and extract the latest version of ghost. The following commands do just that.

```
$ sudo apt-get update
$ sudo apt-get install zip wget
$ sudo mkdir -p /var/www/
$ cd /var/www
$ sudo wget https://ghost.org/zip/ghost-latest.zip
$ sudo unzip -d ghost ghost-latest.zip
$ cd ghost/
$ sudo npm install --production
```
* **Configure Ghost**

Ghost stores your blog url in a `config.js` file. We'll first create the file using a default template and then set the appropriate url. Open the config file  by using the following commands, and then under `config > production`, change the url to your new domain.

```
$ sudo cp config.example.js config.js
$ sudo nano config.js
```

### 6. Set up Nginx
Nginx (read Engine-X) is the server software (that technically makes your droplet a server) that we'll use to serve our blog. We won't use the ghost development server as that isn't a good practice.

* **Install nginx**

It's our favourite `apt-get` command again.

```
$ sudo apt-get install nginx
```
* **Configure nginx**

We'll create a configuration file for your site in the  `sites-available` directory.

```
$ cd /etc/nginx/
$ sudo touch /etc/nginx/sites-available/ghost
$ sudo nano /etc/nginx/sites-available/ghost
```
Now edit the config file. The following snippet is the simplest configuration for a working ghost blog. We are simply running your blog behind a proxy server. This way we get the robustness of nginx along with a simple config.
Paste this code into the editor. Replace `<domain.tld>` with your actual domain.

```
server {
    listen 80;
    server_name <domain.tld>;
    location / {
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_set_header   Host      $http_host;
        proxy_pass         http://127.0.0.1:2368;
    }
}
```
* **Enable your site**

Simply symlink the file you just created to the `sites-enabled` directory and restart nginx.

```
$ sudo ln -s /etc/nginx/sites-available/ghost /etc/nginx/sites-enabled/ghost
$ sudo service nginx restart
```
* At this stage, test your config

```
$ cd /var/www/ghost
$ sudo npm start --production
```
You should be able to access your new ghost blog on your new domain. If not, there's been a mistake. C'mon. Get working. Fix that crap!

### 7. Set up autostart

Now that your blog is set up, we just finally need to ensure that the blog doesn't stop for any reason and even autostarts on droplet restart. I'll do this using supervisor.

* **Install supervisor**

Again, `apt-get`.

```
$ sudo apt-get install supervisor
```
* **Configure your blog for supervisor**

We'll again create a config file as we did for nginx.

```
$ sudo nano /etc/supervisor/conf.d/ghost.conf
```
Now edit the file to create an autostart process so that your blog remains stable.
`user` should be an existing user on the system. Better to use a dedicated user as a safety measure as described in `4. Secure your server` above.

```
[program:ghost]
command = node /var/www/ghost/index.js
directory = /var/www/ghost
user = ghost
autostart = true
autorestart = true
stdout_logfile = /var/log/supervisor/ghost.log
stderr_logfile = /var/log/supervisor/ghost_err.log
environment = NODE_ENV="production"
```
* **Restart**

```
$ sudo supervisorctl reread
$ sudo supervisorctl update
$ sudo supervisorctl restart ghost
```
Now your blog is all set and would even withstand restarts. Yay.
