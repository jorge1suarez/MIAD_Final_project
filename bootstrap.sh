#!/usr/bin/env bash

# ssh access
cat id_rsa.pub >> /home/vagrant/.ssh/authorized_keys
rm -rf id_rsa.pub

# copy hosts
sudo cat hosts >> /etc/hosts
rm -rf hosts
