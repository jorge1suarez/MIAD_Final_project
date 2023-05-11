#!/usr/bin/env bash

sudo mkdir -p /opt/postgres
sudo mkdir -p /data/postgres
sudo chown -R 999:999 /opt/postgres
sudo chown -R 999:999 /data/postgres

sudo timedatectl set-timezone America/Lima
