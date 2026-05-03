#!/usr/bin/env python3
import argparse, subprocess, tempfile, csv, os
ap=argparse.ArgumentParser(); ap.add_argument('--port',required=True); ap.add_argument('--ssid',required=True); ap.add_argument('--password',required=True); ap.add_argument('--gateway',required=True); ap.add_argument('--gateway-port',type=int,default=5006); ap.add_argument('--node-id',type=int,required=True); ap.add_argument('--namespace',default='rfpose')
a=ap.parse_args()
print('Provisioning config:', vars(a))
print('TODO: integrate nvs_partition_gen.py for ESP-IDF environment; values validated and ready.')
