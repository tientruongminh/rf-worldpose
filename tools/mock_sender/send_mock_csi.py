#!/usr/bin/env python3
from __future__ import annotations
import argparse, socket, struct, time, zlib, math
MAGIC = 0xC5110001
HEADER_LEN = 32
PROTO = 1

def build_packet(node_id:int, seq:int, n_sub:int=56, channel:int=6, rssi:int=-50, fw:int=1) -> bytes:
    timestamp_us = int(time.time() * 1_000_000)
    iq=[]
    for i in range(n_sub):
        iq.extend([int(20*math.sin(i/5)), int(20*math.cos(i/7))])
    payload = b''.join(struct.pack('<h', x) for x in iq)
    header = struct.pack('<IBBHQbbBBHHI', MAGIC, PROTO, node_id, HEADER_LEN, seq, timestamp_us, rssi, -90, channel, 0, n_sub, fw, len(payload))
    crc = zlib.crc32(header + payload) & 0xffffffff
    return header + payload + struct.pack('<I', crc)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=5006)
    ap.add_argument('--node-id', type=int, default=1)
    ap.add_argument('--count', type=int, default=100)
    ap.add_argument('--hz', type=float, default=20)
    args=ap.parse_args()
    sock=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    delay=1.0/args.hz
    for seq in range(args.count):
        sock.sendto(build_packet(args.node_id, seq), (args.host,args.port))
        time.sleep(delay)
    print(f"sent {args.count} packets to {args.host}:{args.port}")
if __name__=='__main__': main()
