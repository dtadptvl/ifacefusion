#!/usr/bin/env python3
import struct,zlib,os
W=H=1024
raw=bytearray()
for y in range(H):
    raw.append(0)
    for x in range(W):
        # Deep indigo to cyan radial field, with a white two-face fusion mark.
        dx=(x-512)/512; dy=(y-512)/512
        t=min(1,(dx*dx+dy*dy)**.5)
        r=int(24+22*(1-t)); g=int(35+115*(1-t)); b=int(76+130*(1-t))
        # two overlapping face rings + central fusion seam
        d1=((x-410)**2+(y-490)**2)**.5; d2=((x-614)**2+(y-490)**2)**.5
        ring=(225<d1<255) or (225<d2<255)
        seam=abs(x-512)<16 and 330<y<700
        if ring or seam: r=g=b=245
        raw.extend((r,g,b,255))
def chunk(t,d):
    return struct.pack(">I",len(d))+t+d+struct.pack(">I",zlib.crc32(t+d)&0xffffffff)
png=b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR",struct.pack(">IIBBBBB",W,H,8,6,0,0,0))+chunk(b"IDAT",zlib.compress(bytes(raw),9))+chunk(b"IEND",b"")
path="iFaceFusion/Assets.xcassets/AppIcon.appiconset/AppIcon-1024.png"
os.makedirs(os.path.dirname(path),exist_ok=True)
open(path,"wb").write(png)
print(path)
