import os
import struct

import numpy as np
import pandas as pd
from pyntcloud.io.pcd import parse_header, build_dtype


def read_pcd(filename):
    data = {}
    with filename.open('rb') as f:
        header = []
        while True:
            ln = f.readline().strip().decode()
            header.append(ln)
            if ln.startswith('DATA'):
                metadata = parse_header(header)
                dtype = build_dtype(metadata)
                break

        if metadata['data'] == 'ascii':
            pc_data = np.loadtxt(f, dtype=dtype, delimiter=' ')

        elif metadata['data'] == 'binary':
            rowstep = metadata['points'] * dtype.itemsize
            # for some reason pcl adds empty space at the end of files
            buf = f.read(rowstep)

            pc_data = np.fromstring(buf, dtype=dtype)  # type: ignore[misc]

        elif metadata['data'] == 'binary_compressed':
            raise NotImplementedError("Go ask PCD why they use lzf compression.")
            # compressed size of data (uint32)
            # uncompressed size of data (uint32)
            # compressed data
            # junk
            fmt = 'II'
            compressed_size, uncompressed_size =\
                struct.unpack(fmt, f.read(struct.calcsize(fmt)))
            compressed_data = f.read(compressed_size)
            # TODO what to use as second argument? if buf is None
            # (compressed > uncompressed)
            # should we read buf as raw binary?
            #buf = lzf.decompress(compressed_data, uncompressed_size)
            if len(buf) != uncompressed_size:
                raise Exception('Error decompressing data')
            # the data is stored field-by-field
            pc_data = np.zeros(metadata['width'], dtype=dtype)
            ix = 0
            for dti in range(len(dtype)):
                dt = dtype[dti]
                bytes = dt.itemsize * metadata['width']
                column = np.fromstring(buf[ix:(ix + bytes)], dt)
                pc_data[dtype.names[dti]] = column
                ix += bytes

    df = pd.DataFrame(pc_data)

    # check if dataframe contains color info
    col = 'rgb'
    if col in df.columns:
        # get the 'rgb' column from dataframe
        packed_rgb = df.rgb.values
        # 'rgb' values are stored as float
        # treat them as int
        packed_rgb = packed_rgb.astype(np.float32).tostring()
        packed_rgb = np.frombuffer(packed_rgb, dtype=np.int32)
        # unpack 'rgb' into 'red', 'green' and 'blue' channel
        df['red'] = np.asarray((packed_rgb >> 16) & 255, dtype=np.uint8)
        df['green'] = np.asarray((packed_rgb >> 8) & 255, dtype=np.uint8)
        df['blue'] = np.asarray(packed_rgb & 255, dtype=np.uint8)
        # remove packed rgb since we don't need it anymore
        df.drop(col, axis=1, inplace=True)

    data['points'] = df
    return data