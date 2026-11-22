"""Map blobs: zlib streams in the repo, stored raw in the image.

The maps are kept zlib-compressed in the repo (size limit). The image build
(rawmaps.py) writes each compressed blob again as RAW_MAGIC + the bytes that
zlib.decompress gives, so the run skips the decompression. unz() reads both forms
and returns the same bytes in both cases.
"""
import zlib

RAW_MAGIC = b"RAW\x00"  # never the start of a zlib stream (0x78 ...)


def unz(blob):
    if blob[:4] == RAW_MAGIC:
        return blob[4:]
    return zlib.decompress(blob)
