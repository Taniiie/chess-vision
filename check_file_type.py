"""Quick check: what file format is this really, regardless of its extension?"""

with open("chessboard1.jpg", "rb") as f:
    header = f.read(12)

print("First 12 bytes (hex):", header.hex())

if header.startswith(b'\xff\xd8\xff'):
    print("This IS a real JPEG file.")
elif header.startswith(b'\x89PNG'):
    print("This is actually a PNG file, just named .jpg")
elif header[8:12] == b'WEBP':
    print("This is actually a WEBP file, just named .jpg")
elif header.startswith(b'<') or header.startswith(b'<!DOCTYPE') or b'html' in header.lower():
    print("This looks like an HTML file, not an image at all (probably a failed/blocked download).")
else:
    print("Unknown format - doesn't match common image signatures.")
