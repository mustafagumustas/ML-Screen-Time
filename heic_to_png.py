import os
from PIL import Image
import pillow_heif
import shutil
import time

__author__ = "Mustafa Gumustas"
"""This program convert iphone photo extention HEIC into PNG format. There is a problem with output right now, some of them are broken. I will fix that later"""

for root, _, files in os.walk("/Users/mustafagumustas/Downloads/mustafa_pictures"):
    for pic in files:
        if "HEIC" in pic:
            name = pic.split(".")[0]
            heif_file = pillow_heif.read_heif(root + "/" + pic)
            image = Image.frombytes(
                heif_file.mode,
                heif_file.size,
                heif_file.data,
                "raw",
            )
            image = image.rotate(270)
            # time.sleep(0.5)
            image.save(root + "/" + "png" + "/" + name + ".png", format="png")
            shutil.move(root + "/" + pic, root + "/saved/" + pic)
            print("saved")
