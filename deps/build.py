import tensorflow as tf
import os

dirpath = os.path.split(tf.__file__)[0]
if "libtensorflow_framework.so.1" in os.listdir(dirpath) and \
    "libtensorflow_framework.so" not in os.listdir(dirpath):
    os.symlink(os.path.join(dirpath, "libtensorflow_framework.so.1"),
                os.path.join(dirpath, "libtensorflow_framework.so")) 
    print("""shared library path = {}""".format(os.path.join(dirpath, "libtensorflow_framework.so")))
