from AudioFunctionPersonal import AudioFunction as Audio
from models.research.audioset import vggish_input as input
import tarfile
import tensorflow as tf
import numpy as np

def main():
    with tarfile.open("C:/Users/GBasi/Desktop/features.tar.gz","r:gz") as tar:
        """subdir_and_files = [
            tarinfo for tarinfo in tar.getmembers()
            if tarinfo.name.startswith("audioset_v1_embeddings/bal_train")
        ]"""
        for example in tf.python_io.tf_record_iterator():
            result = tf.train.Example.FromString(example)
        print (example)
if __name__ == "__main__":
    main()