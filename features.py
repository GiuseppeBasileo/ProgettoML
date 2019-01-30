import numpy as np
import tensorflow as tf

def estrattore_features(dataset):
    x_list=[]
    Y_list=[]
    for elem in dataset:
        x_elem,labels=estrattore_features_tensor(elem)
        x_list.append(x_elem)
    x=np.array(x_elem)
    Y=np.array(Y_list)
    return x,labels


def estrattore_features_tensor(element):
    labels_list = []
    x_list=[]
    for example in tf.python_io.tf_record_iterator(element):
        tf_example = tf.train.Example.FromString(example)
        labels_list.append(tf_example.features.feature['labels'].int64_list.value)
        tf_seq_example = tf.train.SequenceExample.FromString(example)
        sess = tf.InteractiveSession()
        n_frames = len(tf_seq_example.feature_lists.feature_list['audio_embedding'].feature)
        x_elem_list = []
        for i in range(n_frames):
            x_elem_list.append(tf.decode_raw(
                tf_seq_example.feature_lists.feature_list['audio_embedding'].feature[i].bytes_list.value[0],
                tf.uint8).eval())
        sess.close()
        x_list.append(x_elem_list)
    x=np.array(x_list)
    labels=np.array(labels_list)
    return x,labels