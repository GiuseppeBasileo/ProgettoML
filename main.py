import models.research.audioset.vggish_input as input
import numpy as np
import tarfile
import tensorflow as tf
import os

tar = tarfile.open("features.tar.gz", "r:gz")
fileeval=list()
filebalan=list()
fileunbalan=list()
for member in tar.getmembers():
    if "tfrecord" in member.name:
        p = os.path.dirname(os.path.abspath(member.name))
        x=member.name.split('/')
        if "eval" in member.name:
            fileeval.append(p+'/'+x[2])
        elif "unbal_train" in member.name:
            fileunbalan.append(p+'/'+x[2])
        elif "bal_train" in member.name:
            filebalan.append(p+'/'+x[2])
dataset_eval=tf.data.TFRecordDataset(fileeval)
print(dataset_eval)
dataset_balan=tf.data.TFRecordDataset(filebalan)
dataset_unbalan=tf.data.TFRecordDataset(fileunbalan)
vid_ids = []
labels = []
audio_embedding = []
start_time_seconds = [] # in secondes
end_time_seconds = []
feat_audio = []
count = 0
for example in tf.python_io.tf_record_iterator(fileeval[0]):
    tf_example = tf.train.Example.FromString(example)
    vid_ids.append(tf_example.features.feature['video_id'].bytes_list.value[0].decode(encoding='UTF-8'))
    labels.append(tf_example.features.feature['labels'].int64_list.value)
    start_time_seconds.append(tf_example.features.feature['start_time_seconds'].float_list.value)
    end_time_seconds.append(tf_example.features.feature['end_time_seconds'].float_list.value)
    tf_seq_example = tf.train.SequenceExample.FromString(example)
    n_frames = len(tf_seq_example.feature_lists.feature_list['audio_embedding'].feature)

    sess = tf.InteractiveSession()
    rgb_frame = []
    audio_frame = []
    # iterate through frames
    for i in range(n_frames):
        audio_frame.append(tf.cast(tf.decode_raw(
            tf_seq_example.feature_lists.feature_list['audio_embedding'].feature[i].bytes_list.value[0], tf.uint8)
            , tf.float32).eval())

    sess.close()
    feat_audio.append([])

    feat_audio[count].append(audio_frame)
    count += 1

idx = 0 # test a random video

print('video ID',vid_ids[idx])
print('start_time:',np.array(start_time_seconds[idx]))
print('end_time:',np.array(end_time_seconds[idx]))

print('labels : ')
print(np.array(labels[idx]))
print('audio_frame: ')
print(audio_frame[idx])