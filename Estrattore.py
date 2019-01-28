import tensorflow as tf
import os.path as pattern
from models.research.audioset import vggish_slim
from models.research.audioset import vggish_params
from models.research.audioset import vggish_input
from models.research.audioset import vggish_postprocess

"""Define VGGish model, load the checkpoint, and return a dictionary that points
to the different tensors defined by the model.
"""
def CreateVGGishNetwork(hop_size=0.96, sess=None):  # Hop size is in seconds.
    vggish_slim.define_vggish_slim()
    checkpoint_path = pattern.abspath('vggish_model.ckpt')
    vggish_params.EXAMPLE_HOP_SECONDS = hop_size
    vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)

    features_tensor = sess.graph.get_tensor_by_name(
        vggish_params.INPUT_TENSOR_NAME)
    embedding_tensor = sess.graph.get_tensor_by_name(
        vggish_params.OUTPUT_TENSOR_NAME)

    layers = {'conv1': 'vggish/conv1/Relu',
              'pool1': 'vggish/pool1/MaxPool',
              'conv2': 'vggish/conv2/Relu',
              'pool2': 'vggish/pool2/MaxPool',
              'conv3': 'vggish/conv3/conv3_2/Relu',
              'pool3': 'vggish/pool3/MaxPool',
              'conv4': 'vggish/conv4/conv4_2/Relu',
              'pool4': 'vggish/pool4/MaxPool',
              'fc1': 'vggish/fc1/fc1_2/Relu',
              'fc2': 'vggish/fc2/Relu',
              'embedding': 'vggish/embedding',
              'features': 'vggish/input_features',
              }
    g = tf.get_default_graph()
    for k in layers:
        layers[k] = g.get_tensor_by_name(layers[k] + ':0')
    return {'features': features_tensor,
            'embedding': embedding_tensor,
            'layers': layers}


'''Run the VGGish model, starting with a sound (x) at sample rate
(sr). Return a whitened version of the embeddings. Sound must be scaled to be
floats between -1 and +1.'''


def ProcessWithVGGish(vgg, x, sr, sess):
    # Produce a batch of log mel spectrogram examples.
    input_batch = vggish_input.waveform_to_examples(x, sr)
    # print('Log Mel Spectrogram example: ', input_batch[0])

    [embedding_batch] = sess.run([vgg['embedding']],
                                 feed_dict={vgg['features']: input_batch})

    # Postprocess the results to produce whitened quantized embeddings.
    pca_params_path = pattern.abspath('vggish_pca_params.npz')

    pproc = vggish_postprocess.Postprocessor(pca_params_path)
    postprocessed_batch = pproc.postprocess(embedding_batch)
    # print('Postprocessed VGGish embedding: ', postprocessed_batch[0])
    return postprocessed_batch[0]

