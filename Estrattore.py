import tensorflow as tf
import os.path as pattern
from models.research.audioset import vggish_slim
from models.research.audioset import vggish_params
from models.research.audioset import vggish_input


def CreateVGGishNetwork(hop_size=0.96, sess=None):  # Hop size is in seconds.
    """Define VGGish model, load the checkpoint, and return a dictionary that points
    to the different tensors defined by the model.
    """
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
            'layers': layers,
            }