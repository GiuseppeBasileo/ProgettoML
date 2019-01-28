import Estrattore as Est
import numpy as np
import tensorflow as tf

def main():
    tf.reset_default_graph()
    sess = tf.Session()

    vgg = Est.CreateVGGishNetwork(0.01, sess)
    # Generate a 1 kHz sine wave at 44.1 kHz (we use a high sampling rate
    # to test resampling to 16 kHz during feature extraction).
    num_secs = 3
    freq = 1000
    sr = 44100
    t = np.linspace(0, num_secs, int(num_secs * sr))
    x = np.sin(2 * np.pi * freq * t)  # Unit amplitude input signal

    postprocessed_batch = ProcessWithVGGish(vgg, x, sr)

    # print('Postprocessed VGGish embedding: ', postprocessed_batch[0])
    expected_postprocessed_mean = 123.0
    expected_postprocessed_std = 75.0
    np.testing.assert_allclose(
        [np.mean(postprocessed_batch), np.std(postprocessed_batch)],
        [expected_postprocessed_mean, expected_postprocessed_std],
        rtol=rel_error)


if __name__ == "__main__":
    main()