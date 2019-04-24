import argparse
from facenet_sandberg import facenet
import tensorflow as tf
import os
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='face model test')
parser.add_argument('--model', default='./20180402-114759', help='path to load model.')
parser.add_argument('--image_size', type=int, help='Image size (height, width) in pixels.', default=160)
args = parser.parse_args()


# print(args.model)

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


if __name__ == "__main__":
    with tf.Graph().as_default():
        with tf.Session() as sess:
            facenet.load_model(args.model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            if not os.path.exists("./embedding"):
                os.makedirs("./embedding")

            for r_dir, s_dir, files in os.walk("./data_align"):
                if len(s_dir) > 0:
                    for path in s_dir:
                        if not os.path.exists("./embedding/" + path):
                            os.makedirs("./embedding/" + path)
                    continue
                for file in files:
                    file_name, file_ext = os.path.splitext(file)
                    path_file = "/".join([r_dir, file])
                    img = cv2.imread(path_file)
                    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    img = prewhiten(img)
                    img = np.expand_dims(img, axis=0)
                    feed_dict = {images_placeholder: img, phase_train_placeholder: False}
                    embed = sess.run(embeddings, feed_dict=feed_dict)
                    np.save("{}/{}.npy".format(r_dir.replace("data_align", "embedding"), file_name), embed)
