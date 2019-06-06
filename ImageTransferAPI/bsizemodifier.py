import tensorflow as tf
import src.transform as transform
from src.utils import get_img
import numpy as np
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants


# Loads Checkpoint from `ckpt` folder, converts it to a
# TensorFlow SavedModel, ready to serve.

g = tf.Graph()
soft_config = tf.ConfigProto(allow_soft_placement=True)
soft_config.gpu_options.allow_growth = True
with g.as_default(), tf.Session(config=soft_config) as sess:
    img_placeholder = tf.placeholder(tf.float32, shape=(1, 256, 256, 3),
                                     name='img_placeholder')

    preds = transform.net(img_placeholder)
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state('ckpt')
    saver.restore(sess, ckpt.model_checkpoint_path)
    # Load image (can use any image, just need one arbitrary run of model)
    img = get_img('images/input/input_italy.jpg', (256, 256, 3))
    X = np.zeros((1, 256, 256, 3), dtype=np.float32)
    X[0] = img
    # run
    _preds = sess.run(preds, feed_dict={img_placeholder: X})

    # If you want to freeze your graph instead of outputting a tensorflow
    # SavedModel, uncomment code and comment out all code below

    # frozen_graph_def = tf.graph_util.convert_variables_to_constants(
    #     sess,
    #     sess.graph_def,
    #     ['add_37'])
    #
    # # Save the frozen graph
    # with open('output_graph.pb', 'wb') as f:
    #     f.write(frozen_graph_def.SerializeToString())

    builder = tf.saved_model.builder.SavedModelBuilder(
        './saved_model/00000123') # DO NOT REMOVE THE NUMBER AT THE END OF THE PATH
    input = g.get_tensor_by_name('img_placeholder:0') #INPUT NODE
    output = g.get_tensor_by_name('add_37:0') #OUTPUT NODE
    sigs = {}
    sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
        tf.saved_model.signature_def_utils.predict_signature_def(
            {"in": input}, {"out": output})

    builder.add_meta_graph_and_variables(sess,
                                         [tag_constants.SERVING],
                                         signature_def_map=sigs)

    builder.save()
