import tensorflow as tf
import src.transform as transform
from src.utils import get_img
import numpy as np
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

g = tf.Graph()
curr_num = 0
soft_config = tf.ConfigProto(allow_soft_placement=True)
soft_config.gpu_options.allow_growth = True
with g.as_default(), tf.Session(config=soft_config) as sess:
    img_placeholder = tf.placeholder(tf.float32, shape=(1, 256, 256, 3),
                                     name='img_placeholder')

    preds = transform.net(img_placeholder)
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state('ckpt')
    saver.restore(sess, ckpt.model_checkpoint_path)
    # Load weights
    img = get_img('images/input/input_italy.jpg', (256, 256, 3))
    print(type(img))
    X = np.zeros((1, 256, 256, 3), dtype=np.float32)
    X[0] = img
    _preds = sess.run(preds, feed_dict={img_placeholder: X})
    # frozen_graph_def = tf.graph_util.convert_variables_to_constants(
    #     sess,
    #     sess.graph_def,
    #     ['add_37'])
    #
    # # Save the frozen graph
    # with open('output_graph.pb', 'wb') as f:
    #     f.write(frozen_graph_def.SerializeToString())

    graph_def = tf.GraphDef()
    builder = tf.saved_model.builder.SavedModelBuilder(
        './saved_model/00000123')
    input = g.get_tensor_by_name('img_placeholder:0')
    output = g.get_tensor_by_name('add_37:0')

# with tf.gfile.GFile('output_graph.pb', "rb") as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())

    sigs = {}
    # graoh_def = tf.GraphDef()
    #
    #
    # tf.import_graph_def(graph_def, name="")
    # g = tf.get_default_graph()

    sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
        tf.saved_model.signature_def_utils.predict_signature_def(
            {"in": input}, {"out": output})

    builder.add_meta_graph_and_variables(sess,
                                         [tag_constants.SERVING],
                                         signature_def_map=sigs)

    builder.save()
