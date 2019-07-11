# Keras examples

This directory contains an example model trained on mnist and one on cifar10. These are very similar to the examples from the keras repo, however we've added command line arguments for many of the model parameters. This makes it easy to vary these and experiement with parameter optimization. As examples you can run the follow on Spell

spell hyper random \
    -t K80 --pip idx2numpy \
    --param conv3-filters=4:64:linear:int \
    --param dense-size=10:100:linear:int \
    --param dropout=0.001:0.999:linear:float \
    -- python mnist.py --conv3-filters :conv3-filters: --dense-size :dense-size: --dropout :dropout:

spell hyper bayesian \
    -t K80 \
    --metric keras/val_acc --metric-agg last \
    --param conv2_filter=16:128:linear:int \
    --param conv2_kernel=2:8:linear:int \
    --param dense_layer=64:1024:linear:int \
    --param dropout_3=0.001:0.999:linear:float \
    -- python cifar10_cnn.py --epochs 25 --conv2_filters :conv2_filters: --conv2_kernal :conv2_kernal: --dense_layer :dense_layer: --dropout_3 :dropout_3:
