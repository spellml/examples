# Spell Workspace Example: deepDream

This is an example of how to use Jupyter Notebook using Spell - adapted from https://github.com/google/deepdream

For more information on deep dream, check out the blogpost https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html

For more information regarding Spell, the Python API, runs, and workflows,
check out the [Spell documentation](https://spell.run/docs).

To run this code you first need to [sign up for a Spell account](https://web.spell.run/register)
and install Spell: `pip install spell`.

# Introduction
Google's DeepDream is a computer vision program created by Alexander Mordvintsev which uses a CNN to find and enhance patterns in images.

It uses the Caffe neural network framework trained on the ImageNet dataset to produce "dream visuals".

We will use it today to attempt to turn a picture of dinosaurs into flowers (inspiration https://twitter.com/chrisrodley/status/875266719660482560)

# Steps

### 1. Create a Workspace
Learn more at https://spell.run/docs/workspaces_overview

Initialize from the git repo https://github.com/spell-examples/deepdream

Make sure to select Caffe as the framework and K80 as the machine type

### 2. Run the Jupyter Notebook!
That's it - your notebook is now running on Spell!

Feel free to experiment with your own images by uploading them to the Notebook!

Other Resources:
* Feel free to experiment with other models from Caffe [Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo).
* Why does DeepDream see dogs? Learn more at [Seeing Dogs](https://www.fastcompany.com/3048941/why-googles-deep-dream-ai-hallucinates-in-dog-faces)
