# Spell Workflow Example: Video Generation
This is an example of how to use the Spell Python API and a workflow to train a [pix2pix](https://github.com/affinelayer/pix2pix-tensorflow) model and generate videos with it.

For more information regarding Spell, the Python API, runs, and workflows,
check out the [Spell documentation](https://spell.run/docs).

To run this code you first need to [sign up for a Spell account](https://web.spell.run/register)
and install Spell: `pip install spell`.

# Introduction
Pix2pix is an image-to-image transformation method. Given a video, we can train a pix2pix model to predict the next frame based on current frame. After training, such a model could be used to generate a video starting from only one frame. In the generation process, every newly created frame would be fed back to the model to predict the next one.

# Run it

1. Clone the [Video Generation](https://github.com/chengz3906/video-generation) repo:
```ShellSession
$ git clone https://github.com/chengz3906/video-generation.git
```

2. Clone this repo:
```ShellSession
$ git clone https://github.com/spellrun/spell-examples.git
```

3. Run the workflow:
```ShellSession
$ cd spell-examples
$ spell workflow --repo video-gen=../video-generation/ "python workflow.py --video video/fireworks.mp4"
```
Notice: 
*The above command uses a sample video. But you can use your own video, which requires you add a new command to download it at the first run of the workflow. 
*You can specify other optional settings apart from the video path. Check `python workflow.py -h`.

4. Download generated video from the latest run on the web console or:
```ShellSession
$ spell cp runs/<RUN_ID>/gen.mp4
```

# Details

The file `workflow.py` is a python script that:
1. Transform fireworks.mp4 into a set of images, where each image is a combination of 2 consecutive frames.
2. Split the images into train and validation sets.
3. Randomly select a frame from the val set as the starting point for video generation.
4. Train the pix2pix model with this dataset.
5. Generate frames with trained model.
6. Transform these frames into a video.

Notice:
The [video generation](https://github.com/chengz3906/video-generation) repo is forked from [pix2pix-tensorflow](https://github.com/affinelayer/pix2pix-tensorflow), which has no built-in video generation method. Current solution is to run `pix2pix.py` once for each single frame, which is relatively slow. In experiments, it costs about 20 seconds to generate a frame on K80.
