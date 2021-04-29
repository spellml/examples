# workflows

This folder contains a tutorial and some example scripts covering [Spell workflows](http://spell.run/docs/workflow_overview/).

## Tutorial <a href="https://web.spell.ml/workspace_create?workspaceName=workflows-demo&githubUrl=https%3A%2F%2Fgithub.com%2Fspellml%2Fexamples"><img src=https://spell.ml/badge.svg height=20px/></a>

![](https://i.imgur.com/W5Ugs0S.png)

For users new to workflows, we recommend starting with the `workflows-demo.ipynb` demo notebook.

```bash
spell jupyter \
    --lab \
    --github-url https://github.com/spellml/examples.git \
    workflows-demo-workspace
```

## Batch scoring on big data

![](https://i.imgur.com/97Yz8kp.png)

The blog post ["Batch scoring on big data using scikit-learn and Spell workflows"](https://spell.ml/blog/batch-scoring-on-big-data-using-scikit-learn-and-spell-X4YjZBEAACQAHtJw) is a complete end-to-end example of an application of Spell workflows to a batch scoring job.

## Other examples

This folder also contains two other workflow examples.

The `video-generation-workflow` folder contains an example workflow generating videos using the [pix2pix](https://github.com/affinelayer/pix2pix-tensorflow) model.

The `with-metrics` folder contains an example workflow leveraging the [Spell metics API](http://spell.run/docs/metrics/).
