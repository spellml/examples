# workflows <a href="https://web.spell.ml/workspace_create?workspaceName=workflows-demo&githubUrl=https%3A%2F%2Fgithub.com%2Fspellml%2Fexamples"><img src=https://spell.ml/badge.svg height=20px/></a>

This folder contains a tutorial and some example scripts covering [Spell workflows](http://spell.run/docs/workflow_overview/).

## Tutorial

![](https://i.imgur.com/W5Ugs0S.png)

Check out the `workflows-demo.ipynb` notebook to get started. You can run this tutorial on Spell using:

```bash
spell jupyter \
    --lab \
    --github-url https://github.com/spellml/examples.git \
    workflows-demo-workspace
```

## Other examples

This folder also contains two other workflow examples.

The `video-generation-workflow` folder contains an example workflow generating videos using the [pix2pix](https://github.com/affinelayer/pix2pix-tensorflow) model.

The `with-metrics` folder contains an example workflow leveraging the [Spell metics API](http://spell.run/docs/metrics/).
