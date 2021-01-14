# ludwig

Ludwig is an open-source AutoML toolkit. Ludwig was originally developed internally within Uber. [It was open sourced in February 2019](https://eng.uber.com/introducing-ludwig/), and is presently under incubation with [the Linux Foundation](https://lfaidata.foundation/).

This folder is a small demo showing how Ludwig can be run on Spell. We recommend Ludwig to all of our users looking to do AutoML on the platform.

The `ludwig-demo.ipynb` Jupyter notebook contains the code. You can launch this notebook on Spell by running the following CLI command (requires having the `spell` package installed):

```python
spell jupyter \
    --lab \
    --github-url https://github.com/spellml/examples.git \
    ludwig-demo-workspace
```

## An introduction to AutoML with Ludwig

![](https://i.imgur.com/YazB1Hn.png)

This blog post, ["An introduction to AutoML with Ludwig"](https://spell.ml/blog/an-introduction-to-automl-with-ludwig-X_OSWhAAACMA6eYD) discusses Ludwig and AutoML in some detail.
