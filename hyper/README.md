# hyper

This folder contains some resources for getting started using the Spell hyperparameter search feature.

## An introduction to hyperparameter search with CIFAR10

![](https://i.imgur.com/ewmZvLw.png)

This tutorial showcases hyperparameter searches on Spell from the CLI. This is the code compliment to the post ["An introduction to hyperparameter search with CIFAR10"](https://spell.run/blog/an-introduction-to-hyperparameter-search-with-cifar10-Xo8_6BMAACEAkwVs) on our blog.

To follow along with this tutorial in code, you will need to have:

* The `spell`, `pandas`, and `keras` Python packages.
* An account on [Spell for Team](https://spell.run/pricing).

To get started, open the `hyperparameter-search.ipynb` notebook. Alternatively, you can launch this notebook from a Spell workspace by running the following CLI command (requires having the `spell` package installed):

```bash
spell jupyter \
    --lab \
    --github-url https://github.com/spellrun/spell-examples.git \
    hyper-demo-workspace
```
