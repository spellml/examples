# spark

This folder contains some recipes for using Spell with a Spark cluster like Databricks or AWS EMR.

## Using Spark for model featurization with Spell

This tutorial shows how you can combine Apache Spark, Amazon S3, and Spell Workspaces to build model training features at scale in Spark, then train models using them in a Spell workspace. This is a code compliment to the post ["Using Spark for model featurization with Spell"](https://spell.run/blog/using-spark-for-model-featurization-with-spell-XnEedBUAACcAjfTV) on our blog.

To follow along with this tutorial in code, you will need:

* A Databricks Community account.
* A Kaggle account.
* An account on Spell for Team.

If you don't already have a cluster set up, the easiest way to try out Spark is to sign up for a free community account on [Databricks](https://databricks.com/). This will give you access to a free community version of Databricks platform, including access to [Databricks Notebooks](https://docs.databricks.com/notebooks/index.html). Once you've created an account, use the import feature to upload the `spark-featurizing-demo.ipynb` notebook in this folder to Databricks:

![](https://i.imgur.com/Cndnno5.png)

You will also need to have a (free) [Kaggle](https://www.kaggle.com/) account. Once you have made one go to the [Liberty Mutual Property Inspection competition page](https://www.kaggle.com/c/liberty-mutual-group-property-inspection-prediction), download and unzip `train.csv`, and upload it to the Databricks notebook (using the `Data` tab in the sidebar).
