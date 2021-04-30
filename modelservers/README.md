# modelservers

Model servers allow you to deploy a trained model as a web API.

The [Model Servers](https://spell.ml/docs/model_servers) page in the docs for more information. If you are just getting started with Spell model servers, we recommend taking a look at [the Spell quickstart](https://spell.ml/docs/quickstart/) instead.

This directory contains a few simple demo scripts featuring model servers:

* `bert_squad`&mdash;a `AutoModelForQuestionAnswering` `transformers` model server for question-answer pairs.
* `cifar`&mdash;a simple `CIFAR10` example using `tensorflow` and `keras`.
* `hello_nvidia`&mdash;a model server that prints out `nvidia-smi` when hit, used for GPU testing.
* `resnet50`&mdash;a complex `resnet50` example using `tensorflow`.
* `simple_metrics`&mdash;a demo script demonstrating the [model server metrics](https://spell.ml/docs/model_servers/#viewing-and-writing-model-server-metrics) feature.
* `auth`&mdash;a demo script demonstrating using [HTTP Basic Authorization](https://developer.mozilla.org/en-US/docs/Web/HTTP/Authentication#security_of_basic_authentication) with a Spell model server.
