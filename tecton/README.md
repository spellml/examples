# tecton <a href="https://web.spell.ml/workspace_create?workspaceName=tecton-demo&githubUrl=https%3A%2F%2Fgithub.com%2Fspellml%2Fexamples&pip=https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Ftecton.ai.public%2Fpip-repository%2Fitorgation%2Ftecton%2Ftecton-latest-py3-none-any.whl"><img src=https://spell.ml/badge.svg height=20px/></a>

This demo shows off how a Spell model server can be instrumented to draw data from a [Tecton](https://arize.com/) data service.

What it looks like on Spell:

![](https://i.imgur.com/PX8qiaU.png)

What it looks like on Tecton:

![](https://i.imgur.com/C22aqh8.png)

To try this code yourself (requires accounts on both Spell and Tecton):

```bash
$ spell server serve \
  --node-group default \
  --min-pods 1 --max-pods 3 \
  --target-requests-per-second 10 \
  --pip https://s3-us-west-2.amazonaws.com/tecton.ai.public/pip-repository/itorgation/tecton/tecton-latest-py3-none-any.whl \
  --env TECTON_API_KEY="$TECTON_API_KEY" \
  --env TECTON_API_SERVICE="$TECTON_API_SERVICE" \
  --env TECTON_FEATURE_SERVICE="$TECTON_FEATURE_SERVICE" \
  tecton-demo:v1 serve.py
$ curl -d '{"ad_id": "5417", "user_uuid": "6c423390-9a64-52c8-9bb3-bbb108c74198"}' -X POST https://us-west-2.external-aws.spell.services/external-aws/tecton-demo/predict
{"target":0}%
```

For more information refer to our blog post.
