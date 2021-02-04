# arize <a href="https://web.spell.ml/workspace_create?workspaceName=arize-demo&githubUrl=https%3A%2F%2Fgithub.com%2Fspellml%2Fexamples&pip=lightgbm,arize"><img src=https://spell.ml/badge.svg height=20px/></a>

This demo shows off how a Spell model server can be instrumented using [Arize](https://arize.com/) for model logging, tracking, and drift detection.

```bash
# First, train the model in a Spell run.
$ spell run \
    --github-url https://github.com/spellml/examples \
    --machine-type cpu \
    --mount public/tutorial/churn_data/:/mnt/churn_prediction/ \
    --pip arize --pip lightgbm \
    -- python arize/train.py
# Register the model artifact as a model with Spell.
$ spell model create churn-prediction runs/$RUN_ID
# Serve the model!
$ spell server serve \
  --node-group default \
  --min-pods 1 --max-pods 3 \
  --target-requests-per-second 100 \
  --pip lightgbm --pip arize \
  --env ARIZE_ORG_KEY=$ARIZE_ORG_KEY \
  --env ARIZE_API_KEY=$ARIZE_API_KEY \
  churn-prediction:v1 serve_sync.py  # or serve_async.py
# Once the model is up, test it using curl.
curl -X POST -d '@test_payload.txt' \
    https://$REGION.$CLUSTER.spell.services/$ORGANIZATION/churn-prediction/predict
```

`serve_sync.py` is thea version of this model server that logs synchronously. `serve_async.py` is a higher-performance version of this entrypoint that logs asynchronously using [Starlette background tasks](https://spell.ml/docs/model_servers/).

For more information refer to our forthcoming blog post.
