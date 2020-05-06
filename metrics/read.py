import pandas as pd
import spell.client

client = spell.client.from_environment()

# replace with the actual run id value
RUN_ID = 123
run = client.runs.get(RUN_ID)

# we return the metrics data as a generator
metric = run.metrics("value")

df = pd.DataFrame(metric, columns=["timestamp", "index", "value"])
df.to_csv('metrics.csv')