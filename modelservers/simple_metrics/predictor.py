from datetime import datetime
import json

from spell.serving import BasePredictor
import spell.serving.metrics as m

# create prometheus metric (y'know, in case you want it)

class Predictor(BasePredictor):
    """ Simple metrics example. """
    def __init__(self):
        # prometheus metrics objects can only be defined once
        self.as_prom_gauge = m.prometheus.Gauge("prometheus_minutes_since_hour", "minutes past the hour")
        self.with_labels = m.prometheus.Gauge("prometheus_minutes_since_hour_with_labels", "minutes past the hour", ["spell_metric_name", "spell_tag"])

    def predict(self, payload):
        now = datetime.now()
        minutes_since_hour = (now - now.replace(minute=0, second=0, microsecond=0)).total_seconds() / 60.

        # send a metric
        m.send_metric("minutes_since_hour", minutes_since_hour)

        # A metric can also have arbitrary unicode, and its b16-encoding is used for the prometheus metric name
        m.send_metric("Minutes past the hour 🕑", minutes_since_hour)

        # send a metric with a tag
        m.send_metric("minutes_since_hour", minutes_since_hour, tag=str(now.hour))

        # send a metric using the prometheus client library
        self.as_prom_gauge.set(minutes_since_hour)

        # use the prometheus client library with spell's labels
        self.with_labels.labels(spell_metric_name="as prometheus metric", spell_tag=str(now.hour))

        return json.dumps({"response" : minutes_since_hour})

