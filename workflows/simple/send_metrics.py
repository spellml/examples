import spell.metrics
import time

for x in range(30, 0, -1):
    print(x)
    spell.metrics.send_metric("loss", x)
    time.sleep(1)
