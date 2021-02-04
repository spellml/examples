import os
import uuid
from asyncio import wrap_future

import numpy as np
import lightgbm as lgb

from spell.serving import BasePredictor
from arize.api import Client
from starlette.background import BackgroundTasks

async def log_to_arize(results, client, model_id, model_version):
    futures = []
    for result in results:
        prediction_id = str(uuid.uuid4())
        future = client.log_prediction(
            model_id=model_id,
            model_version=model_version,
            prediction_id=prediction_id,
            prediction_label=bool(result)  # {1,0} => {true,false}
        )
        future = wrap_future(future)  # SO#34376938
        futures.append(future)
    for future in futures:
        await future
        status_code = future.result().status_code
        if status_code != 200:
            raise IOError(f"Could not reach Arize! Got error code {status_code}.")
    return True

class PythonPredictor(BasePredictor):
    def __init__(self):
        self.model = lgb.Booster(model_file="/model/churn_model/lgb_classifier.txt")
        self.arize_client = Client(
            organization_key=os.environ['ARIZE_ORG_KEY'],
            api_key=os.environ['ARIZE_API_KEY']
        )
        self.model_id = 'churn-model'
        self.model_version = '0.0.1'

    def predict(self, request, tasks: BackgroundTasks):
        payload = request['payload']
        # use np.round to squeeze to binary {0,1}
        results = list(np.round(self.model.predict(payload)))

        tasks.add_task(
            log_to_arize, self.arize_client, self.model_id, self.model_version
        )

        response = {'result': results}
        return response
