import os

from spell.serving import BasePredictor
import tecton
from tecton import conf

TECTON_API_KEY = os.environ['TECTON_API_KEY']
TECTON_API_SERVICE = os.environ['TECTON_API_SERVICE']
TECTON_FEATURE_SERVICE = os.environ['TECTON_FEATURE_SERVICE']


class PythonPredictor(BasePredictor):
    def __init__(self):
        self.model = beats_average  # simple heuristical model

        # configure tecton
        conf.set("TECTON_API_KEY", TECTON_API_KEY)
        conf.set('API_SERVICE', TECTON_API_SERVICE)
        conf.set('FEATURE_SERVICE', TECTON_FEATURE_SERVICE)

        self.feature_service = tecton.get_feature_service("ctr_prediction_service")

    def predict(self, request):
        # The request body only contains the unique key identifying this user and ad in this
        # datastream. We use Tecton's online feature service to retrieve the features relevant
        # to the model using the identifying keys.
        feature_vector_key = {"ad_id": request['ad_id'], "user_uuid": request['user_uuid']}
        feature_vector = self.feature_service.get_feature_vector(feature_vector_key)

        # compute a response and return it
        response = self.model(feature_vector)
        return response

def beats_average(fv):
    # fv is a clickstream feature vector with the following shape:
    fv = fv.to_dict()
    # {'ad_ground_truth_ctr_performance_7_days.ad_total_clicks_7days': int,
    #  ...,
    #  'ad_ground_truth_ctr_performance_7_days.ad_total_impressions_7days': int}
    clicks = fv.get('ad_ground_truth_ctr_performance_7_days.ad_total_clicks_7days')
    impressions = fv.get('ad_ground_truth_ctr_performance_7_days.ad_total_impressions_7days')
    if impressions == 0:
        return {'target': 0}
    else:
        # "9% clickthrough rate is the average across the industry." - someone on the Internet
        return {'target': int((clicks / impressions) >= 0.09)}
