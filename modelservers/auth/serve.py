from spell.serving import BasePredictor

from starlette.requests import Request
from starlette.exceptions import HTTPException

import base64


def validate_auth(auth):
    if not auth.startswith("Basic "):
        return False
    auth = auth[len("Basic "):]
    token = base64.b64decode(auth).decode("ascii")
    token = token.rstrip("\n ")  # strip out newlines
    return token == "example-token"


class PythonPredictor(BasePredictor):
    def __init__(self):
        pass

    def predict(self, payload: Request):
        # Cf. https://www.starlette.io/requests/
        if "Authorization" not in payload.headers:
            print("Received a payload without an auth token, returning 401.")
            raise HTTPException(401)
        auth = payload.headers["Authorization"]
        if not validate_auth(auth):
            print("Received a payload with invalid an auth token, returning 401.")
            raise HTTPException(401)

        # auth check passed, proceed to predict body
        return {"status": "ok"}
