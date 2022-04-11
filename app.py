from flask import Flask
from flask import Response
from flask import request
from io import BytesIO
import xgboost
import joblib
import json
import numpy as np
import boto3
import os

app = Flask(__name__)


region_name = os.environ.get("region_name")
aws_access_key_id = os.environ.get("aws_access_key_id")
aws_secret_access_key = os.environ.get("aws_secret_access_key")
bucket = os.environ.get("bucket")
folder = os.environ.get("folder")
filename = os.environ.get("filename")

s3 = boto3.client(
            "s3",
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )

with BytesIO() as f:
    
    objeto = s3.download_fileobj(Bucket=bucket, Key=f'{folder}/{filename}', Fileobj=f)
    f.seek(0)
    model1 = joblib.load(f)


@app.route("/model", methods=["POST", "GET"])
def server():
  try:
      if request.method == "POST":
        if request.data:
            # print(json.loads(request.data))
            data = json.loads(request.data)
            payload = data["payload"]
            print(payload)
            payload_tratado = np.array(payload).reshape((1,-1))
            result = model1.predict(payload_tratado)
            print(result)
            
            return Response(f"Resultado: {result}", status=201, mimetype="application/json")
  except OSError as err:
      print(err)
      return Response(f"{err}", status=404, mimetype="application/json")

if __name__ == "__main__":
    app.debug = True
    app.run()