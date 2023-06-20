import os
import requests
import grpc
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


def get_request_header():

    xsuaa_base_url = "your_url"
    client_id = "your_client_id"
    client_secret = "your_client_secret"

    response = requests.post(url=xsuaa_base_url + '/oauth/token',
                             data={'grant_type': 'client_credentials',
                                   'client_id': client_id,
                                   'client_secret': client_secret})
    access_token = response.json()["access_token"]
    return {'Authorization': 'Bearer {}'.format(access_token), 'Accept': 'application/json'}


def upload_model():

    MODEL_REPO_URL = "https://mlfproduction-model-api.cfapps.eu10.hana.ondemand.com/api/v2/models/fashionClassifier/versions"
    file = open('fashionClassifier.zip', 'rb')
    headers = get_request_header()
    files = {'file': file}
    response = requests.post(MODEL_REPO_URL, files=files, headers=headers)
    print(response.json())


def deploy_model():
    DEPLOYMENT_API_URL = "https://mlfproduction-deployment-api.cfapps.eu10.hana.ondemand.com/api/v2/modelServers/"
    headers = get_request_header()
    data = {
        "specs": {
            "models": [
                {
                    "modelName": "fashionClassifier",
                    "modelVersion": 1
                }
            ],
            "modelRuntimeId": "tf-1.11"
        }
    }
    response = requests.post(DEPLOYMENT_API_URL, json=data, headers=headers)
    print(response.json())


def model_status():
    DEPLOYMENT_API_URL = "https://mlfproduction-deployment-api.cfapps.eu10.hana.ondemand.com/api/v2/modelServers"
    headers = get_request_header()
    payload = {'modelName': 'fashionClassifier'}
    response = requests.get(
        DEPLOYMENT_API_URL,  params=payload, headers=headers)
    print(response.json())   
    return response.json()


def apply_inference():
    metadata = []
    headers = get_request_header()
    model_details = model_status()
    metadata.append(('authorization',  headers['Authorization']))

    MODEL_NAME = model_details['modelServers'][0]['specs']['models'][0]['modelName']
    MODEL_SERVER_HOST = model_details['modelServers'][0]['endpoints'][0]['host']
    MODEL_SERVER_PORT = int(model_details['modelServers'][0]['endpoints'][0]['port'])
    ROOT_CERT = model_details['modelServers'][0]['endpoints'][0]['caCrt']

    credentials = grpc.ssl_channel_credentials(
        root_certificates=ROOT_CERT.encode())
    channel = grpc.secure_channel('{}:{}'.format(
        MODEL_SERVER_HOST, MODEL_SERVER_PORT), credentials)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    data = open('image.jpg', 'rb').read()
    
    decode_img = tf.image.decode_jpeg(data, channels=1)
    img = tf.image.convert_image_dtype(decode_img, tf.float32)
    img = tf.image.resize(img, [28, 28])
    img = tf.reshape(img, [-1, 28, 28, 1])

    request = predict_pb2.PredictRequest()
    request.model_spec.name = MODEL_NAME
    request.model_spec.signature_name = 'serving_default'


    request.inputs['input_1'].CopyFrom(
        tf.compat.v1.make_tensor_proto(img, shape=[1, 28, 28, 1]))
    print(stub.Predict(request, 100, metadata=tuple(metadata)))

if __name__ == "__main__":
    #model_status()   
    # upload_model()
    # deploy_model()
    apply_inference()

