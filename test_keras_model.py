from keras.models import model_from_json, Sequential
import numpy as np


def pred(img):
    img7_arr = np.asarray(img).reshape(1,28,28,1)

    print(img7_arr.shape)


    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    predict = loaded_model.predict(img7_arr)
    return output(predict)

def output(x):
    x = np.squeeze(x)
    x = list(x)
    print(x.index(max(x)))
    return x.index(max(x))
