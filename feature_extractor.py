
import numpy as np
import tensorflow as tf
import keras
                         

class Feature_Extractor():
    def load():
        model = keras.models.load_model('./model_features/CUSTOM_NET-500ep_b64_MaxP_DBCrop.h5')
        return model

    def __init__(self):
        model = self.load()
        flat_layer = model.get_layer("flatten")
        self.feature_extractor = keras.Model(inputs=model.input, outputs=flat_layer.output)
    
    def __call__(self, image):
        tensor = tf.io.decode_image(image, channels=3, dtype=tf.dtypes.float32)
        tensor = tf.image.resize(tensor, [176, 123])
        input_tensor = tf.expand_dims(tensor, axis=0)
        feature_vector = self.feature_extractor(input_tensor)
        feature_vector = feature_vector.numpy()[0]
        return feature_vector
    

    



