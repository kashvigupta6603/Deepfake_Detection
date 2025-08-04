# d_image.py
import face_recognition
import numpy as np
import cv2
from keras.models import load_model
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense


class MesoNet:
    def __init__(self, model_path='meso4.h5'):
        self.model = self._build_model()
        self.model.load_weights(model_path)

    def _build_model(self):
        x = Input(shape=(256, 256, 3))
        y = Conv2D(8, (3, 3), padding='same')(x)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = MaxPooling2D(pool_size=(2, 2), padding='same')(y)

        y = Conv2D(8, (5, 5), padding='same')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = MaxPooling2D(pool_size=(2, 2), padding='same')(y)

        y = Conv2D(16, (5, 5), padding='same')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = MaxPooling2D(pool_size=(2, 2), padding='same')(y)

        y = Conv2D(16, (5, 5), padding='same')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = MaxPooling2D(pool_size=(4, 4), padding='same')(y)

        y = Flatten()(y)
        y = Dense(units=16)(y)
        y = Activation('relu')(y)
        y = Dense(units=1, activation='sigmoid')(y)

        return Model(inputs=x, outputs=y)

    def predict(self, face_img):
        img = cv2.resize(face_img, (256, 256))
        batch = np.expand_dims(img.astype('float32') / 255., axis=0)
        prob = self.model.predict(batch)[0][0]
        return prob


def predict_image(image_path, reference_image_path='gujju.jpg', model_path='meso4.h5'):
    """Run deepfake detection on one image"""
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    # Load reference
    ref = face_recognition.load_image_file(reference_image_path)
    ref_encodings = face_recognition.face_encodings(ref)
    if not ref_encodings:
        return "Reference image has no face", 0.0
    ref_enc = ref_encodings[0]

    if len(face_locations) == 0:
        return "No face detected", 0.0

    meso = MesoNet(model_path)
    for (top, right, bottom, left), enc in zip(face_locations, face_encodings):
        distance = face_recognition.face_distance([ref_enc], enc)[0]
        match = face_recognition.compare_faces([ref_enc], enc, tolerance=0.6)[0]

        if match:
            face_img = image[top:bottom, left:right]
            prob = meso.predict(face_img)
            label = "REAL" if prob > 0.5 else "FAKE"
            return label, prob * 100

    return "No match with reference face", 0.0
