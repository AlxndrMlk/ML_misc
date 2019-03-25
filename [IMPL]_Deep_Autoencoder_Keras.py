# This code is partially authored by the Authors of Coursera's Deep Learning Specialization by HSE & Yandex:
# https://www.coursera.org/learn/intro-to-deep-learning/

from keras import layers as L

def build_deep_autoencoder(img_shape, code_size):

    H,W,C = img_shape
    
    # encoder
    encoder = keras.models.Sequential()
    encoder.add(L.InputLayer(img_shape))

    encoder.add(L.Conv2D(32, kernel_size=(3,3), activation='elu', padding='same'))
    encoder.add(L.MaxPooling2D(2, padding='same'))
    
    encoder.add(L.Conv2D(64, kernel_size=(3,3), activation='elu', padding='same'))
    encoder.add(L.MaxPooling2D(2, padding='same'))
    
    encoder.add(L.Conv2D(128, kernel_size=(3,3), activation='elu', padding='same'))
    encoder.add(L.MaxPooling2D(2, padding='same'))
    
    encoder.add(L.Conv2D(256, kernel_size=(3,3), activation='elu', padding='same'))
    encoder.add(L.MaxPooling2D(2, padding='same'))
    
    encoder.add(L.Flatten())
    
    encoder.add(L.Dense(H*W))
    encoder.add(L.Dense(code_size))
    
    
    # decoder
    decoder = keras.models.Sequential()
    decoder.add(L.InputLayer((code_size,)))
    
    decoder.add(L.Dense(H*W))
    
    shape1 = int((H*W/256)**0.5)
    decoder.add(L.Reshape((shape1, shape1, 256)))
    
    decoder.add(L.Conv2DTranspose(128, kernel_size=(3,3), strides=2, activation='elu', padding='same'))    
    decoder.add(L.Conv2DTranspose(64, kernel_size=(3,3), strides=2, activation='elu', padding='same'))
    decoder.add(L.Conv2DTranspose(32, kernel_size=(3,3), strides=2, activation='elu', padding='same'))
    decoder.add(L.Conv2DTranspose(3, kernel_size=(3,3), strides=2, activation=None, padding='same'))
    
    return encoder, decoder

# Build and train autoencoder
s = reset_tf_session()

encoder, decoder = build_deep_autoencoder(IMG_SHAPE, code_size=32)

inp = L.Input(IMG_SHAPE)
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = keras.models.Model(inputs=inp, outputs=reconstruction)
autoencoder.compile(optimizer="adamax", loss='mse')

autoencoder.fit(x=X_train, y=X_train, epochs=25,
                validation_data=[X_test, X_test],
                callbacks=[keras_utils.ModelSaveCallback(model_filename),
                           keras_utils.TqdmProgressCallback()],
                verbose=0,
                initial_epoch=last_finished_epoch or 0)


