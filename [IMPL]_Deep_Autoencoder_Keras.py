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
