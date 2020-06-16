import math
import numpy as np
from PIL import Image
from tensorflow import keras


# from keras import layers, optimizers, metrics, losses
#
# layers.UpSampling2D()
# layers.MaxPool2D()
# layers.Conv2D()
# layers.Dense()
# layers.BatchNormalization()
#
# optimizers.SGD()


def generator_model():
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(100,)))
    model.add(keras.layers.Dense(1024, activation='tanh'))
    model.add(keras.layers.Dense(128 * 7 * 7))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('tanh'))
    model.add(keras.layers.Reshape((7, 7, 128)))
    model.add(keras.layers.UpSampling2D(size=(2, 2)))
    model.add(keras.layers.Conv2D(64, (5, 5), padding='same', activation='tanh'))
    model.add(keras.layers.UpSampling2D(size=(2, 2)))
    model.add(keras.layers.Conv2D(1, (5, 5), padding='same', activation='tanh'))
    return model


def discriminator_model():
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(28, 28, 1)))
    model.add(keras.layers.Conv2D(64, (5, 5), padding='same', activation='tanh'))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Conv2D(128, (5, 5), activation='tanh'))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1024, activation='tanh'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    return model


def generator_containing_discriminator(g, d):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(100,)))
    model.add(g)
    d.trainable = False
    model.add(d)
    return model


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height * shape[0], width * shape[1]), dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]: (j + 1) * shape[1]] = img[:, :, 0]
    return image


def train(batch_size=32):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_train = x_train[:, :, :, None]
    x_test = x_test[:, :, :, None]

    d = discriminator_model()
    g = generator_model()
    d_on_g = generator_containing_discriminator(g, d)

    keras.utils.plot_model(g, to_file='g.png', show_shapes=True)
    keras.utils.plot_model(d, to_file='d.png', show_shapes=True)
    keras.utils.plot_model(d_on_g, to_file='d_on_g.png', show_shapes=True)

    print(g.summary())
    print(d.summary())
    print(d_on_g.summary())

    g_opt = keras.optimizers.SGD()
    d_opt = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)
    d_on_g_opt = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)

    g.compile(loss='binary_crossentropy', optimizer=g_opt)
    d_on_g.compile(loss='binary_crossentropy', optimizer=d_on_g_opt)

    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_opt)

    for epoch in range(500):
        print('epoch is: ', epoch)
        print('number of batch', int(x_train.shape[0] / batch_size))
        for index in range(int(x_train.shape[0] / batch_size)):
            noise = np.random.uniform(-1, 1, size=(batch_size, 100))
            image_batch = x_train[index * batch_size: (index + 1) * batch_size]
            generated_image = g.predict(noise, verbose=0)

            if index % 100 == 0:
                image = combine_images(generated_image)
                image = image * 127.5 + 127.5
                Image.fromarray(image.astype(np.uint8)).save(
                    f'./gan/{epoch}-{index}.png'
                )

            x = np.concatenate((image_batch, generated_image))
            y = [1] * batch_size + [0] * batch_size

            d_loss = d.train_on_batch(x, y)
            print(f'batch {index}, d_loss {d_loss}')

            noise = np.random.uniform(-1, 1, (batch_size, 100))
            d.trainable = False

            d_on_g_loss = d_on_g.train_on_batch(noise, [1] * batch_size)
            print(f'batch {index}, d_on_g_loss {d_on_g_loss}')

            d.trainable = True

            if index % 100 == 9:
                g.save_weights('models/generator', True)
                d.save_weights('models/discriminator', True)


def generate(batch_size, nice=False):
    g = generator_model()
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    g.load_weights('models/generator')

    if nice:
        d = discriminator_model()
        d.compile(loss='binary_crossentropy', optimizer="SGD")
        d.load_weights('models/discriminator')

        noise = np.random.uniform(-1, 1, (batch_size * 10, 100))
        generated_images = g.predict(noise, verbose=1)
        d_pre = d.predict(generated_images, verbose=1)
        index = np.arange(0, batch_size * 10)
        index.resize((batch_size * 10, 1))
        pre_with_index = list(np.append(d_pre, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((batch_size,) + generated_images.shape[1:3], dtype=np.float32)
        nice_images = nice_images[:, :, :, None]
        for i in range(batch_size):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
        image = combine_images(nice_images)
    else:
        noise = np.random.uniform(-1, 1, (batch_size, 100))
        generated_images = g.predict(noise, verbose=0)
        image = combine_images(generated_images)
    image = image * 127.5 + 127.5
    Image.fromarray(image.astype(np.uint8)).save("./generated_image.png")


if __name__ == '__main__':
    train(100)
    generate(9, True)
