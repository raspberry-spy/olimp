import tensorflow as tf
from tensorflow.keras.datasets import mnist
# Загрузка и нормализация датасета
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
# Функция сохранения весов в файл
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='cp/cp.ckpt',
                                                 save_weights_only=True,
                                                 verbose=1)

# Задание формы модели (стек слоев)
model = tf.keras.models.Sequential([
    # первый слой сети - преобразует формат изображений в одномерный массив 28 * 28 = 784 пикселей
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # второй слой = 128 узлов (нейронов)
    tf.keras.layers.Dense(128, activation='relu'),
    # третий слой (выходной) = 10 узлов (нейронов)
    tf.keras.layers.Dense(10)
])

model.compile(
    # 
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.fit(
    x_train,
    y_train,
    epochs=50,
    callbacks=[cp_callback]
)
