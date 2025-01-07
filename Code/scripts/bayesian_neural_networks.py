import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds

# Chargement du dataset
(ds_train, ds_test), ds_info = tfds.load(
    'fashion_mnist',
    split=['train', 'test'],
    as_supervised=True,
    with_info=True
)

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

ds_train = ds_train.map(preprocess).batch(128).shuffle(1000)
ds_test = ds_test.map(preprocess).batch(128)

tfd = tfp.distributions

def prior_fn(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = tf.keras.Sequential([
        tfp.layers.DistributionLambda(lambda t: tfd.MultivariateNormalDiag(
            loc=tf.zeros(n), scale_diag=tf.ones(n)))
    ])
    return prior_model

def posterior_fn(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = tf.keras.Sequential([
        tfp.layers.VariableLayer(
            params=2 * n,
            dtype=dtype,
            initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.1)
        ),
        tfp.layers.DistributionLambda(lambda t: tfd.MultivariateNormalDiag(
            loc=t[..., :n],
            scale_diag=tf.nn.softplus(t[..., n:])
        ))
    ])
    return posterior_model

# Modèle fonctionnel
inputs = tf.keras.Input(shape=(28, 28))  # Définir l'entrée
x = tf.keras.layers.Flatten()(inputs)    # Transformation en tenseur plat

# Première couche bayésienne
x = tfp.layers.DenseVariational(
    units=128,
    make_prior_fn=prior_fn,
    make_posterior_fn=posterior_fn,
    kl_weight=1/ds_info.splits['train'].num_examples,
    activation='relu'
)(x)

# Deuxième couche bayésienne
outputs = tfp.layers.DenseVariational(
    units=10,
    make_prior_fn=prior_fn,
    make_posterior_fn=posterior_fn,
    kl_weight=1/ds_info.splits['train'].num_examples,
    activation='softmax'
)(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Entraînement et Évaluation
model.fit(ds_train, epochs=10, validation_data=ds_test)
loss, accuracy = model.evaluate(ds_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
