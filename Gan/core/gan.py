import pandas as pd 
import tensorflow as tf

from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler


###   LABEL   ###
# 0 = fake
# 1 = real


class GAN():
    def __init__(self, noise_dim, features, epochs=300, batch_size=32, save=None, load=None):
        self.noise_dim = noise_dim
        self.features_dim = len(features)
        self.epochs = epochs
        self.batch_size = batch_size

        self.loss_fcn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
        self.gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

        if load is None:
            self.discriminator = tf.keras.Sequential([
                tf.keras.layers.Dense(128, input_dim=self.features_dim, activation='relu'),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dense(2, activation='sigmoid'),
            ])

            self.generator = tf.keras.Sequential([
                tf.keras.layers.Dense(128, input_dim=noise_dim, activation='relu'),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dense(self.features_dim, activation='tanh'),
            ])

        else:
            self.discriminator = load_model(f'data/GAN/discriminator_{crypto}') 
            self.generator = load_model(f'data/GAN/generator_{crypto}')
        

    def fit(self, df):
        dataset = tf.convert_to_tensor(df.values, dtype=tf.float32)
        dataset = tf.data.Dataset.from_tensor_slices(dataset)
        max_iter = 10

        for epoch in range(self.epochs):
            print(f'\nEPOCH {epoch}')
            dataset_shuffled = dataset.shuffle(buffer_size=len(dataset))
            batches = dataset_shuffled.batch(self.batch_size, drop_remainder=True)
            loss_disc = 0.8
            loss_gen = 1.2
            for batch in batches:
                iteration = 0
                while loss_disc > 0.7 and iteration < max_iter:
                    loss_disc = self.TrainDiscriminator(batch)
                
                iteration = 0
                while loss_gen > 1.1 and iteration < max_iter:
                    loss_gen = self.TrainGenerator()

            print(f'discriminator loss = {loss_disc:.4f}')
            print(f'generator loss = {loss_gen:.4f}')
        
        return self


    def TrainDiscriminator(self, batch):
        noise = tf.random.normal([len(batch), self.noise_dim])
        generated_data = self.generator(noise)

        real_labels = tf.ones_like(batch[:, 0])
        fake_labels = tf.zeros_like(generated_data[:, 0])

        X = tf.concat([batch, generated_data], axis=0)
        y = tf.concat([real_labels, fake_labels], axis=0)

        idx = tf.range(start=0, limit=tf.shape(X)[0], dtype=tf.int32)
        shuffled_idx = tf.random.shuffle(idx)

        X = tf.gather(X, shuffled_idx)
        y = tf.gather(y, shuffled_idx)

        with tf.GradientTape() as tape:
            predictions = self.discriminator(X, training=True)
            
#            print('\nDiscriminator preds:')
#            preds = tf.argmax(predictions, axis=1)
#            for true, pred in zip(y, preds):
#                print(f'   {true}  ===>  {pred}')
            
            loss = self.loss_fcn(y, predictions[:, 1])

        gradients = tape.gradient(loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))
        return loss


    def TrainGenerator(self):
        with tf.GradientTape() as tape:
            noise = tf.random.normal([self.batch_size, self.noise_dim])
            gen_data = self.generator(noise, training=True)
            predictions = self.discriminator(gen_data, training=True)
            loss = self.loss_fcn(tf.ones_like(predictions[:, 1]), predictions[:, 1])

        gradients = tape.gradient(loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))
        return loss



if __name__ == '__main__':
    dataframe = pd.read_csv('data/diabetes.csv', index_col=False)
    columns = list(dataframe.columns)

    scaler = StandardScaler()
    dataframe = pd.DataFrame(scaler.fit_transform(dataframe), columns=columns)

    gan = GAN(20, columns)
    gan.fit(dataframe)













