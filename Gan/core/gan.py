import pandas as pd 
import tensorflow as tf

from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler


###   LABEL   ###
# 0 = fake
# 1 = real


class GAN():
    def __init__(self, noise_dim, classes, features, epochs=300, batch_size=32, save=None, load=None):
        self.features = features
        self.classes = classes
        self.noise_dim = noise_dim
        self.features_dim = len(features)
        self.epochs = epochs
        self.batch_size = batch_size

        self.loss_fcn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
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
        

    def fit(self, dataframe):
        dataframe = dataframe[self.features]
        dataframe['LABEL'] = 1
        for epoch in range(self.epochs):
            print(f'\nEPOCH {epoch}')
            
            disc_losses = []
            gen_losses = []
            batches = self.CreateBatches(dataframe)
            for batch in batches:
                disc_losses.append(self.TrainDiscriminator(batch.copy()))
                breakpoint()
                gen_losses.append(self.TrainGenerator(batch))

            avg_disc_loss = sum(disc_losses) / len(disc_losses)
            avg_gen_loss = sum(gen_losses) / len(gen_losses)
            print(f'discriminator loss = {avg_disc_loss:.4f}')
            print(f'generator loss = {avg_gen_loss:.4f}')
        
        return self


#def save(self):


#    def Generate(nb_generation): # when training is done, gen synthetic data



    def TrainDiscriminator(self, batch):
        noise = tf.random.normal([self.batch_size, self.noise_dim])
        generated_data = self.generator(noise)
        generated_df = pd.DataFrame(generated_data.numpy(), columns=self.features)
        generated_df['LABEL'] = 0
        batch = pd.concat([batch, generated_df], ignore_index=True)

        X = batch[self.features].values
        y = batch['LABEL'].values

        X = tf.convert_to_tensor(X, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)

        breakpoint()

        gradients = tape.gradient(loss, self.discriminator.trainable_variables)
        breakpoint()
        self.disc_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))
        return loss


    def TrainGenerator(self, batch):
        noise = tf.random.normal([batch_size, noise_dim])
        generated_data = self.generator(noise)
        generated_df = pd.DataFrame(generated_data.numpy(), columns=self.features)
        generated_df['LABEL'] = 1

        batch = pd.concat([batch, generated_df], ignore_index=True)

        X = batch[self.features].values
        y = batch['LABEL'].values

        X = tf.convert_to_tensor(X, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)

        with tf.GradientTape() as tape:
            predictions = self.discriminator(X, training=True)
            predicted_class = tf.argmax(predictions, axis=1)
            loss = self.loss_fcn(y, tf.cast(predicted_class, tf.float32))

        gradients = tape.gradient(loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))
        return loss


    def CreateBatches(self, dataframe):
        return [dataframe.iloc[i:i + self.batch_size] for i in range(0, len(dataframe), self.batch_size)]



if __name__ == '__main__':
    dataframe = pd.read_csv('data/diabetes.csv', index_col=False)
    dataframe = dataframe[0:1000]

    features = list(dataframe.columns)
    gan = GAN(20, [0, 1], features)

    scaler = StandardScaler()
    dataframe_scaled = pd.DataFrame(scaler.fit_transform(dataframe[features]), columns=features)
    dataframe[features] = dataframe_scaled
    gan.fit(dataframe)













