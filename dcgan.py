import argparse as arg
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras.layers as kl

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import create_dataset

# Generator
class Generator(tf.keras.Model):
    def __init__(self, input_shape):
        super().__init__()

        parm = int(input_shape[1] / 4)
            
        noise_shape = (100,)

        self.dens1 = kl.Dense(parm*parm*128, use_bias=False, input_shape=noise_shape)
        self.bn1 = kl.BatchNormalization()
        self.act1 = kl.Activation(kl.ReLU())

        self.re = kl.Reshape((parm,parm,128), input_shape=(128*parm*parm,))

        self.deconv1 = kl.Conv2DTranspose(128, 5, padding='same', use_bias=False,)
        self.bn2 = kl.BatchNormalization()
        self.act2 = kl.Activation(kl.ReLU())

        self.deconv2 = kl.Conv2DTranspose(64, 5, 2, padding='same', use_bias=False,)
        self.bn3 = kl.BatchNormalization()
        self.act3 = kl.Activation(kl.ReLU())
        
        self.deconv3 = kl.Conv2DTranspose(1, 5, 2, padding="same", use_bias=False, activation="tanh")

    def call(self, x):

        d1 = self.act1(self.bn1(self.dens1(x)))
        d2 = self.re(d1)
        d3 = self.act2(self.bn2(self.deconv1(d2)))
        d4 = self.act3(self.bn3(self.deconv2(d3)))
        d5 = self.deconv3(d4)

        return d5

# Discriminator
class Discriminator(tf.keras.Model):
    def __init__(self, input_shape):
        super().__init__()

        input_shape = input_shape[1:4]
        
        self.conv1 = kl.Conv2D(64, 5, 2, padding="same", input_shape=input_shape)
        self.act1 = kl.Activation(kl.ReLU())
        self.drop1 = kl.Dropout(0.3)

        self.conv2 = kl.Conv2D(128, 5, 2, padding="same")
        self.act2 = kl.Activation(kl.ReLU())
        self.drop2 = kl.Dropout(0.3)

        self.flt = kl.Flatten()
        self.dens2 = kl.Dense(1, activation="sigmoid")

    def call(self, x):    

        d1 = self.drop1(self.act1(self.conv1(x)))
        d2 = self.drop2(self.act2(self.conv2(d1)))
        d3 = self.flt(d2)
        d4 = self.dens2(d3)

        return d4

# GAN
class GAN(tf.keras.Model):
    def __init__(self, generator, discriminator):
        super().__init__()

        self.generator = generator
        self.discriminator = discriminator

    def call(self, x):

        g = self.generator(x)
        d = self.discriminator(g)
        
        return d

class trainer():
    def __init__(self, input_shape):
        
        # Discriminator
        self.discriminator = Discriminator(input_shape)
        self.discriminator.compile(optimizer=tf.keras.optimizers.Adam(),
                                loss=tf.keras.losses.BinaryCrossentropy(),
                                metrics=['accuracy'])

        # Generator
        self.generator = Generator(input_shape)

        # GAN
        self.model = GAN(self.generator, self.discriminator)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                            loss=tf.keras.losses.BinaryCrossentropy(),
                            metrics=['accuracy'])
        
    def train(self, tr_images, out_path, batch_size, iteration):
            
        """
        self.discriminator.build(input_shape=(None, 28, 28, 1))
        self.discriminator.summary()
        self.generator.build(input_shape=(None, 28, 28, 1))
        self.generator.summary()
        """

        g_loss_plt = []
        d_loss_plt = []

        h_batch = int(batch_size / 2)
        real_y = np.ones((h_batch, 1)) # Discriminatorラベル(本物データ)
        fake_y = np.zeros((h_batch, 1)) # Discriminatorラベル(偽物データ)
        valid_y = np.array([1] * batch_size) # Generatorラベル
        
        # 学習
        for ite in range(iteration):

            # 生成画像取得
            noise = np.random.normal(0, 1, (h_batch, 100)) # 入力データ作成
            gen_imgs = self.generator.predict(noise)

            # 初期生成画像保存
            if ite == 0:
                self.save_imgs(ite)

            # 学習データピックアップ
            idx = np.random.randint(0, tr_images.shape[0], h_batch)
            imgs = tr_images[idx]

            # Discriminator 学習
            self.discriminator.trainable = True # Discriminator学習有効
            d_loss_real = self.discriminator.train_on_batch(imgs, real_y)   # 本物データに対する学習
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake_y) # 偽物データに対する学習
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake) # 平均損失値

            noise = np.random.normal(0, 1, (batch_size, 100)) # 入力データ作成

            # Generator 学習
            self.model.discriminator.trainable = False # Discriminator学習無効
            g_loss = self.model.train_on_batch(noise, valid_y)

            print ("iteration {0} [D loss: {1:.3f}] [G loss: {2:.3f}]".format(ite, d_loss[0], g_loss[0]))

            g_loss_plt.append(g_loss[0])
            d_loss_plt.append(d_loss[0])

            plt.plot(g_loss_plt)
            plt.plot(d_loss_plt)
    
            # 50イテレーション毎に生成画像保存
            if ite % 50 == 0:
                plt.pause(2)
                plt.savefig("graph.jpg")
                self.save_imgs(ite)
    
    # 生成画像保存
    def save_imgs(self, iteration):

        # 生成画像数(rows x cols)
        r, c = 5, 5

        # 生成画像取得
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)

        # 生成データ加工
        gen_imgs = 0.5 * gen_imgs + 0.5

        # 出力
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        # 保存
        fig.savefig("mnist_{}.png".format(iteration))
        plt.close()                   

def main():
    
    # プログラム情報
    print("DCGAN")
    print("Last update date:    2020/07/12\n")
    
    # コマンドラインオプション作成
    parser = arg.ArgumentParser(description='DCGAN')
    parser.add_argument('--data_dir', '-d', type=str, default=None,
                        help='画像フォルダパスの指定(未指定ならエラー)')
    parser.add_argument('--out', '-o', type=str,
                        default=os.path.dirname(os.path.abspath(__file__)),
                        help='生成画像の保存先指定(デフォルト値=./result')
    parser.add_argument('--batch_size', '-b', type=int, default=256,
                        help='ミニバッチサイズの指定(デフォルト値=64)')
    parser.add_argument('--iter', '-i', type=int, default=5000,
                        help='学習回数の指定(デフォルト値=10)')
    args = parser.parse_args()

    """
    # 画像フォルダパス未指定->例外
    if args.data_dir == None:
        print("\nException: Folder not specified.\n")
        sys.exit()
    # 存在しない画像フォルダ指定時->例外
    if os.path.exists(args.data_dir) != True:
        print("\nException: Folder \"{}\" is not found.\n".format(args.data_dir))
        sys.exit()
    """
        
    # 設定情報出力
    print("=== Setting information ===")
    #print("# Images folder: {}".format(os.path.abspath(args.data_dir)))
    print("# Output folder: {}".format(args.out))
    print("# Minibatch-size: {}".format(args.batch_size))
    print("# Iteration: {}".format(args.iter))
    print("===========================")
    
   
    # 出力フォルダの作成(フォルダが存在する場合は作成しない)
    os.makedirs(args.out, exist_ok=True)

    #train_images = create_dataset.create_dataset(args.data_dir)
  
    
    
    # 生成対象: MNIST手書き数字
    mnist = tf.keras.datasets.mnist
    (train_images, _), (_, _) = mnist.load_data()
    
    train_images = (train_images.astype(np.float32) - 127.5) / 127.5
    train_images = np.expand_dims(train_images, axis=3)
   
    
    Trainer = trainer(train_images.shape)
    Trainer.train(train_images, out_path=args.out, batch_size=args.batch_size, iteration=args.iter)

if __name__ == '__main__':
    main()
