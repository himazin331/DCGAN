import argparse as arg
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras.layers as kl

import numpy as np

import matplotlib.pyplot as plt

# Generator
class Generator(tf.keras.Model):
    def __init__(self):
        super().__init__()

        input_wh_size = 32
        parm = int(input_wh_size / 4)

        # 入力データ
        noise_shape = (100,)

        self.dens1 = kl.Dense(parm*parm*128, activation="tanh")
        self.bn1 = kl.BatchNormalization()

        self.re = kl.Reshape((parm,parm,128), input_shape=(128*parm*parm,))
        self.ups1 = kl.UpSampling2D((2,2))

        self.conv1 = kl.Conv2D(64, 5, padding="same", activation="tanh")
        self.ups2 = kl.UpSampling2D((2,2))

        self.conv2 = kl.Conv2D(32, 5, padding="same", activation="tanh")
        self.conv3 = kl.Conv2D(16, 5, padding="same", activation="tanh")
        self.conv4 = kl.Conv2D(1, 5, padding="same", activation="tanh")

    def call(self, x):

        d1 = self.bn1(self.dens1(x))
        d2 = self.ups1(self.re(d1))
        d3 = self.ups2(self.conv1(d2))
        d4 = self.conv2(d3)
        d5 = self.conv3(d4)
        d6 = self.conv4(d5)
        
        return d6

# Discriminator
class Discriminator(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.conv1 = kl.Conv2D(32, 5, padding="same", input_shape=(28, 28, 1))
        self.act1 = kl.Activation(kl.LeakyReLU(0.2))
        self.mp1 = kl.MaxPooling2D((2, 2))

        self.conv2 = kl.Conv2D(64, 5, padding="same")
        self.act2 = kl.Activation(kl.LeakyReLU(0.2))
        self.mp2 = kl.MaxPooling2D((2, 2))

        self.conv3 = kl.Conv2D(128, 5, padding="same")
        self.act3 = kl.Activation(kl.LeakyReLU(0.2))
        self.mp3 = kl.MaxPooling2D((2, 2))

        self.flt = kl.Flatten()

        self.dens1 = kl.Dense(1024)
        self.act4 = kl.Activation(kl.LeakyReLU(0.2))
        self.dens2 = kl.Dense(1, activation="sigmoid")

    def call(self, x):    

        d1 = self.mp1(self.act1(self.conv1(x)))
        d2 = self.mp2(self.act2(self.conv2(d1)))
        d3 = self.mp3(self.act3(self.conv3(d2)))
        d4 = self.dens1(self.act4(self.flt(d3)))
        d5 = self.dens2(d4)

        return d5

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

# 学習
class trainer():
    def __init__(self):
        
        # Discriminator
        self.discriminator = Discriminator()
        self.discriminator.compile(optimizer=tf.keras.optimizers.Adam(),
                                loss=tf.keras.losses.BinaryCrossentropy(),
                                metrics=['accuracy'])

        # Generator
        self.generator = Generator()

        # GAN
        self.model = GAN(self.generator, self.discriminator)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                            loss=tf.keras.losses.BinaryCrossentropy(),
                            metrics=['accuracy'])
        
    def train(self, tr_images, out_path, batch_size, iteration):
            
        h_batch = int(batch_size / 2)

        # 学習
        for ite in range(iteration):

            # 生成画像取得
            noise = np.random.normal(0, 1, (h_batch, 100)) # 入力データ作成
            gen_imgs = self.generator.predict(noise)

            # 初期生成画像保存
            if ite+1 == 1:
                self.save_imgs(ite-1)

            # 学習データピックアップ
            idx = np.random.randint(0, tr_images.shape[0], h_batch)
            imgs = tr_images[idx]
            
            # Discriminator 学習
            self.discriminator.trainable = True # Discriminator学習有効
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((h_batch, 1)))    # 本物データに対する学習
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((h_batch, 1))) # 偽物データに対する学習
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake) # 平均損失値

            noise = np.random.normal(0, 1, (batch_size, 100)) # 入力データ作成
            valid_y = np.array([1] * batch_size) # Generatorラベル

            # Generator 学習
            self.model.discriminator.trainable = False # Discriminator学習無効
            g_loss = self.model.train_on_batch(noise, valid_y)

            print ("iteration {0} [D loss: {1}] [G loss: {2}]".format(ite, d_loss, g_loss))

            # 50イテレーション毎に生成画像保存
            if (ite+1) % 50 == 0:
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
        fig.savefig("mnist_%d.png" % iteration)
        plt.close()                   

def create_dataset(data_dir):

    train_img = []
    for c in os.listdir(data_dir):
        d = os.path.join(data_dir, c)

        _, ext = os.path.splitext(c)
        if ext.lower() != '.jpg':
            continue

        img = tf.io.read_file(d)    
        img = tf.image.decode_image(img, channels=1)

        train_img.append(img)

    train_img = tf.convert_to_tensor(train_img, np.float32)
    train_img = train_img.numpy()
    train_img = (train_img.astype(np.float32) - 127.5) / 127.5
    
    return train_img

def main():
    
    # プログラム情報
    print("DCGAN ver.4")
    print("Last update date:    2020/05/08\n")
    
    # コマンドラインオプション作成
    parser = arg.ArgumentParser(description='DCGAN')
    parser.add_argument('--data_dir', '-d', type=str, default=None,
                        help='画像フォルダパスの指定(未指定ならエラー)')
    parser.add_argument('--out', '-o', type=str,
                        default=os.path.dirname(os.path.abspath(__file__)),
                        help='生成画像の保存先指定(デフォルト値=./result')
    parser.add_argument('--batch_size', '-b', type=int, default=64,
                        help='ミニバッチサイズの指定(デフォルト値=64)')
    parser.add_argument('--iter', '-i', type=int, default=3000,
                        help='学習回数の指定(デフォルト値=10)')
    args = parser.parse_args()

    # 画像フォルダパス未指定->例外
    if args.data_dir == None:
        print("\nException: Folder not specified.\n")
        sys.exit()
    # 存在しない画像フォルダ指定時->例外
    if os.path.exists(args.data_dir) != True:
        print("\nException: Folder \"{}\" is not found.\n".format(args.data_dir))
        sys.exit()
        
    # 設定情報出力
    print("=== Setting information ===")
    print("# Images folder: {}".format(os.path.abspath(args.data_dir)))
    print("# Output folder: {}".format(args.out))
    print("# Minibatch-size: {}".format(args.batch_size))
    print("# Iteration: {}".format(args.iter))
    print("===========================")
    
    # 出力フォルダの作成(フォルダが存在する場合は作成しない)
    os.makedirs(args.out, exist_ok=True)

    train_images = create_dataset(args.data_dir)

    # 学習対象: MNIST手書き数字
    """
    mnist = tf.keras.datasets.mnist
    (train_images, _), (_, _) = mnist.load_data()

    train_images = (train_images.astype(np.float32) - 127.5) / 127.5
    train_images = np.expand_dims(train_images, axis=3)
    """
    
    Trainer = trainer()
    Trainer.train(train_images, out_path=out_path, batch_size=batch_size, iteration=iteration)

if __name__ == '__main__':
    main()
