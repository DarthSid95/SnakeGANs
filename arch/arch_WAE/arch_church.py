from __future__ import print_function
import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import math

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from .resnet_ops import *

class ARCH_church():

	def __init__(self):
		print("CREATING ARCH_AAE CLASS")
		return

	def encdec_model_dcgan_church(self):

		def ama_relu(x):
			x = tf.clip_by_value(x, clip_value_min = -6., clip_value_max = 6.)
			return x

		# init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.001, seed=None)
		if self.loss in ['SW', 'FS']:
			init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.02, seed=None)
		else:
			init_fn = tf.keras.initializers.glorot_uniform()

		init_fn = tf.function(init_fn, autograph=False)
		bias_init_fn = tf.keras.initializers.Zeros()
		bias_init_fn = tf.function(bias_init_fn, autograph=False)

		inputs = tf.keras.Input(shape=(self.output_size,self.output_size,3)) #64x64x3

		enc1 = tf.keras.layers.Conv2D(128, 4, strides=2, padding='same',kernel_initializer=init_fn, use_bias=False)(inputs) #32x32x64
		enc1 = tf.keras.layers.BatchNormalization()(enc1)
		# enc1 = tf.keras.layers.Dropout(0.1)(enc1)
		enc1 = tf.keras.layers.LeakyReLU()(enc1)
		# enc1 = tf.keras.layers.Activation( activation = 'tanh')(enc1)

		enc2 = tf.keras.layers.Conv2D(256, 4, strides=2, padding='same',kernel_initializer=init_fn, use_bias=False)(enc1) #16x16x128
		enc2 = tf.keras.layers.BatchNormalization()(enc2)
		# enc2 = tf.keras.layers.Dropout(0.1)(enc2)
		enc2 = tf.keras.layers.LeakyReLU()(enc2)
		# enc2 = tf.keras.layers.Activation( activation = 'tanh')(enc2)


		enc3 = tf.keras.layers.Conv2D(512, 4, strides=2, padding='same',kernel_initializer=init_fn, use_bias=False)(enc2) #8x8x256
		enc3 = tf.keras.layers.BatchNormalization()(enc3)
		# enc3 = tf.keras.layers.Dropout(0.1)(enc3)
		enc3 = tf.keras.layers.LeakyReLU()(enc3)
		# enc3 = tf.keras.layers.Activation( activation = 'tanh')(enc3)


		enc4 = tf.keras.layers.Conv2D(1024, 4, strides=2, padding='same',kernel_initializer=init_fn, use_bias=False)(enc3) #4x4x128
		enc4 = tf.keras.layers.BatchNormalization()(enc4)
		# enc4 = tf.keras.layers.Dropout(0.5)(enc4)
		enc4 = tf.keras.layers.LeakyReLU()(enc4)
		# enc4 = tf.keras.layers.Activation( activation = 'tanh')(enc4)

		enc4 = tf.keras.layers.Conv2D(512, 4, strides=2, padding='same',kernel_initializer=init_fn, use_bias=False)(enc4) #4x4x128
		enc4 = tf.keras.layers.BatchNormalization()(enc4)
		# enc4 = tf.keras.layers.Dropout(0.5)(enc4)
		enc4 = tf.keras.layers.LeakyReLU()(enc4)
		# enc4 = tf.keras.layers.Activation( activation = 'tanh')(enc4)

		if self.output_size == 128:
			enc4 = tf.keras.layers.Conv2D(1024, 4, strides=2, padding='same',kernel_initializer=init_fn, use_bias=False)(enc4) #4x4x128
			enc4 = tf.keras.layers.BatchNormalization()(enc4)
			# enc4 = tf.keras.layers.Dropout(0.5)(enc4)
			enc4 = tf.keras.layers.LeakyReLU()(enc4)
			# enc4 = tf.keras.layers.Activation( activation = 'tanh')(enc4)


		dense = tf.keras.layers.Flatten()(enc4)

		dense = tf.keras.layers.Dense(self.latent_dims, kernel_initializer = init_fn, use_bias = True, bias_initializer = bias_init_fn)(dense)
		enc = tf.keras.layers.Dense(self.latent_dims, kernel_initializer = init_fn, use_bias = True, bias_initializer = bias_init_fn)(dense)
		
		if self.loss in ['FS']:
			enc  = tf.keras.layers.Lambda(ama_relu)(enc)


		encoded = tf.keras.Input(shape=(self.latent_dims,))

		den = tf.keras.layers.Dense(1024*int(self.output_size/16)*int(self.output_size/16), kernel_initializer = init_fn, use_bias = True, bias_initializer = bias_init_fn)(encoded)
		enc_res = tf.keras.layers.Reshape([int(self.output_size/16),int(self.output_size/16),1024])(den)
		# enc_res = tf.keras.layers.Reshape([1,1,int(self.latent_dims)])(den) #1x1xlatent

		if self.output_size < 128:
			den = tf.keras.layers.Dense(1024*int(self.output_size/16)*int(self.output_size/16), kernel_initializer = init_fn, use_bias = True, bias_initializer = bias_init_fn)(encoded)
			enc_res = tf.keras.layers.Reshape([int(self.output_size/16),int(self.output_size/16),1024])(den)
		# enc_res = tf.keras.layers.Reshape([1,1,int(self.latent_dims)])(den) #1x1xlatent
		else:
			den = tf.keras.layers.Dense(1024*int(self.output_size/32)*int(self.output_size/32), kernel_initializer = init_fn, use_bias = True, bias_initializer = bias_init_fn)(encoded)
			enc_res = tf.keras.layers.Reshape([int(self.output_size/32),int(self.output_size/32),1024])(den)

			enc_res = tf.keras.layers.Conv2DTranspose(1024, 4, strides=2,padding='same',kernel_initializer=init_fn,use_bias=False)(enc_res) #2x2x128
			enc_res = tf.keras.layers.BatchNormalization()(enc_res)
			enc_res = tf.keras.layers.LeakyReLU()(enc_res)




		denc5 = tf.keras.layers.Conv2DTranspose(512, 4, strides=2,padding='same',kernel_initializer=init_fn,use_bias=False)(enc_res) #2x2x128
		denc5 = tf.keras.layers.BatchNormalization()(denc5)
		# denc4 = tf.keras.layers.Dropout(0.5)(denc5)
		denc5 = tf.keras.layers.LeakyReLU()(denc5)
		# denc5 = tf.keras.layers.Activation( activation = 'tanh')(denc5)

		denc5 = tf.keras.layers.Conv2DTranspose(256, 4, strides=2,padding='same',kernel_initializer=init_fn,use_bias=False)(denc5) #2x2x128
		denc5 = tf.keras.layers.BatchNormalization()(denc5)
		# denc4 = tf.keras.layers.Dropout(0.5)(denc5)
		denc5 = tf.keras.layers.LeakyReLU()(denc5)
		# denc5 = tf.keras.layers.Activation( activation = 'tanh')(denc5)

		denc4 = tf.keras.layers.Conv2DTranspose(256, 4, strides=2,padding='same',kernel_initializer=init_fn,use_bias=False)(denc5) #4x4x128
		denc4 = tf.keras.layers.BatchNormalization()(denc4)
		# denc4 = tf.keras.layers.Dropout(0.5)(denc4)
		denc4 = tf.keras.layers.LeakyReLU()(denc4)
		# denc4 = tf.keras.layers.Activation( activation = 'tanh')(denc4)

		denc3 = tf.keras.layers.Conv2DTranspose(128, 4, strides=2,padding='same',kernel_initializer=init_fn,use_bias=False)(denc4) #8x8x256
		denc3 = tf.keras.layers.BatchNormalization()(denc3)
		# denc3 = tf.keras.layers.Dropout(0.5)(denc3)
		denc1 = tf.keras.layers.LeakyReLU()(denc3)
		# denc1 = tf.keras.layers.Activation( activation = 'tanh')(denc3)


		out = tf.keras.layers.Conv2DTranspose(3, 4,strides=1,padding='same', kernel_initializer=init_fn, use_bias = True, bias_initializer = bias_init_fn)(denc1) #64x64x3
		out =  tf.keras.layers.Activation( activation = 'tanh')(out)

		self.Decoder = tf.keras.Model(inputs=encoded, outputs=out)
		self.Encoder = tf.keras.Model(inputs=inputs, outputs=enc)

		return


	def discriminator_model_dcgan_church(self):
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		model = tf.keras.Sequential()
		model.add(layers.Dense(256, use_bias=False, input_shape=(self.latent_dims,), kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		model.add(layers.Activation(activation = 'tanh'))
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(512, use_bias=False, kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(512, use_bias=False, kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(1))
		# model.add(layers.Softmax())
		return model


	def encdec_model_resnet_church(self):

		def ama_relu(x):
			x = tf.clip_by_value(x, clip_value_min = -6., clip_value_max = 6.)
			return x

		D_num_channels = 64
		G_num_channels = 1024

		init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None)
		# init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)
		bias_init_fn = tf.keras.initializers.Zeros()
		bias_init_fn = tf.function(bias_init_fn, autograph=False)

		inputs = tf.keras.Input(shape = (self.output_size, self.output_size, 3,)) #64x64x3

		x = ResBlockDown(inputs, D_num_channels)  # 32x32
		D_num_channels = D_num_channels * 2

		x = ResBlockDown(x, D_num_channels)  # 16x16
		D_num_channels = D_num_channels * 2

		# x = attention(x, ch, is_training=self.is_training, scope="attention", reuse=reuse)  # 32*32*128

		x = ResBlockDown(x, D_num_channels)  #8x8
		D_num_channels = D_num_channels * 2

		x = ResBlockDown(x, D_num_channels)  # 4x4
		D_num_channels = D_num_channels * 2

		# x = ResBlockDown(x, D_num_channels) 

		dense = tf.keras.layers.Flatten()(x)

		dense = tf.keras.layers.Dense(self.latent_dims, kernel_initializer = init_fn, use_bias = True, bias_initializer = bias_init_fn)(dense)
		enc = tf.keras.layers.Dense(self.latent_dims, kernel_initializer = init_fn, use_bias = True, bias_initializer = bias_init_fn)(dense)
		enc  = tf.keras.layers.Lambda(ama_relu)(enc)

		###### Decoder Now #####

		encoded = tf.keras.Input(shape=(self.latent_dims,))

		den = tf.keras.layers.Dense(1024*int(self.output_size/32)*int(self.output_size/32), kernel_initializer = init_fn, use_bias = True, bias_initializer = bias_init_fn)(encoded)
		enc_res = tf.keras.layers.Reshape([int(self.output_size/32),int(self.output_size/32),1024])(den)

		x = ResBlockUp(enc_res, G_num_channels)  # 2*2*G_num_channels
		G_num_channels = G_num_channels // 2

		x = ResBlockUp(x, G_num_channels)  # 4*4*G_num_channels
		G_num_channels = G_num_channels // 2

		x = ResBlockUp(x, G_num_channels)  # 8*8*G_num_channels
		G_num_channels = G_num_channels // 2

		x = ResBlockUp(x, G_num_channels // 2)  # 16*16*G_num_channels
		G_num_channels = G_num_channels // 2

		x = ResBlockUp(x, G_num_channels)  # 32*32*G_num_channels

		x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
		x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
		x = tf.keras.layers.Conv2D(filters=3, kernel_size=1, strides=1, padding = 'SAME')(x)
		out =  tf.keras.layers.Activation( activation = 'tanh')(x)

		self.Decoder = tf.keras.Model(inputs=encoded, outputs=out)
		self.Encoder = tf.keras.Model(inputs=inputs, outputs=enc)

		return



	def discriminator_model_resnet_church(self):
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		model = tf.keras.Sequential()
		model.add(layers.Dense(256, use_bias=False, input_shape=(self.latent_dims,), kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		model.add(layers.Activation(activation = 'tanh'))
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(512, use_bias=False, kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(512, use_bias=False, kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(1))
		# model.add(layers.Softmax())
		return model

	def same_images_FID(self):

		def data_reader_faces(filename):
			with tf.device('/CPU'):
				print(tf.cast(filename[0],dtype=tf.string))
				image_string = tf.io.read_file(tf.cast(filename[0],dtype=tf.string))
				# Don't use tf.image.decode_image, or the output shape will be undefined
				image = tf.image.decode_jpeg(image_string, channels=3)
				image  = tf.image.crop_to_bounding_box(image, 0, 0, 256,256)
				image = tf.image.resize(image,[self.output_size,self.output_size], antialias=True)
			return image

		if self.FID_load_flag == 0:
			### First time FID call setup
			self.FID_load_flag = 1	
			random_points = tf.keras.backend.random_uniform([self.FID_num_samples], minval=0, maxval=int(self.fid_train_images.shape[0]), dtype='int32', seed=None)
			print(random_points)
			self.fid_train_images_names = self.fid_train_images[random_points]

			## self.fid_train_images has the names to be read. Make a dataset with it
			self.fid_image_dataset = tf.data.Dataset.from_tensor_slices(self.fid_train_images_names)
			self.fid_image_dataset = self.fid_image_dataset.map(data_reader_faces,num_parallel_calls=int(self.num_parallel_calls))
			self.fid_image_dataset = self.fid_image_dataset.batch(self.FID_num_samples)

		with tf.device(self.device):
			for image_batch in self.fid_image_dataset:
				for i in range(self.FID_num_samples):
					real = image_batch[i,:,:,:]
					tf.keras.preprocessing.image.save_img(self.FIDRealspath+str(i)+'.png',real,scale=True)
			for i in range(self.FID_num_samples):	
				preds = self.Decoder(self.get_noise(2), training=False)
				preds = preds.numpy()
				fake = preds[0,:,:,:]
				tf.keras.preprocessing.image.save_img(self.FIDFakespath+str(i)+'.png', fake,scale=True)
		return


	def Church_Classifier(self):
		self.FID_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, pooling='avg', weights='imagenet', input_tensor=None, input_shape=(256,256,3), classes=1000)

	def FID_church(self):

		def data_reader_faces(filename):
			with tf.device('/CPU'):
				print(tf.cast(filename[0],dtype=tf.string))
				image_string = tf.io.read_file(tf.cast(filename[0],dtype=tf.string))
				# Don't use tf.image.decode_image, or the output shape will be undefined
				image = tf.image.decode_jpeg(image_string, channels=3)
				image  = tf.image.crop_to_bounding_box(image, 0, 0, 256,256)
				image = tf.image.resize(image,[256,256])
				# This will convert to float values in [0, 1]
				image = tf.divide(image,255.0)
				image = tf.scalar_mul(2.0,image)
				image = tf.subtract(image,1.0)
				# image = tf.image.convert_image_dtype(image, tf.float16)
			return image

		if self.FID_load_flag == 0:
			### First time FID call setup
			self.FID_load_flag = 1	
			random_points = tf.keras.backend.random_uniform([self.FID_num_samples], minval=0, maxval=int(self.fid_train_images.shape[0]), dtype='int32', seed=None)
			print(random_points)
			self.fid_train_images_names = self.fid_train_images[random_points]

			## self.fid_train_images has the names to be read. Make a dataset with it
			self.fid_image_dataset = tf.data.Dataset.from_tensor_slices(self.fid_train_images_names)
			self.fid_image_dataset = self.fid_image_dataset.map(data_reader_faces,num_parallel_calls=int(self.num_parallel_calls))
			self.fid_image_dataset = self.fid_image_dataset.batch(self.fid_batch_size)

			# for element in self.fid_image_dataset:
			# 	self.fid_images = element
			# 	break
			if self.FID_kind != 'latent':
				self.Church_Classifier()


		if self.mode == 'fid':
			print(self.checkpoint_dir)
			# print('logs/130919_ELeGANt_mnist_lsgan_base_01/130919_ELeGANt_mnist_lsgan_base_Results_checkpoints')
			self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
			print('Models Loaded Successfully')

		with tf.device(self.device):
			for image_batch in self.fid_image_dataset:

				if self.FID_kind == 'latent':
					## Measure FID on the latent Gaussians 
					preds = self.Encoder(image_batch)
					act1 = self.get_noise(tf.constant(100))
					act2 = preds
				else:
					# print(self.fid_train_images.shape)
					preds = self.Decoder(self.get_noise(tf.constant(100)), training=False)
					# preds = tf.scalar_mul(2.,preds)
					# preds = tf.subtract(preds,1.0)
					preds = tf.image.resize(preds, [256,256])
					preds = preds.numpy()

					act1 = self.FID_model.predict(image_batch)
					act2 = self.FID_model.predict(preds)

				try:
					self.act1 = np.concatenate((self.act1,act1), axis = 0)
					self.act2 = np.concatenate((self.act2,act2), axis = 0)
				except:
					self.act1 = act1
					self.act2 = act2
			self.eval_FID()
			return

