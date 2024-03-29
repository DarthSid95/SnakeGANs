from __future__ import print_function
import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import math

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt



class ARCH_ukiyoe():

	def __init__(self):
		print("CREATING ARCH_AAE CLASS")
		return
	def encdec_model_deepdcgan_ukiyoe(self):

		def ama_relu(x):
			x = tf.clip_by_value(x, clip_value_min = -6., clip_value_max = 6.)
			return x

		if self.loss in ['SW','JS','KL','LP','ALP']:
			init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.02, seed=None)
		if self.loss in ['FS']:
			init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.001, seed=None)
		else:
			init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None)
		init_fn = tf.function(init_fn, autograph=False)
		bias_init_fn = tf.keras.initializers.Zeros()
		bias_init_fn = tf.function(bias_init_fn, autograph=False)

		inputs = tf.keras.Input(shape=(self.output_size,self.output_size,3)) #512x512x3

		enc1 = tf.keras.layers.Conv2D(128, 4, strides=2, padding='same',kernel_initializer=init_fn, use_bias=False)(inputs) #256x256x128
		enc1 = tf.keras.layers.BatchNormalization()(enc1)
		# enc1 = tf.keras.layers.Dropout(0.1)(enc1)
		enc1 = tf.keras.layers.LeakyReLU()(enc1)
		# enc1 = tf.keras.layers.Activation( activation = 'tanh')(enc1) 

		enc2 = tf.keras.layers.Conv2D(256, 4, strides=2, padding='same',kernel_initializer=init_fn, use_bias=False)(enc1) #128x128x256
		enc2 = tf.keras.layers.BatchNormalization()(enc2)
		# enc2 = tf.keras.layers.Dropout(0.1)(enc2)
		enc2 = tf.keras.layers.LeakyReLU()(enc2)
		# enc2 = tf.keras.layers.Activation( activation = 'tanh')(enc2)


		enc3 = tf.keras.layers.Conv2D(512, 4, strides=2, padding='same',kernel_initializer=init_fn, use_bias=False)(enc2) #64x64x512
		enc3 = tf.keras.layers.BatchNormalization()(enc3)
		# enc3 = tf.keras.layers.Dropout(0.1)(enc3)
		enc3 = tf.keras.layers.LeakyReLU()(enc3)
		# enc3 = tf.keras.layers.Activation( activation = 'tanh')(enc3)


		enc4 = tf.keras.layers.Conv2D(512, 4, strides=2, padding='same',kernel_initializer=init_fn, use_bias=False)(enc3) #32x32x512
		enc4 = tf.keras.layers.BatchNormalization()(enc4)
		# enc4 = tf.keras.layers.Dropout(0.5)(enc4)
		enc4 = tf.keras.layers.LeakyReLU()(enc4)
		# enc4 = tf.keras.layers.Activation( activation = 'tanh')(enc4)


		enc5 = tf.keras.layers.Conv2D(512, 4, strides=2, padding='same',kernel_initializer=init_fn, use_bias=False)(enc4) #16x16x512
		enc5 = tf.keras.layers.BatchNormalization()(enc5)
		# enc5 = tf.keras.layers.Dropout(0.5)(enc5)
		enc5 = tf.keras.layers.LeakyReLU()(enc5)
		# enc5 = tf.keras.layers.Activation( activation = 'tanh')(enc5)

		enc6 = tf.keras.layers.Conv2D(512, 4, strides=2, padding='same',kernel_initializer=init_fn, use_bias=False)(enc5) #8x8x512
		enc6 = tf.keras.layers.BatchNormalization()(enc6)
		# enc6 = tf.keras.layers.Dropout(0.5)(enc6)
		enc6 = tf.keras.layers.LeakyReLU()(enc6)
		# enc6 = tf.keras.layers.Activation( activation = 'tanh')(enc6)


		enc7 = tf.keras.layers.Conv2D(1024, 4, strides=2, padding='same',kernel_initializer=init_fn, use_bias=False)(enc6) #4x4x256
		enc7 = tf.keras.layers.BatchNormalization()(enc7)
		# enc7 = tf.keras.layers.Dropout(0.5)(enc7)
		enc7 = tf.keras.layers.LeakyReLU()(enc7)
		# enc7 = tf.keras.layers.Activation( activation = 'tanh')(enc7)

		### 4x4x256 => 4096
		enc = dense = tf.keras.layers.Flatten()(enc7)

		# dense = tf.keras.layers.Dense(self.latent_dims, kernel_initializer = init_fn, use_bias = True, bias_initializer = bias_init_fn)(dense)
		# enc = tf.keras.layers.Dense(self.latent_dims, kernel_initializer = init_fn, use_bias = True, bias_initializer = bias_init_fn)(dense)

		if self.loss in ['FS','JS','KL','LP','ALP']:
			enc  = tf.keras.layers.Lambda(ama_relu)(enc)


		encoded = tf.keras.Input(shape=(self.latent_dims,))

		den = tf.keras.layers.Dense(1024*int(self.output_size/128)*int(self.output_size/128), kernel_initializer = init_fn, use_bias = True, bias_initializer = bias_init_fn)(encoded)
		enc_res = tf.keras.layers.Reshape([int(self.output_size/128),int(self.output_size/128),1024])(den) ### 4x4x256

		denc5 = tf.keras.layers.Conv2DTranspose(512, 4, strides=2,padding='same',kernel_initializer=init_fn,use_bias=False)(enc_res) #8x8x512
		denc5 = tf.keras.layers.BatchNormalization()(denc5)
		# denc4 = tf.keras.layers.Dropout(0.5)(denc5)
		denc5 = tf.keras.layers.LeakyReLU()(denc5)
		# denc5 = tf.keras.layers.Activation( activation = 'tanh')(denc5)

		denc5 = tf.keras.layers.Conv2DTranspose(512, 4, strides=2,padding='same',kernel_initializer=init_fn,use_bias=False)(denc5) #8x8x512
		denc5 = tf.keras.layers.BatchNormalization()(denc5)
		# denc4 = tf.keras.layers.Dropout(0.5)(denc5)
		denc5 = tf.keras.layers.LeakyReLU()(denc5)
		# denc5 = tf.keras.layers.Activation( activation = 'tanh')(denc5)

		denc5 = tf.keras.layers.Conv2DTranspose(512, 4, strides=2,padding='same',kernel_initializer=init_fn,use_bias=False)(denc5) #16x16x512
		denc5 = tf.keras.layers.BatchNormalization()(denc5)
		# denc4 = tf.keras.layers.Dropout(0.5)(denc5)
		denc5 = tf.keras.layers.LeakyReLU()(denc5)
		# denc5 = tf.keras.layers.Activation( activation = 'tanh')(denc5)

		denc5 = tf.keras.layers.Conv2DTranspose(512, 4, strides=2,padding='same',kernel_initializer=init_fn,use_bias=False)(denc5) #32x32x512
		denc5 = tf.keras.layers.BatchNormalization()(denc5)
		# denc4 = tf.keras.layers.Dropout(0.5)(denc5)
		denc5 = tf.keras.layers.LeakyReLU()(denc5)
		# denc5 = tf.keras.layers.Activation( activation = 'tanh')(denc5)


		denc5 = tf.keras.layers.Conv2DTranspose(512, 4, strides=2,padding='same',kernel_initializer=init_fn,use_bias=False)(denc5) #64x64x512
		denc5 = tf.keras.layers.BatchNormalization()(denc5)
		# denc4 = tf.keras.layers.Dropout(0.5)(denc5)
		denc5 = tf.keras.layers.LeakyReLU()(denc5)
		# denc5 = tf.keras.layers.Activation( activation = 'tanh')(denc5)

		denc4 = tf.keras.layers.Conv2DTranspose(256, 4, strides=2,padding='same',kernel_initializer=init_fn,use_bias=False)(denc5) #128x128x256
		denc4 = tf.keras.layers.BatchNormalization()(denc4)
		# denc4 = tf.keras.layers.Dropout(0.5)(denc4)
		denc4 = tf.keras.layers.LeakyReLU()(denc4)
		# denc4 = tf.keras.layers.Activation( activation = 'tanh')(denc4)

		denc3 = tf.keras.layers.Conv2DTranspose(128, 4, strides=2,padding='same',kernel_initializer=init_fn,use_bias=False)(denc4) #256x256x128
		denc3 = tf.keras.layers.BatchNormalization()(denc3)
		# denc3 = tf.keras.layers.Dropout(0.5)(denc3)
		denc1 = tf.keras.layers.LeakyReLU()(denc3)
		# denc1 = tf.keras.layers.Activation( activation = 'tanh')(denc3)

		out = tf.keras.layers.Conv2DTranspose(3, 4,strides=1,padding='same', kernel_initializer=init_fn, use_bias = True, bias_initializer = bias_init_fn)(denc1) #512x512x3
		out =  tf.keras.layers.Activation( activation = 'tanh')(out)

		self.Decoder = tf.keras.Model(inputs=encoded, outputs=out)
		self.Encoder = tf.keras.Model(inputs=inputs, outputs=enc)

		return



	def discriminator_model_deepdcgan_ukiyoe(self):
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
		if self.loss in ['KL']:
			model.add(layers.Activation(activation = 'sigmoid'))
		# model.add(layers.Softmax())
		return model


	def encdec_model_dcgan_ukiyoe(self):

		def ama_relu(x):
			x = tf.clip_by_value(x, clip_value_min = -6., clip_value_max = 6.)
			return x

		if self.loss in ['SW','JS','KL','LP','ALP']:
			init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.02, seed=None)
		if self.loss in ['FS']:
			init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.001, seed=None)
		else:
			init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None)
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



		dense = tf.keras.layers.Flatten()(enc4)

		dense = tf.keras.layers.Dense(self.latent_dims, kernel_initializer = init_fn, use_bias = True, bias_initializer = bias_init_fn)(dense)
		enc = tf.keras.layers.Dense(self.latent_dims, kernel_initializer = init_fn, use_bias = True, bias_initializer = bias_init_fn)(dense)

		if self.loss in ['FS','JS','KL','LP','ALP']:
			enc  = tf.keras.layers.Lambda(ama_relu)(enc)


		encoded = tf.keras.Input(shape=(self.latent_dims,))

		den = tf.keras.layers.Dense(1024*int(self.output_size/8)*int(self.output_size/8), kernel_initializer = init_fn, use_bias = True, bias_initializer = bias_init_fn)(encoded)
		enc_res = tf.keras.layers.Reshape([int(self.output_size/8),int(self.output_size/8),1024])(den)
		# enc_res = tf.keras.layers.Reshape([1,1,int(self.latent_dims)])(den) #1x1xlatent

		denc5 = tf.keras.layers.Conv2DTranspose(512, 4, strides=2,padding='same',kernel_initializer=init_fn,use_bias=False)(enc_res) #2x2x128
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



	def discriminator_model_dcgan_ukiyoe(self):
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
		if self.loss in ['KL']:
			model.add(layers.Activation(activation = 'sigmoid'))
		# model.add(layers.Softmax())
		return model


	def UkiyoE_Classifier(self):
		self.FID_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, pooling='avg', weights='imagenet', input_tensor=None, input_shape=(299,299,3), classes=1000)

	def same_images_FID(self):

		def data_reader_faces(filename):
			with tf.device('/CPU'):
				print(tf.cast(filename[0],dtype=tf.string))
				image_string = tf.io.read_file(tf.cast(filename[0],dtype=tf.string))
				# Don't use tf.image.decode_image, or the output shape will be undefined
				image = tf.image.decode_jpeg(image_string, channels=3)
				image.set_shape([1024,1024,3])
				# image  = tf.image.crop_to_bounding_box(image, 38, 18, 140,140)
				image = tf.image.resize(image,[self.output_size,self.output_size])
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
					tf.keras.preprocessing.image.save_img(self.FIDRealspath+str(i)+'.png', real,  scale=True)
			for i in range(self.FID_num_samples):	
				preds = self.Decoder(self.get_noise(2), training=False)
				preds = preds.numpy()
				fake = preds[0,:,:,:]
				tf.keras.preprocessing.image.save_img(self.FIDFakespath+str(i)+'.png', fake,  scale=True)
		return

	def FID_ukiyoe(self):

		def data_reader_faces(filename):
			with tf.device('/CPU'):
				print(tf.cast(filename[0],dtype=tf.string))
				image_string = tf.io.read_file(tf.cast(filename[0],dtype=tf.string))
				# Don't use tf.image.decode_image, or the output shape will be undefined
				image = tf.image.decode_jpeg(image_string, channels=3)
				image.set_shape([1024,1024,3])
				# image  = tf.image.crop_to_bounding_box(image, 38, 18, 140,140)
				if self.FID_kind != 'latent':
					image = tf.image.resize(image,[299,299])
				else:
					image = tf.image.resize(image,[self.output_size,self.output_size])
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


			if self.FID_kind != 'latent':
				self.UkiyoE_Classifier()


		if self.mode == 'fid':
			print(self.checkpoint_dir)

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
					preds = self.Decoder(self.get_noise(tf.constant(100)), training=False)
					preds = tf.image.resize(preds, [299,299])
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

