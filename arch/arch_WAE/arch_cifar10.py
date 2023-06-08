from __future__ import print_function
import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import math

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
from tqdm import tqdm
import cleanfid
from cleanfid.downloads_helper import *
from cleanfid.inception_pytorch import InceptionV3
from cleanfid.inception_torchscript import InceptionV3W

class ARCH_cifar10():

	def __init__(self):
		print("CREATING ARCH_AAE CLASS")
		return



	def encdec_model_dcgan_cifar10(self):

		def ama_relu(x):
			x = tf.clip_by_value(x, clip_value_min = -5., clip_value_max = 5.)
			return x

		if self.loss in ['JS','SW', 'RBF']:
			init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.02, seed=None)
		elif self.loss in ['FS','KL','LP','ALP']:
			init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None)
		else:
			init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)
		bias_init_fn = tf.keras.initializers.Zeros()
		bias_init_fn = tf.function(bias_init_fn, autograph=False)


		inputs = tf.keras.Input(shape=(self.output_size,self.output_size,3)) #64x64x3

		enc1 = tf.keras.layers.Conv2D(128, 4, strides=2, padding='same',kernel_initializer=init_fn, use_bias=True, bias_initializer = bias_init_fn)(inputs) #32x32x64
		enc1 = tf.keras.layers.BatchNormalization()(enc1)
		# enc1 = tf.keras.layers.Dropout(0.1)(enc1)
		enc1 = tf.keras.layers.LeakyReLU()(enc1)
		# enc1 = tf.keras.layers.Activation( activation = 'tanh')(enc1)

		enc2 = tf.keras.layers.Conv2D(256, 4, strides=2, padding='same',kernel_initializer=init_fn, use_bias=True, bias_initializer = bias_init_fn)(enc1) #16x16x128
		enc2 = tf.keras.layers.BatchNormalization()(enc2)
		# enc2 = tf.keras.layers.Dropout(0.1)(enc2)
		enc2 = tf.keras.layers.LeakyReLU()(enc2)
		# enc2 = tf.keras.layers.Activation( activation = 'tanh')(enc2)

		enc3 = tf.keras.layers.Conv2D(512, 4, strides=2, padding='same',kernel_initializer=init_fn, use_bias=True, bias_initializer = bias_init_fn)(enc2) #8x8x256
		enc3 = tf.keras.layers.BatchNormalization()(enc3)
		# enc3 = tf.keras.layers.Dropout(0.1)(enc3)
		enc3 = tf.keras.layers.LeakyReLU()(enc3)
		# enc3 = tf.keras.layers.Activation( activation = 'tanh')(enc3)

		enc4 = tf.keras.layers.Conv2D(1024, 4, strides=2, padding='same',kernel_initializer=init_fn, use_bias=True, bias_initializer = bias_init_fn)(enc3) #4x4x128
		enc4 = tf.keras.layers.BatchNormalization()(enc4)
		# enc4 = tf.keras.layers.Dropout(0.5)(enc4)
		enc4 = tf.keras.layers.LeakyReLU()(enc4)
		# enc4 = tf.keras.layers.Activation( activation = 'tanh')(enc4)



		dense = tf.keras.layers.Flatten()(enc4)

		dense = tf.keras.layers.Dense(self.latent_dims, use_bias = True, kernel_initializer=init_fn, bias_initializer = bias_init_fn)(dense)
		enc = tf.keras.layers.Dense(self.latent_dims, use_bias = True, kernel_initializer=init_fn, bias_initializer = bias_init_fn)(dense)

		if self.loss in ['FS','KL'] or (self.testcase == 'single' and self.loss in ['KL', 'FS', 'JS', 'ALP', ]):
			enc  = tf.keras.layers.Lambda(ama_relu)(enc)


		encoded = tf.keras.Input(shape=(self.latent_dims,))

		den = tf.keras.layers.Dense(1024*int(self.output_size/8)*int(self.output_size/8),use_bias=True, bias_initializer = bias_init_fn)(encoded)
		enc_res = tf.keras.layers.Reshape([int(self.output_size/8),int(self.output_size/8),1024])(den)
		# enc_res = tf.keras.layers.Reshape([1,1,int(self.latent_dims)])(den) #1x1xlatent

		denc5 = tf.keras.layers.Conv2DTranspose(512, 4, strides=2,padding='same',kernel_initializer=init_fn,use_bias=True, bias_initializer = bias_init_fn)(enc_res) #2x2x128
		denc5 = tf.keras.layers.BatchNormalization()(denc5)
		# denc4 = tf.keras.layers.Dropout(0.5)(denc5)
		denc5 = tf.keras.layers.LeakyReLU()(denc5)
		# denc5 = tf.keras.layers.Activation( activation = 'tanh')(denc5)

		denc4 = tf.keras.layers.Conv2DTranspose(256, 4, strides=2,padding='same',kernel_initializer=init_fn,use_bias=True, bias_initializer = bias_init_fn)(denc5) #4x4x128
		denc4 = tf.keras.layers.BatchNormalization()(denc4)
		# denc4 = tf.keras.layers.Dropout(0.5)(denc4)
		denc4 = tf.keras.layers.LeakyReLU()(denc4)
		# denc4 = tf.keras.layers.Activation( activation = 'tanh')(denc4)

		denc3 = tf.keras.layers.Conv2DTranspose(128, 4, strides=2,padding='same',kernel_initializer=init_fn,use_bias=True, bias_initializer = bias_init_fn)(denc4) #8x8x256
		denc3 = tf.keras.layers.BatchNormalization()(denc3)
		# denc3 = tf.keras.layers.Dropout(0.5)(denc3)
		denc1 = tf.keras.layers.LeakyReLU()(denc3)
		# denc1 = tf.keras.layers.Activation( activation = 'tanh')(denc3)


		out = tf.keras.layers.Conv2DTranspose(3, 4,strides=1,padding='same', kernel_initializer=init_fn, use_bias = True, bias_initializer = bias_init_fn)(denc1) #64x64x3
		out =  tf.keras.layers.Activation(activation = 'tanh')(out)

	
		self.Decoder = tf.keras.Model(inputs=encoded, outputs=out)
		self.Encoder = tf.keras.Model(inputs=inputs, outputs=enc)

		return


	def discriminator_model_dcgan_cifar10(self):
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		model = tf.keras.Sequential()
		model.add(layers.Dense(256, use_bias=False, input_shape=(self.latent_dims,), kernel_initializer=init_fn))
		model.add(layers.BatchNormalization())
		# model.add(layers.Activation(activation = 'tanh'))
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(512, use_bias=False, kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(1025, use_bias=False, kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(1))

		if self.loss in ['KL']:
			model.add(layers.Activation(activation = 'sigmoid'))
		# model.add(layers.Softmax())
		return model


	def CIFAR10_Classifier(self):
		self.FID_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, pooling='avg', weights='imagenet', input_tensor=None, input_shape=(299,299,3), classes=1000)

	def same_images_FID(self):
		if self.FID_load_flag == 0:
			### First time FID call setup
			self.FID_load_flag = 1	
			random_points = tf.keras.backend.random_uniform([self.FID_num_samples], minval=0, maxval=int(self.fid_train_images.shape[0]), dtype='int32', seed=None)
			print(random_points)
			if self.FID_num_samples <50000:
				self.fid_train_images = self.fid_train_images[random_points]


			## self.fid_train_images has the names to be read. Make a dataset with it
			self.fid_image_dataset = tf.data.Dataset.from_tensor_slices(self.fid_train_images)
			# self.fid_image_dataset = self.fid_image_dataset.map(data_preprocess,num_parallel_calls=int(self.num_parallel_calls))
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

	def save_images_ReconFID(self):
		if self.ReconFID_load_flag == 0:
			### First time FID call setup
			self.ReconFID_load_flag = 1	
			random_points = tf.keras.backend.random_uniform([self.ReconFID_num_samples], minval=0, maxval=int(self.fid_train_images.shape[0]), dtype='int32', seed=None)
			print(random_points)
			if self.ReconFID_num_samples <50000:
				self.fid_train_images = self.fid_train_images[random_points]


			## self.fid_train_images has the names to be read. Make a dataset with it
			self.fid_image_dataset = tf.data.Dataset.from_tensor_slices(self.fid_train_images)
			# self.fid_image_dataset = self.fid_image_dataset.map(data_preprocess,num_parallel_calls=int(self.num_parallel_calls))
			self.fid_image_dataset = self.fid_image_dataset.batch(self.ReconFID_num_samples)

		with tf.device(self.device):
			for image_batch in self.fid_image_dataset:
				preds = self.Decoder(self.Encoder(image_batch, training = False), training=False)
				preds = preds.numpy()
				for i in range(self.ReconFID_num_samples):
					real = image_batch[i,:,:,:]
					tf.keras.preprocessing.image.save_img(self.ReconFIDRealspath+str(i)+'.png', real,  scale=True)
					fake = preds[i,:,:,:]
					tf.keras.preprocessing.image.save_img(self.ReconFIDFakespath+str(i)+'.png', fake,  scale=True)
		return

	def FID_cifar10(self):

		def data_preprocess(image):
			with tf.device('/CPU'):
				image = tf.image.resize(image,[299,299])
				# image = tf.scalar_mul(2.0,image)
				# image = tf.subtract(image,1.0)
			return image


		if self.FID_load_flag == 0:
			### First time FID call setup
			self.FID_load_flag = 1	
			random_points = tf.keras.backend.random_uniform([self.FID_num_samples], minval=0, maxval=int(self.fid_train_images.shape[0]), dtype='int32', seed=None)
			print(random_points)
			self.fid_train_images = self.fid_train_images[random_points]

			# self.fid_train_images_names = self.fid_train_images[random_points]

			## self.fid_train_images has the names to be read. Make a dataset with it
			self.fid_image_dataset = tf.data.Dataset.from_tensor_slices(self.fid_train_images)
			if self.FID_kind != 'latent':
				self.fid_image_dataset = self.fid_image_dataset.map(data_preprocess,num_parallel_calls=int(self.num_parallel_calls))
			self.fid_image_dataset = self.fid_image_dataset.batch(self.fid_batch_size)

			
			self.CIFAR10_Classifier()


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
					# print(self.fid_train_images.shape)
					preds = self.Decoder(self.get_noise(tf.constant(100)), training=False)
					# preds = preds[:,:,:].numpy()		
					preds = tf.image.resize(preds, [299,299])
					# preds = tf.subtract(preds,0.50)
					# preds = tf.scalar_mul(2.0,preds)
					preds = preds.numpy()

					act1 = self.FID_model.predict(image_batch)
					act2 = self.FID_model.predict(preds)
					
				try:
					self.act1 = np.concatenate([self.act1,act1], axis = 0)
					self.act2 = np.concatenate([self.act2,act2], axis = 0)
				except:
					self.act1 = act1
					self.act2 = act2
			# print(self.act1.shape, self.act2.shape)
			self.eval_FID()
			return

	def FID_torch_cifar10(self):

		def CIFAR10_torch_Classifier():
			model = InceptionV3W("/tmp", download=True, resize_inside=True).to(torch.device("cuda"))
			model.eval()
			def model_fn(x): return model(x)
			self.FID_model = model_fn

		def data_preprocess(image):
			with tf.device('/CPU'):
				image = tf.add(image,1.0)
				image = tf.scalar_mul(127.5,image)
				return image


		if self.FID_load_flag == 0:
			### First time FID call setup
			self.FID_load_flag = 1	
			random_points = tf.keras.backend.random_uniform([self.FID_num_samples], minval=0, maxval=int(self.fid_train_images.shape[0]), dtype='int32', seed=None)
			print(random_points)
			self.fid_train_images = self.fid_train_images[random_points]

			# self.fid_train_images_names = self.fid_train_images[random_points]

			## self.fid_train_images has the names to be read. Make a dataset with it
			self.fid_image_dataset = tf.data.Dataset.from_tensor_slices(self.fid_train_images)
			if self.FID_kind != 'latent':
				self.fid_image_dataset = self.fid_image_dataset.map(data_preprocess,num_parallel_calls=int(self.num_parallel_calls))
			self.fid_image_dataset = self.fid_image_dataset.batch(128)

			
			CIFAR10_torch_Classifier()


		if self.mode == 'fid':
			print(self.checkpoint_dir)
			self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
			print('Models Loaded Successfully')

		with tf.device(self.device):

			for image_batch in self.fid_image_dataset:
				torch.cuda.empty_cache()
				# print(self.fid_train_images.shape)
				preds = self.Decoder(self.get_noise(tf.constant(128)), training=False)
				# preds = preds[:,:,:].numpy()		
				# preds = tf.image.resize(preds, [299,299])
				preds = tf.add(preds,1.0)
				preds = tf.scalar_mul(127.5,preds)
				# preds = tf.subtract(preds,0.50)
				# preds = tf.scalar_mul(2.0,preds)
				# preds = preds.numpy()

				data1 = torch.from_numpy(tf.transpose(image_batch, [0, 3, 1, 2]).numpy())
				data2 = torch.from_numpy(tf.transpose(preds, [0, 3, 1, 2]).numpy())

				with torch.no_grad():
					feat1 = self.FID_model(data1.to(torch.device("cuda")))
					act1 = feat1.detach().cpu().numpy()
					feat2 = self.FID_model(data2.to(torch.device("cuda")))
					act2 = feat2.detach().cpu().numpy()
					
				try:
					self.act1 = np.concatenate([self.act1,act1], axis = 0)
					self.act2 = np.concatenate([self.act2,act2], axis = 0)
				except:
					self.act1 = act1
					self.act2 = act2
			# print(self.act1.shape, self.act2.shape)
			self.eval_FID()
			return
