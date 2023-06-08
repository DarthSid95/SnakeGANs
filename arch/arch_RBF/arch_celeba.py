from __future__ import print_function
import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class ARCH_celeba():

	def __init__(self):
		print("Creating CelebA architectures for base cases ")
		return
	def generator_model_aedcgan_celeba(self):

		# init_fn = tf.keras.initializers.Identity()
		# init_fn = tf.function(init_fn, autograph=False)
		# FOr regular comarisons when 2m-n = 0
		# init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.075, seed=None) 
		# FOr comparisons when 2m-n > 0 
		# init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None)
		# FOr comparisons when 2m-n < 0 

		# init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.1, seed=None)
		# init_fn = tf.function(init_fn, autograph=False)
		# bias_init_fn = tf.keras.initializers.Zeros()
		# bias_init_fn = tf.function(bias_init_fn, autograph=False)

		init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.075, seed=None)
		init_fn = tf.function(init_fn, autograph=False)
		bias_init_fn = tf.keras.initializers.glorot_uniform()#tf.keras.initializers.Zeros()
		bias_init_fn = tf.function(bias_init_fn, autograph=False)
		
		inputs = tf.keras.Input(shape=(self.noise_dims,))

		enc0 = tf.keras.layers.Dense(32*32*3, kernel_initializer=init_fn, use_bias = True, bias_initializer = bias_init_fn)(inputs)
		enc0 = tf.keras.layers.LeakyReLU()(enc0)

		enc_res = tf.keras.layers.Reshape([32,32,3])(enc0)

		enc1 = tf.keras.layers.Conv2D(1024, 4, strides=2, padding='same',kernel_initializer=init_fn, use_bias=True, bias_initializer = bias_init_fn)(enc_res) #16x16x64
		enc1 = tf.keras.layers.BatchNormalization()(enc1)
		# enc1 = tf.keras.layers.Dropout(0.1)(enc1)
		enc1 = tf.keras.layers.LeakyReLU()(enc1)

		enc2 = tf.keras.layers.Conv2D(256, 4, strides=2, padding='same',kernel_initializer=init_fn, use_bias=True, bias_initializer = bias_init_fn)(enc1) #8x8x128
		enc2 = tf.keras.layers.BatchNormalization()(enc2)
		# enc2 = tf.keras.layers.Dropout(0.1)(enc2)
		enc2 = tf.keras.layers.LeakyReLU()(enc2)


		enc3 = tf.keras.layers.Conv2D(128, 4, strides=2, padding='same',kernel_initializer=init_fn, use_bias=True, bias_initializer = bias_init_fn)(enc2) #4x4x256
		enc3 = tf.keras.layers.BatchNormalization()(enc3)
		# enc3 = tf.keras.layers.Dropout(0.1)(enc3)
		enc3 = tf.keras.layers.LeakyReLU()(enc3)


		enc4 = tf.keras.layers.Conv2D(128, 4, strides=2, padding='same',kernel_initializer=init_fn, use_bias=True, bias_initializer = bias_init_fn)(enc3) #2x2x128
		enc4 = tf.keras.layers.BatchNormalization()(enc4)
		# enc4 = tf.keras.layers.Dropout(0.5)(enc4)
		enc4 = tf.keras.layers.LeakyReLU()(enc4)

		enc5 = tf.keras.layers.Conv2D(int(self.latent_dims), 4, strides=2, padding='same',kernel_initializer=init_fn, use_bias=True, bias_initializer = bias_init_fn)(enc4) #1x1xlatent
		# enc5 = tf.keras.layers.BatchNormalization()(enc5)

		enc = tf.keras.layers.Flatten()(enc5)
	
		enc = tf.keras.layers.Dense(self.latent_dims, kernel_initializer=init_fn, use_bias = True, bias_initializer = bias_init_fn)(enc)

		model = tf.keras.Model(inputs = inputs, outputs = enc)

		return model

	def discriminator_model_aedcgan_celeba(self):
		inputs = tf.keras.Input(shape=(self.output_size,))
		w0_nt_x = tf.keras.layers.Dense(self.L, activation=None, use_bias = False)(inputs)
		w0_nt_x2 = tf.math.scalar_mul(2., w0_nt_x)
		cos_terms = tf.keras.layers.Activation( activation = tf.math.cos)(w0_nt_x)
		sin_terms = tf.keras.layers.Activation( activation = tf.math.sin)(w0_nt_x)
		cos2_terms  = tf.keras.layers.Activation( activation = tf.math.cos)(w0_nt_x2)

		cos_sum = tf.keras.layers.Dense(1, activation=None, use_bias = True)(cos_terms)
		sin_sum = tf.keras.layers.Dense(1, activation=None, use_bias = False)(sin_terms)

		cos2_c_sum = tf.keras.layers.Dense(1, activation=None, use_bias = False)(cos2_terms) #Tau_c weights
		cos2_s_sum = tf.keras.layers.Dense(1, activation=None, use_bias = False)(cos2_terms) #Tau_s weights

		lambda_x_term = tf.keras.layers.Subtract()([cos2_s_sum, cos2_c_sum]) #(tau_s  - tau_r)
		if self.homo_flag:
			if self.latent_dims == 1:
				Out = tf.keras.layers.Add()([cos_sum, sin_sum, tf.divide(tf.abs(inputs),2.)])
			elif self.latent_dims >= 2:
				Out = tf.keras.layers.Add()([cos_sum, sin_sum, tf.divide(tf.reduce_sum(inputs,axis=-1,keepdims=True),self.latent_dims)])
		else:
			Out = tf.keras.layers.Add()([cos_sum, sin_sum])
		model = tf.keras.Model(inputs=inputs, outputs=[Out,lambda_x_term])
		return model



	def generator_model_aedense_celeba(self):
		init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.5, seed=None)
		# else:
		# 	tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)
		bias_init_fn = tf.keras.initializers.Zeros()
		bias_init_fn = tf.function(bias_init_fn, autograph=False)

		inputs = tf.keras.Input(shape=(self.noise_dims,))

		enc1 = tf.keras.layers.Dense(1024, kernel_initializer=init_fn, use_bias = True, bias_initializer = bias_init_fn)(inputs)
		enc1 = tf.keras.layers.LeakyReLU()(enc1)

		enc2 = tf.keras.layers.Dense(512, kernel_initializer=init_fn, use_bias = True, bias_initializer = bias_init_fn)(enc1)
		enc2 = tf.keras.layers.LeakyReLU()(enc2)

		enc3 = tf.keras.layers.Dense(256, kernel_initializer=init_fn, use_bias = True, bias_initializer = bias_init_fn)(enc2)
		enc3 = tf.keras.layers.LeakyReLU()(enc3)

		enc4 = tf.keras.layers.Dense(128, kernel_initializer=init_fn, use_bias = True, bias_initializer = bias_init_fn)(enc3)
		enc4 = tf.keras.layers.LeakyReLU()(enc4)

		enc5 = tf.keras.layers.Dense(64, kernel_initializer=init_fn, use_bias = False)(enc4)
		enc5 = tf.keras.layers.LeakyReLU()(enc4)

		enc6 = tf.keras.layers.Dense(self.latent_dims, kernel_initializer=init_fn, use_bias = False)(enc5)
		# enc6 = tf.keras.layers.LeakyReLU()(enc6)

		model = tf.keras.Model(inputs = inputs, outputs = enc6)

		return model


	def discriminator_model_aedense_celeba(self):
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)


		model = tf.keras.Sequential()
		model.add(layers.Dense(512, use_bias=False, input_shape=(2,), kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())


		model.add(layers.Dense(256, use_bias=False, kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(64, use_bias=True, kernel_initializer=init_fn))
		# # model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(32, use_bias=True, kernel_initializer=init_fn))

		model.add(layers.Dense(1))
		# model.add(layers.Softmax())

		return model




	def generator_model_dcgan_celeba(self):
		init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None)
		# init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		inputs = tf.keras.Input(shape = (self.noise_dims,))

		dec1 = tf.keras.layers.Dense(int(self.output_size/16)*int(self.output_size/16)*1024, kernel_initializer=init_fn, use_bias=False)(inputs)		
		dec1 = tf.keras.layers.LeakyReLU()(dec1)

		un_flat = tf.keras.layers.Reshape([int(self.output_size/16),int(self.output_size/16),1024])(dec1) #4x4x1024

		deconv1 = tf.keras.layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=init_fn)(un_flat) #8x8x512
		deconv1 = tf.keras.layers.BatchNormalization()(deconv1)
		deconv1 = tf.keras.layers.LeakyReLU()(deconv1)

		deconv2 = tf.keras.layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=init_fn)(deconv1) #16x16x256
		deconv2 = tf.keras.layers.BatchNormalization()(deconv2)
		deconv2 = tf.keras.layers.LeakyReLU()(deconv2)

		deconv4 = tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=init_fn)(deconv2) #32x32x128
		deconv4 = tf.keras.layers.BatchNormalization()(deconv4)
		deconv4 = tf.keras.layers.LeakyReLU()(deconv4)

		out = tf.keras.layers.Conv2DTranspose(self.output_dims, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=init_fn, activation = 'tanh')(deconv4) #64x64x3

		model = tf.keras.Model(inputs = inputs, outputs = out)
		return model

	def discriminator_model_dcgan_celeba(self):
		# init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None)
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		model = tf.keras.Sequential() #64x64x3
		model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init_fn,input_shape=[self.output_size, self.output_size, 3])) #32x32x64
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init_fn)) #16x16x128
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init_fn)) #8x8x256
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init_fn)) #4x4x512
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Flatten()) #8192x1
		model.add(layers.Dense(1)) #1x1

		return model


	def same_images_FID(self):
		import glob

		def data_reader_faces(filename):
			with tf.device('/CPU'):
				print(tf.cast(filename[0],dtype=tf.string))
				image_string = tf.io.read_file(tf.cast(filename[0],dtype=tf.string))
				# Don't use tf.image.decode_image, or the output shape will be undefined
				image = tf.image.decode_jpeg(image_string, channels=3)
				image.set_shape([218,178,3])
				image  = tf.image.crop_to_bounding_box(image, 38, 18, 140,140)
				image = tf.image.resize(image,[self.output_size,self.output_size])
				image = tf.divide(image,255.0)
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
			for i in range(self.FID_num_samples):	
				preds = self.generator(self.get_noise([2, self.noise_dims]), training=False)
				if self.gan in ['WGAN_AE']:
					preds = self.Decoder(preds, training = False)
				preds = preds.numpy()
				fake = preds[0,:,:,:]
				tf.keras.preprocessing.image.save_img(self.FIDFakespath+str(i)+'.png', fake,  scale=True)

			cur_num_reals = len(glob.glob(self.FIDRealspath))
			if cur_num_reals < self.FID_num_samples:
				for image_batch in self.fid_image_dataset:
					for i in range(self.FID_num_samples):
						real = image_batch[i,:,:,:]
						tf.keras.preprocessing.image.save_img(self.FIDRealspath+str(i)+'.png', real,  scale=True)
		return

	def save_interpol_figs(self):
		num_interps = 11
		from scipy.interpolate import interp1d
		with tf.device(self.device):
			for i in range(self.FID_num_samples):
				# self.FID_num_samples
				start = self.get_noise([1, self.noise_dims])#current_batch[i:1+i].numpy()
				end = self.get_noise([1, self.noise_dims]) #current_batch[num_interps+i:num_interps+1+i].numpy()
				stack = np.vstack([start, end])

				linfit = interp1d([1,num_interps+1], stack, axis=0)
				interp_latents = linfit(list(range(1,num_interps+1)))

				# print(interp_latents.shape)
				mid = interp_latents[5:6]
				mid_img = self.generator(mid)
				mid_img = (mid_img + 1.0)/2.0
				mid_img = mid_img.numpy()
				mid_img = mid_img[0,:,:,:]
				tf.keras.preprocessing.image.save_img(self.FIDInterpolpath+str(i)+'.png', mid_img,  scale=True)
		return
	

	def CelebA_Classifier(self):
		self.FID_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, pooling='avg', weights='imagenet', input_tensor=None, input_shape=(80,80,3), classes=1000)

	def FID_celeba(self):

		def data_reader_faces(filename):
			with tf.device('/CPU'):
				print(tf.cast(filename[0],dtype=tf.string))
				image_string = tf.io.read_file(tf.cast(filename[0],dtype=tf.string))
				# Don't use tf.image.decode_image, or the output shape will be undefined
				image = tf.image.decode_jpeg(image_string, channels=3)
				image.set_shape([218,178,3])
				image  = tf.image.crop_to_bounding_box(image, 38, 18, 140,140)
				image = tf.image.resize(image,[80,80])
				# This will convert to float values in [0, 1]
				image = tf.divide(image,255.0)
				image = tf.scalar_mul(2.0,image)
				image = tf.subtract(image,1.0)
				# image = tf.image.convert_image_dtype(image, tf.float16)
			return image

		if self.FID_load_flag == 0:
			### First time FID call setup
			self.FID_load_flag = 1	
			if self.testcase in ['bald', 'hat']:
				self.fid_train_images_names = self.fid_train_images
			else:
				random_points = tf.keras.backend.random_uniform([self.FID_num_samples], minval=0, maxval=int(self.fid_train_images.shape[0]), dtype='int32', seed=None)
				print(random_points)
				self.fid_train_images_names = self.fid_train_images[random_points]

			## self.fid_train_images has the names to be read. Make a dataset with it
			self.fid_image_dataset = tf.data.Dataset.from_tensor_slices(self.fid_train_images_names)
			self.fid_image_dataset = self.fid_image_dataset.map(data_reader_faces,num_parallel_calls=int(self.num_parallel_calls))
			self.fid_image_dataset = self.fid_image_dataset.batch(self.fid_batch_size)
			self.CelebA_Classifier()


		with tf.device(self.device):
			for image_batch in self.fid_image_dataset:
				# noise = tf.random.normal([self.fid_batch_size, self.noise_dims],self.noise_mean, self.noise_stddev)
				noise = self.get_noise([self.fid_batch_size, self.noise_dims])
				preds = self.generator(noise, training=False)
				preds = tf.image.resize(preds, [80,80])
				preds = tf.scalar_mul(2.,preds)
				preds = tf.subtract(preds,1.0)
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