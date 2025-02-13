##
# @file ADImage.py
# @brief Enscapsilation of an AreaDetector Image
# AreaDetector is a an EPICS package that is used to provide an EPICS framework
# for image detectors. This file will handle common operations to interact
# with it's image array, which comes as a 1D EPICS waveform. Also provides 
# class to contain a collection of image objects 

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.signal as sig

class ADImage():
	"""!ADImage class is the main container that will hold data from the area detector image
	
	"""
	
	def __init__(self , x_size, y_size , array_data , uid=None , max_pix_val=None ):
		"""!Constructor takes in the x and y sizes and data array
		@param x_size - the horizontal size of the image
		@param y_size - the vertical size of the image
		@param array_data - is the 1D or 2D image array
		@param uid - an optional id for the image
		@param max_pix_val - optional max pixel value
		
		"""
		# if uid or max_pix_val is not None then set the parameters
		if uid is not None:
			self.uid = uid
		else:
			self.uid = ""
			
		if max_pix_val is not None:
			self.max_pix_val = max_pix_val
			print("MAX PIX VAL = "+str(self.max_pix_val))
		else:
			self.max_pix_val = 255 # 8-bit image default
		
		#set the array
		self.set_array(array_data , x_size , y_size)
		
		

	def print_image_params(self):
		"""!Print the parameters"""
		print("x size = " + str( self._x_size ))
		print("y size = " + str( self._y_size ))
		print("array data = " + str( self.array_data ))
		print("unique id = " + str( self.uid ))
		print("x_profile length = " + str( len( self.x_profile ) ))
		#print("x_profile = " + str( self.x_profile ) )
		print("y_profile length = " + str( len( self.y_profile ) ))
		#print("y_profile = " + str( self.y_profile ) )
		print("data_array max = " + str( np.max( self.array_data ) ) )
		#print("2d data_array max = " + str( np.max( self.img_arr_2d ) ) )
		print("x_profile max = " + str( np.max( self.x_max_val ) ) )
		print("y_profile max = " + str( np.max( self.y_max_val ) ) )
		print("***********************************************************")
		

	def set_array(self , array_data , x_size , y_size):
		"""!Set the array data 
		Function looks at the shape of the array, and decides if it is already
		A 1D or 2D array, then sets the class members accordingly 
		
		"""
		
		#check the shape of array_data, if size 2 then it is a 2d array
		if len(array_data.shape)  == 2 :
			self._x_size=array_data.shape[1]
			self._y_size=array_data.shape[0]
			##tuple of size, useful for making an PIL image
			self._size=( self.x_size , self.y_size )
			self.img_arr_2d = np.array( array_data)
			self.array_data = self.img_arr_2d.flatten()
		else: #this is a 1d array
			self._x_size=x_size
			self._y_size=y_size
			##tuple of size, useful for making an PIL image
			self._size=( x_size , y_size )
			self.array_data = np.array( array_data )
			self.set_img_arr_2d(self.array_data)
		
		#set profiles, sum of horizontal and vertical columns
		self.set_x_profile()
		self.set_y_profile()
		
		#np arrays to use for plotting
		self.xdata = np.arange(self._x_size)
		self.ydata = np.arange(self._y_size)
		
			
	def get_x_size(self ):
		"""!getter for x_size"""
		return self._x_size

	def get_y_size(self):
		"""!getter for y_size"""
		return self._y_size
	
	def get_uid(self):
		"""!getter for uid"""
		return self.uid
	
	def set_img_arr_2d(self, array_data):
		"""!Set a 2D array, from a 1D array """
		if len(self.array_data.shape)  == 1 :
			self.img_arr_2d = array_data.reshape(  int( self._y_size) , int( self._x_size ) )
			#print(self.img_arr_2d)
			
	
	def set_x_profile(self, x_profile = None):
		"""!Set the x profile from the image array, optionally use this to set the profile"""
		if (x_profile is not None) and len(x_profile) == ( self._x_size ):
			self.x_profile = x_profile
		else:
			#self.x_profile =   np.sum( self.img_arr_2d , axis=0 ) / self.max_pix_val
			self.x_profile =   np.sum( self.img_arr_2d , axis=0 ) 
			#self.x_profile =  self.x_profile.astype(int)
		
		#find peaks in ydata
		self.x_peaks, _ = sig.find_peaks( self.x_profile )
		#print(str(peaks))
		#find valleys by inverting the data
		self.x_valleys, _ =sig.find_peaks(-self.x_profile)	
		#set the max value of the profile
		self.set_x_profile_max()
			
	def set_y_profile(self, y_profile = None):
		"""!Set the y profile from the image array, optionally use this to set the profile"""
		if (y_profile is not None) and len(y_profile) == ( self._y_size ):
			self.y_profile = y_profile
		else:
			self.y_profile =   np.sum( self.img_arr_2d , axis=1 ) / self.max_pix_val
		#find peaks in ydata
		self.y_peaks, _ = sig.find_peaks( self.y_profile )
		#print(str(peaks))
		#find valleys by inverting the data
		self.y_valleys, _ = sig.find_peaks(-self.y_profile)		
		#set the profile max value
		self.set_y_profile_max()
	
	def set_x_profile_max(self):
		"""!set the max number for the x profile, and the index it is found"""
		if(self.x_profile is not None) and (self.x_peaks is not None):
			self.x_max_val = np.max(self.x_profile[self.x_peaks])
			self.x_max_index = self.x_peaks[0]
			
			for index in self.x_peaks:
				if self.x_profile[index] == self.x_max_val:
					self.x_max_index = index 
			#print( "MAX X = " + str( self.x_max_val ) + ",  MAX X INDEX =" + str(self.x_max_index))
	
	def set_y_profile_max(self):
		"""!set the max number for the y profile, and the index it is found"""
		if(self.y_profile is not None) and (self.y_peaks is not None):
			self.y_max_val = np.max(self.y_profile[self.y_peaks])
			self.y_max_index = self.y_peaks[0]
			
			for index in self.y_peaks:
				if self.y_profile[index] == self.y_max_val:
					self.y_max_index = index 
			#print( "MAX Y = " + str( self.y_max_val ) + ",  MAX Y INDEX =" + str(self.y_max_index))
	
	def get_x_profile(self):
		"""!getter for the x profile"""
		return self.x_profile
		
	def get_y_profile(self):
		"""!getter for y profile"""
		return self.y_profile
	
	def save_2d_to_image(self , path=None):
		"""!Save the 2d array to an image, this one still needs some work
		@param path - path to where we want to save the image 
		"""
		p="./"
		if path is not None:
			p=path
		
		name="image_"+str(self.uid)
		im = Image.fromarray(self.img_arr_2d , mode='L')
		print("path = '" + p+name+".jpg")
		im.save(p+name+".jpg" )
		
	def xplot(self):
		"""!Save the x profile to a png image.
		"""
		plt.cla()
		plt.plot(self.xdata, self.x_profile)
		plt.plot(self.x_peaks, peaks, self.x_profile[self.x_peaks], 'o')
		plt.plot(self.x_valleys, self.x_profile[self.x_valleys], 'o')
		plt.savefig("./data/xplot_"+str(self.uid)+".png")
		#plt.show()

	
	def yplot(self):
		"""!Save the y profile to a png image.
		"""
		
		plt.cla()
		plt.plot(self.ydata, self.y_profile)
		plt.plot(self.y_peaks, self.y_profile[self.y_peaks], 'o')
		plt.plot(self.y_valleys, self.y_profile[self.y_valleys], 'o')
		plt.savefig("./data/yplot_"+str(self.uid)+".png")
	

class ADImageCollection():
	
	"""!Convience class to store ADImages and perform Operations on set of them
	"""
	
	##array counter
	counter=0
	
	def __init__(self):
		"""! Initiate a list object to store images
		"""
		self.ad_collection = list()
		
	def add(self, ad_image):
		"""!Add an ADImage to the collection"""
		self.ad_collection.append( ad_image )
		#increment the image counter
		self.counter=self.counter+1
		
	
	def clear(self):
		#clear the list
		#self.length = len(self.ad_collection)
		self.ad_collection.clear()
		
	
	
	def average(self):
		"""!This function will average the arrays in the collection
		@return an average image, as an ADImage object
		"""
		
		self.arr = list()
		self.num_non_zero = 0
		
		#make an array of image arrays
		for i,img in enumerate( self.ad_collection ):
			if(img.array_data is not None) and (np.any(img.array_data)): 
				self.arr.append(img.array_data)
				self.num_non_zero = self.num_non_zero + 1
		
		#changing the data type here is important, as it will limit the to 8 bit int values
		#need to change this to a more dynamic method for figuring out the type
		self.avg_arr = np.array( self.arr[0] , dtype=float)
		
		#iterate through the array and add them up
		for i , arr_1d in enumerate(self.arr):
			if i > 0: 
				tmp = np.array( self.avg_arr )
				self.avg_arr = self.arr[i] + tmp 
				
	
		#if there was more than 1, then give the average else return the 1st element
		if(len( self.ad_collection) > 1):
			self.x_size = self.ad_collection[0].get_x_size()
			self.y_size = self.ad_collection[0].get_y_size()
			#set the average
			self.avg_1d = self.avg_arr/self.num_non_zero
			#self.ad_avg_img.print_image_params() 
			return ADImage(self.x_size , self.y_size , self.avg_1d  )
		else:
			
			return self.ad_collection[0]
		
	
		
		
	
