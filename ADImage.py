import numpy as np
from PIL import Image
#import cv2 as cv
#print("open cv version" + cv.__version__)
import matplotlib.pyplot as plt
import scipy.signal as sig



"""

Container for Image Data

This should be responsible for updating its own plots

"""

class ADImage():
	x_size=0
	y_size=0
	uid=0
	dtype=""
	max_pix_val=255
	
	def __init__(self , x_size, y_size , array_data , uid=None , max_pix_val=None ):
		
		if uid is not None:
			self.uid = uid
		if max_pix_val is not None:
			self.max_pix_val = max_pix_val
			print("MAX PIX VAL = "+str(self.max_pix_val))
		
		#check the shape of array_data, if size 2 then it is a 2d array
		if len(array_data.shape)  == 2 :
			self.x_size=array_data.shape[1]
			self.y_size=array_data.shape[0]
			self.size=( self.x_size , self.y_size )
			self.img_arr_2d = np.array( array_data)
			self.array_data = self.img_arr_2d.flatten()
		else: #this is a 1d array
			self.x_size=x_size
			self.y_size=y_size
			self.size=( x_size , y_size )
			self.array_data = np.array( array_data )
			self.set_img_arr_2d(self.array_data)
		
		#set profiles
		self.set_x_profile()
		self.set_y_profile()
		
		#np arrays to use for plotting
		self.xdata = np.arange(self.x_size)
		self.ydata = np.arange(self.y_size)
		
		

	def print_image_params(self):
		print("x size = " + str( self.x_size ))
		print("y size = " + str( self.y_size ))
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
		

	def set_array(self , array_data):
		self.array_data = array_data
		self.set_img_arr_2d(self.array_data)
			
	def set_x_size(self , x_size):
		self.x_size = x_size

	def set_y_size(self , y_size):
		self.y_size = y_size
	
	def set_uid(self , uid):
		self.uid = uid
	
	def set_img_arr_2d(self, array_data):
		if ( array_data is not None) and (self.x_size is not None ) and ( self.y_size is not None ):
			self.img_arr_2d = array_data.reshape(  int( self.y_size) , int( self.x_size ) )
			#print(self.img_arr_2d)
			
	
	def set_x_profile(self, x_profile = None):
		if (x_profile is not None) and len(x_profile) == ( x_size ):
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
		
		self.set_x_profile_max()
			
	def set_y_profile(self, y_profile = None):
		if (y_profile is not None) and len(y_profile) == ( y_size ):
			self.y_profile = y_profile
		else:
			self.y_profile =   np.sum( self.img_arr_2d , axis=1 ) / self.max_pix_val
		#find peaks in ydata
		self.y_peaks, _ = sig.find_peaks( self.y_profile )
		#print(str(peaks))
		#find valleys by inverting the data
		self.y_valleys, _ = sig.find_peaks(-self.y_profile)		
		
		self.set_y_profile_max()
	
	def set_x_profile_max(self):
		if(self.x_profile is not None) and (self.x_peaks is not None):
			self.x_max_val = np.max(self.x_profile[self.x_peaks])
			self.x_max_index = self.x_peaks[0]
			
			for index in self.x_peaks:
				if self.x_profile[index] == self.x_max_val:
					self.x_max_index = index 
			#print( "MAX X = " + str( self.x_max_val ) + ",  MAX X INDEX =" + str(self.x_max_index))
	
	def set_y_profile_max(self):
		if(self.y_profile is not None) and (self.y_peaks is not None):
			self.y_max_val = np.max(self.y_profile[self.y_peaks])
			self.y_max_index = self.y_peaks[0]
			
			for index in self.y_peaks:
				if self.y_profile[index] == self.y_max_val:
					self.y_max_index = index 
			#print( "MAX Y = " + str( self.y_max_val ) + ",  MAX Y INDEX =" + str(self.y_max_index))
	
	def get_x_profile(self):
		return self.x_profile
		
	def get_y_profile(self):
		return self.y_profile
	
	def save_2d_to_image(self , path=None):
		p="./"
		if path is not None:
			p=path
		
		name="image_"+str(self.uid)
		im = Image.fromarray(self.img_arr_2d , mode='L')
		print("path = '" + p+name+".jpg")
		im.save(p+name+".jpg" )
		
	def xplot(self):
		
		
		plt.cla()
		plt.plot(self.xdata, self.x_profile)
		plt.plot(self.x_peaks, peaks, self.x_profile[self.x_peaks], 'o')
		plt.plot(self.x_valleys, self.x_profile[self.x_valleys], 'o')
		plt.savefig("./data/xplot_"+str(self.uid)+".png")
		#plt.show()

	
	def yplot(self):
		
		
		plt.cla()
		plt.plot(self.ydata, self.y_profile)
		plt.plot(self.y_peaks, self.y_profile[self.y_peaks], 'o')
		plt.plot(self.y_valleys, self.y_profile[self.y_valleys], 'o')
		plt.savefig("./data/yplot_"+str(self.uid)+".png")
	

"""
class to store a collection of ADImages
""" 
class ADImageCollection():
	
	counter=0
	
	def __init__(self):
		
		self.ad_collection = list()
		
	def add(self, ad_image):
		self.ad_collection.append( ad_image )
		#ad_image.print_image_params()
		self.counter=self.counter+1
		#if(self.counter<5):
			#ad_image.save_2d_to_image("./images/") 
		#print("______________________________________________")
	
	def clear(self):
		self.length = len(self.ad_collection)
		self.ad_collection.clear()
		#print( "Before clear = "+str(length) + " , ADImageCOllection cleared length now = " + str(len(self.ad_collection)) )
	
	"""
	
	calculate an average of the 2d arrays 
	
	"""
	def average(self):
		#need to ensure that each 2d array has the same size then average them
		arr = list()
		num_non_zero = 0
		
		#make an array of 2d arrays
		for i,img in enumerate( self.ad_collection ):
			if(img.array_data is not None) and (np.any(img.array_data)):
				arr.append(img.array_data)
				num_non_zero = num_non_zero + 1
				#print( "adding into array ... " + str( img.array_data ) )
		
		#changing the data type here is important, as it will limit the to 8 bit int values
		avg_arr = np.array( arr[0] , dtype=float)
		n = len(arr)
		
		#print("n = " + str(n))
		#print("num_non_zero = " + str(num_non_zero))
		
		
		for i , arr_1d in enumerate(arr):
			if i > 0: 
				# ~ print("i in loop = " + str(i) )
				# ~ print("max in the current = " + str(np.max( arr[i] ))) 
				# ~ print("max in the avg = " + str(np.max( avg_arr ))) 
				tmp = np.array( avg_arr )
				avg_arr = arr[i] + tmp 
				#avg_arr =  sum_arr + avg_arr
				
				#print( "array 1d "+ str( arr_1d ) )
				#print( "avg array "+ str( np.max( avg_arr ) ) )
		
		#self.avg_2d = ( avg_arr / n )
		self.avg_1d = avg_arr/num_non_zero
		self.x_size = 0
		self.y_size = 0
		
		#print("MAX in avg = " + str( np.max(avg_arr) ))
		#print("MIN in avg = " + str( np.min(avg_arr) ))
		#print("MAX in avg_1d = " + str( np.max(self.avg_1d) ))
		#print("MIN in avg_1d = " + str( np.min(self.avg_1d) ))
		
		#for x in avg_arr:
			#print(x)
		
		if(len( self.ad_collection) > 0):
			self.x_size = self.ad_collection[0].x_size
			self.y_size = self.ad_collection[0].y_size
		
			#print("avg arr" + str( avg_arr ) )
			#print("avg 1d array" + str( self.avg_1d ) )
		
			#put 2d profile in an adimage
			self.ad_avg_img = ADImage(self.x_size , self.y_size , self.avg_1d  )
			#self.ad_avg_img.print_image_params() 
			return self.ad_avg_img 
		
		
	
		
		
	
