#!/home/brian/venv/bin/python3
#!/cs/dvlhome/apps/s/sliServer/dvl/src/venv/bin/python3

##
# @file SliData.py
# @section sli_data_description Description:
# This file will hold the model information for a diffraction pattern
# and provide functions for fitting
# 
# Much of this was content was parsed from papers and the old SLI C code, written by 
# Pavel Chestov.  
#
# Will use the lmfit python package to do least squares fitting
# 


import numpy as np
import scipy as sp
import math
from lmfit import Model, Minimizer, Parameters, report_fit
from scipy.optimize import curve_fit
import pandas as pd
import ADImage as ad
import scipy.signal as sig #useful for peak finding 
import matplotlib.pyplot as plt
import tifffile
import configparser
import scipy.stats as stats
import os



def calc_noise( arr ):
	"""! Calcualte the mean and the noise in the arr 
	@return mean , and the standard deviation of the array
	"""
	return np.mean(arr), np.sqrt(np.var( arr, ddof=1) )


def gaussian(x, amp, mu, sigma ):
	"""! simple gaussian model function
	@param x - x value point
	@param amp - amplitude
	@param sigma - sigma
	"""
	return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))


def center_of_mass(arr):
  """!Calculates the center of mass of a 1D array.
  @param arr  The 1D array.
  @return The center of mass of the array.
  """

  return np.sum(np.arange(len(arr)) * arr) / np.sum(arr)



def get_gauss_fit_guess( arr ):   
	"""!Take in a 1d array, and attempt to fit a gaussian to it
	@param array - 1d array
	@return - the parameters with errors
	"""
	
	
	#center of mass is our centroid guess
	mu = center_of_mass(arr)
	
	#amp, estimate is the max value
	amp = np.max(arr)
	
	#estimate sigma
	#get mean and variance of data
	mean,noise = calc_noise(arr)
	
	
	l_index =0
	r_index = len(arr)-1
	# use the mean of the data to get the index on the left and right that is just less than the mean
	for index,x in enumerate(arr):
		#print( "index = " + str(index) + " , x="+str(x) )
		if x>=mean:
			l_index = index
			break
			
	# use the mean of the data to get the index on the right that is just less than the mean
	for index in range( len(arr)-1 , -1 , -1 ):
		#print( "index = " + str(index) + " , x="+str(arr[index]) )
		if arr[index] >= mean:
			r_index = index-1
			#print( "value = " + str(arr[index]) + " , mean = " + str(mean)   )
			break
			
	#guess initial sigma
	sigma = r_index - l_index
	
	#create an array that is bounded some number of sigma
	bounding_width = 2*sigma
	l_bound = int( (mu - (bounding_width/2)) )
	r_bound = int( (mu + (bounding_width/2)) )
	bounded_arr = arr[ l_bound:r_bound ]
	
	# ~ print("left = " + str(l_bound))
	# ~ print("right = " + str(r_bound))
	# ~ print("bound width = " + str(bounding_width))
	# ~ print("arr = " + str(bounded_arr))
	
	#need a new mu, if using bounded array
	bounded_mu = center_of_mass(bounded_arr)
	
	# ~ print("mean = " + str( np.mean(arr) ) )
	# ~ print("mean = " + str( mean ) )
	# ~ print("noise = " + str(noise) )
	
	# ~ print( "left index = " + str(l_index)  )
	# ~ print( "right index = " + str(r_index)  )
	
	# ~ print("center of mass = " + str(mu) )
	# ~ print("bounded center of mass = " + str(bounded_mu) )
	# ~ print( "sigma = " + str(sigma)  )
	# ~ print( "amp = " + str(amp)  )
	
	params = Parameters()
	params.add( "amp" , amp , min=None,max=None )
	params.add( "mu" , mu , min=None,max=None )
	params.add( "sigma" , sigma , min=None,max=None )
	
	#return the initial guess of the three fit parameters amplitude, centroid, and width
	return params, l_bound , r_bound
	

def fit_gauss(arr):
	"""!Fit a gaussian to the data
	"""
	#get params, and left and right bound
	params , l , r = get_gauss_fit_guess(arr)
	mod = Model(gaussian)
	xdata = np.arange( len(arr[l:r]) )
	ydata=arr[l : r]
	result = mod.fit( ydata , params , x= xdata )
	res = ydata - result.residual
	


def diff_model_min( params , x , data ):
	"""!Simple diffraction model function, that uses minimization
	@param params an lmfit.Parameter object that contains the parameters
	@param x x value
	@param data minimized array	
	@return minimized value
	"""
	#parse parameters from object
	parvals = params.valuesdict()
	IN = parvals['IN']
	I0 = parvals['I0']
	A = parvals['A']
	B = parvals['B']
	V = parvals['V']
	D = parvals['D']
	F = parvals['F']
	#G = parvals['G']
	
	#model function to use
	model = IN + (I0*(np.sinc( A * (x-B)/math.pi )**2))*(1 + V * np.cos(D*(x-B)-F ) )
	#trying one more parameter, not used
	#model = IN + (I0*(np.sinc( A * (x-B)/math.pi )**2))*(1 + V * np.cos(D*(x-B)-F )) + G
	return model - data


def diff_model( x , IN , I0 , A , B , D, V , F):
	"""!Simple diffraction model without minimization
	"""
	return ( IN + (I0*(np.sinc( A * (x-B)/math.pi )**2))*(1 + V * np.cos(D*(x-B)-F ) ) )


class SliData():
	"""!SliData class, will hold functions initial guesses of parameters
	and other methods to manipulate the diffraction data
	"""
	
	#using microns
	dist_slits = 6000 #slit seperation [m]
	R=9021000 # distnace from SR and source [m] 
	lam=0.450 #lambda is the bandpass filter wavelength [m] 
	
	#first pass defaults , take in from configfile
	etax=-3701650 #desgin dispersion x this will come from CED [m]
	betax=20038000 #desgin beta x this will come from CED [m]
	emitx=0000000.000178 #10^-10 meters design emittance x from CED [m]
	sigx=77.38
	
	"""
	Model Function = 
	Yi = IN + I0 * (sinc(A*(Xi-B)/pi)**2) * ( 1 + V * cos(D*(Xi-B)-F) )
	"""
	#fit parameters
	##visibility parameter 
	V = 0 
	##depends on slit size
	A = 0 
	##shift of interferogram and image axis, depends on slit size
	B = 0 
	##background intensity
	IN = 0 
	##background intensity 
	I0 = 0
	##Phi or phase difference of the light reaching the slits, depends on size 
	F = 0 
	##depends on slit dist
	D = dist_slits 
	
	#variable to show that if the fit was bad or good
	relibility = False
	
	def __init__(self, profile_array=None, dist=None, expected_es=None  ):
		"""!Init the SliData Object
		@param profile_array optional array to start
		@param optional dist is the slit distance
		@param optional expected_es energy spread
		"""
			
		#if these were input then set the fields
		if dist is not None:
			self.dist_slits = dist
		
		if(profile_array is not None):
			self.set_array(profile_array)
		

	
	def set_array(self, profile_array, dist_slits=None):
		"""!Set the profile array
		@param profile array
		@param optionally set the slit distnace parameter
		"""
		
		#if dist_slits is included then set it
		if dist_slits is not None:
			self.dist_slits = dist_slits
		
		#set the data type 
		dt = self.get_data_type(profile_array)
		print(dt)
		
		#set the profile array for this model
		self.profile_array = np.array( profile_array , dtype=dt  )
		
			
	
	def set_beam_parameters(self , emitx , betax , etax , sigx,  expected_es=None ):
		"""!Set all the beam parameters
		@param emitx - the emittance twiss parameter
		@param betax - the beta function twiss parameter
		@param etax - the dispersion twiss parmeter
		@sigx sigx - the sigma of the beam
		@param expected_es - optionally set the expected energy spread
		"""
		
		#set the parameters to these new values
		self.emitx = emitx
		self.betax = betax
		self.etax = etax
		self.sigx = sigx
		
		#calculate the size due to the emittance
		self.sigma_emit = self.calc_sigma_beta_x( self.emitx , self.betax  )
		
		#from parameters
		self.sigma_beam = self.calc_sigma_beam( self.V , self.dist_slits , self.R , self.lam )
		
		#size from design dispersion
		self.sigma_disp_design = self.calc_sigma_dispersion( self.sigx , self.sigma_emit  )
		#beam size due to dispersion
		self.sigma_disp = self.calc_sigma_dispersion( self.sigma_beam , self.sigma_emit  )
		
		#now calculate the energy spread
		self.espread = self.calc_espread(self.sigma_disp , self.etax)
		self.espread_design = self.calc_espread(self.sigma_disp_design , self.etax)
		
		#****TODO**** 
		#- need to still estimate error
		self.espread_err = 0;
		
		#expected e_spread
		if expected_es is not None:
			self.expected_es = expected_es
		else:
			self.expected_es = self.espread_design
		
		#if the expected is included, calculate the error from the expected
		self.error_is_espread = abs(  self.expected_es - self.espread )	
			
		# ~ print(" ---------------------------------------")
		# ~ print( "optimized V = " + str(self.V)   +" and optimized sigma beam = "+str(self.sigma_beam) )
		# ~ print("sigma disp => " + str( self.sigma_disp ))
		# ~ print("sigma emit => " + str( self.sigma_emit ))
		# ~ print("espread design  => " + str( self.espread_design ))
		# ~ print("espread  => " + str( self.espread ))
		# ~ print("err from expected  => " + str( self.error_is_espread  ))	
		
		return self.espread
	
	def fit(self):
		"""!do the fit"""
		
		#set all the initial parameters
		self.params = self.set_initial_model_params(self.profile_array)
		
		#fit the model
		self.result , self.chi2 , self.relibility = self.fit_diff_model( self.params , self.profile_array , self.xdata  )		
		
		#~ if self.relibility == False:
			#~ print("Fit was BAD!")
		#~ else:
			#~ print("FIT was  GOOD")
		
		#report_fit(self.result)
	
		#print("Params" + str(self.result.params.pretty_print()) )
		self.final = self.result.residual + self.profile_array	
		
		#self.set_beam_parameters()
		
	def set_xdata(self, profile_arr):
		"""!Set the x data"""
		return np.arange(len(profile_arr))		
		
	def get_data_type(self, profile_array):
		"""!figure out a data type based on the max value. Used for the numpy array
		@return the data type
		"""
		# guess data type based on max values
		max_val = np.max( profile_array )
		print("Max value => " + str(max_val) )
		
		dt = np.int8
		
		if(max_val <= 2**8 ): 
			dt = np.int8
		elif(max_val <= 2**16 ): 
			dt = np.short
		elif(max_val <= 2**32 ): 
			dt = np.float32
		elif(max_val <= 2**64 ): 
			dt = np.float64
		
		return dt
		
	
	def set_initial_model_params(self , profile_arr , paramaters=None):
		"""!This function will take an intelligent guess at initial parameters
		This will likley need tweaking along the way
		@param profile_arr the profile array of the image
		@param parameters optional argument is lmfit.Parameter object
		@return the lmfit.Parameter object with guesses and bounds
		"""
		
		#set the profile array for this model, ensure that the numpy array data type is set for 
		#max possible value
		self.profile_array = np.array( profile_arr , dtype= self.get_data_type(profile_arr) )
		
		#set x data array
		self.xdata = self.set_xdata(self.profile_array)
		
		#set peaks and valleys
		self.peaks, self.max_val , self.max_index = self.set_peaks(self.profile_array)
		self.valleys ,self.v_left_val,self.v_left_index ,self.v_right_val,self.v_right_index = self.set_valleys(self.profile_array)
		
		#set Imax to the max val
		self.Imax = self.max_val
		
		#set Imin to the average of the first two minumums around the center of mass
		self.Imin =  (self.v_left_val + self.v_right_val)/2
		#set the visibility
		self.V  = self.calc_visibility( self.Imax , self.Imin )
		#set the background level
		self.IN = self.set_background(profile_arr , self.max_index)
		#set the intensity
		self.I0 = self.set_sr_intensity(  self.Imax , self.IN , self.V  )
		#set A
		self.A = self.set_A( self.dist_slits , self.v_left_index , self.v_right_index )
		#B is set to the center of mass
		self.B = center_of_mass(self.profile_array)
		#set D based on the slit distance
		self.D = (self.A * self.dist_slits)*2 
		
		#~ print("Imin estimate = " + str(self.Imin) )
		#~ print( "V Estimate = " + str(self.V) )
		#~ print("Background Number is = " + str( self.IN ))
		#~ print("SR intenstity estimate , I0 Number is = " + str( self.I0 ))
		#~ print( "A = " + str(self.A) )
		#~ print("B = " + str(self.B))
		#~ print( "D = " + str( self.D ) )
		
		#F is the phase offset, may try to add a clever guess eventually
		self.F = 0 # just set this to zero at first
		#G is the offset of the signal, may or may not need this param
		self.G = 0 # just set this to zero at first
		
		
		#create parameters, this is an lmfit.Parameter object with initial bounds
		params = Parameters()
		params.add( 'IN' , self.IN , min=None , max=self.I0)
		params.add( 'I0' , self.I0 , min =None , max=self.I0*10)
		params.add( 'A' , self.A , min=None , max=None)
		params.add( 'B' , self.B , min=None , max=None)
		params.add( 'D' , self.D , min=None , max=None )
		params.add( 'F' , self.F , min=-math.pi , max=math.pi )
		#params.add( 'G' , self.G , min=None , max=None)
		params.add( 'V' , self.V , min=0.1 , max=0.98)
		
		return params
	
	def get_signifigance_level(self, alpha=0.05 , param , param_err , dof ):
		"""!Uses the scipy.stats library to find the signafigance level of the parameter
		@param alpha is the alpha to use, default and standard is 0.05
		@param param is the parmeter that we need to get the pvalue
		@param param_err is te parameter error
		@param dof is the degrees of freedom 
		@return fit_good boolean to indicate if fit was gooe
		@return t_stat the t statistic for the parameter fit
		@return p_value is the p value for fit
		 """
		if param is not None and param_err is not None:
			t_stat = param / param_err
			p_value = 2 * stats.t.sf(abs(t_stat), df=dof )
		else:
			t_stat = 0
			p_value = 1
			
		#alpha = 0.05 #standard signifigance level
		
		if p_value > alpha:
			fit_good = False
		else:
			fit_good = True  
		
		#print("T Stat = " + str(t_stat) )
		#print("P Val = " + str(p_value) )
		
		#return boolean fit_good, t_stat, and p_value
		return fit_good , t_stat , p_value
	
	def set_params_to_optimized(self, params):
		"""! This function will be responsible for setting the class parameters to 
		the optimized parameters. Also, will calculate the p_value and t_stat 
		for the relivant parameters
		@param params the lmfit.Parameter objects
		
		"""
		
		self.params_opt = params
		
		self.V = params.get('V').value
		self.F = params.get('F').value
		self.A = params.get('A').value
		self.B = params.get('B').value
		self.D = params.get('D').value
		self.IN = params.get('IN').value
		self.I0 = params.get('I0').value
		#self.G = params.get('G').value
		
		self.V_err = params.get('V').stderr
		self.F_err = params.get('F').stderr
		self.D_err = params.get('D').stderr
		self.A_err = params.get('A').stderr
		self.B_err = params.get('B').stderr
		self.IN_err = params.get('IN').stderr
		self.I0_err = params.get('I0').stderr
		#self.G = params.get('G').stderr
		
		#using standard alpha value of 0.05 for parameter fit goodness
		self.relibility , self.V_t_stat , self.V_p_value = self.get_signifigance_level( 0.05 , self.V , self.V_err , (len( self.xdata ) - len(params)) )
		
		
		return( self.relibility , self.V_t_stat , self.V_p_value )
		
	
	def set_peaks( self , profile_arr ):
		"""!Find the peaks in the ydata
		@param profile_arr is the profile array of the image
		@return peaks the array of peaks found
		@return max_val the max value in the peaks array
		@return max_index the index in the array where to find the max_val
		"""
		#find peaks in ydata
		peaks, _ = sig.find_peaks( profile_arr, prominence=0.3, width=3 )
		#max_value in peaks
		max_val = np.max(self.profile_array[peaks])
		
		#index of the max, start at zero 
		max_index = peaks[0]
		
		#iterate through and then set the max_index
		#save the index off the max pixel	
		for index in peaks:
			if profile_arr[index] == max_val:
				max_index = index 
		
		#return the peaks array , the max value, and the max index
		return peaks, max_val , max_index
		
	
	def set_valleys( self , profile_arr ):
		"""!Find the valleys in the ydata
		@param profile_arr is the profile array of the image
		@return valleys the array of valleys found
		v_left_val , v_left_index , v_right_val, v_right_index 
		@return  v_left_val is the value of the first valley left of the peak
		@return  v_lef_index is the index of the first valley left of the peak
		@return  v_right_val is the value of the first valley right of the peak
		@return  v_right_index is the index of the first valley right of the peak
		
		"""
		#find valleys by inverting the data
		valleys, _ =sig.find_peaks(-profile_arr, prominence=0.3, width=3)	
		
		#get the first indexes that are closest to the index of the max on the left and right
		v_left_index = 0
		v_right_index = 0
		tmp_index=0
		
		# get the integer value of the center of mass
		cm = center_of_mass( profile_arr )
		
		
		for index in valleys:
			
			if index < cm:
				tmp_index = index
				
			v_left_index=tmp_index
		
		tmp_index=0		
		for index in valleys:
			if index > cm:
				v_right_index = index
				break	
			
				
		#the first min on the left and the right of the max		
		v_left_val = self.profile_array[ v_left_index ]
		v_right_val = self.profile_array[ v_right_index ]
		
		return valleys , v_left_val , v_left_index , v_right_val, v_right_index 			
		

		
			
		
	def fit_max_min_array_to_gaussian(self, profile_arr, max_array, min_array):
		"""!Fit both the max and min or peaks and valleys arrays to a gaussian
		This is not used yet, but may be useful to guess fit bounds better
		or calculate the Imax and Imin values
		@param profile_arr profile array of the image 
		@param max_array array of peaks in data
		@param min_array array of valleys in data
		@return the results arrays
		"""
		
		#rely on that this an array of indexes returned by scipy.signal.find_peaks()
		xdata_max = max_array
		xdata_min = min_array
		
		#y data for max and min
		ydata_max = profile_arr[ max_array ]
		ydata_min = profile_arr[ min_array ]
		
		# Create a model from the function
		gmodel_max = Model(gaussian)
		gmodel_min = Model(gaussian)

		#need initial guesses for params
		#best guesses, amplitude
		amp_max = np.max( ydata_max )
		amp_min = np.max( ydata_min )
		
		#best guess centroid, will be the x_value at max		
		for index in xdata_max:
			if profile_arr[index] == amp_max:
				cen_max = index 
		
		# ~ #assume that the centroid of the min, is half way between the middle indexes
		# ~ if self.v_left_index is not None and self.v_right_index is not None:		
			# ~ cen_min =  ( (self.v_right_index - self.v_right_index) /2 ) 
			
		#use the max centroid for the min
		cen_min = cen_max
		
		#guess that the sigma will be (right_most_index-left_most_index)/3
		sig_max = (xdata_max[-1]-xdata_max[0])/3
		sig_min = (xdata_min[-1]-xdata_min[0])/3
		
		# Set initial parameter values
		params_max = gmodel_max.make_params(amp=amp_max, cen=cen_max, sigma=sig_max)
		params_min = gmodel_min.make_params(amp=amp_min, cen=cen_min, sigma=sig_min)
		

		# Fit the model to the data
		result_max = gmodel_max.fit(ydata_max, params_max, x=xdata_max, c=0)
		result_min = gmodel_min.fit(ydata_min, params_min, x=xdata_min, c=0)

		# Print the fit report
		print(result_max.fit_report())
		print(result_min.fit_report())
		
		self.gauss_max_fit = ydata_max + result_max.residual
		self.gauss_min_fit = ydata_min + result_min.residual
		
		#reset Visibility based on the 2 amplitudes
		#self.Imax = result_max.params['amp'].value
		#self.Imin = result_min.params['amp'].value
		
		#self.V = self.calc_visibility( self.Imax , self.Imin )
		
		#self.beam_size = self.calc_sigma_beam( self.V , self.dist_slits , self.R , self.lam )
		
		#print( "Imax and Imin = "  + str(self.Imax) + " " + str(self.Imin) )
		
		#print( "New V = " + str( self.V ) + " , new Beam Size =  " + str(self.beam_size ) ) 
		#print( "After Gauss Fit; Imax = " + str( result_max.params['amp'] ) + ", Imin = " + str(result_min.params['amp']) )
		
		
		return ( result_max , result_min )		
		
			
	def calc_visibility(self , Imax , Imin):
		"""!Estimate the visibility parameter
		@param Imax the value of the peak of the difraction pattern model
		@param Imin the value of the amplitude of the minimum of the diffraction patterns
		@retrun v the visibility estimate
		"""
	
		#estimate the visibility 	
		v = (Imax - Imin) / (Imax + Imin)
		
		
		return v
	
	"""
	beam size = [(lambda * R) / (pi* dist_slits)]*sqrt( 0.5*ln(1/V) ) 
	"""	
	def calc_sigma_beam( self , V , dist_slits , R , lam ):
		"""! Use the parameters to calculate the beam size
		@param V is the visibility parameter
		@param dist_slits distance of the slits
		@param R the distance from the the detector to the SL source.
		@param lam is the wavelength of light, which is set by the bandpass filter 
		@return sigma_beam the total beam size
		"""
		
		return ( ( lam * R) / ( math.pi * dist_slits) ) * np.sqrt( 0.5 * np.log( 1 / V ) ) 
		
	def set_background(self , profile_arr , split_val ):
		"""!Set the background level. This may need some tweaking
		doing stuff here that is not used yet, but may be useful for improving this
		@param profile_arr is the profile array of the image
		@param split_val is where to spilt the array
		"""
			
		#split the profile array into 2 parts at the max
		arr1,arr2 = np.split(profile_arr , [split_val])
		
		#pick the min values on each side of the max
		minL = np.min(arr1)
		minR = np.min(arr2)
		
		#print( "minL = " + str(minL) )
		#print( "minR = " + str(minR) )
		
		#choose the larger of the 2, and this is the background estimate
		if minL > minR:
			IN = minL
		else:
			IN = minR
		
		#background less than zero or equal to zero than set to small number
		if IN <= 0:
			IN = 0.01
		
		mean , noise = calc_noise( profile_arr )
		
		#print("IN = " + str(IN) )
		#print("noise = " + str(noise) )
		
		return noise
	
	"""
	set the estimate of IN, or just set the value and return
	
	"""	
	def set_sr_intensity(self , max_val , noise , V  ):
		"""!Set the SR intensity or I0 
		(max value - background) / (1 + Visibility)
		@param max_val is the max value in the profile array
		@param noise is the noise floor of the profile array
		@param V is the visibility parameter
		"""
		
		I0 = ( ( max_val - noise ) / ( 1 + V) )
		
		#return the absoluty value of the calculated value
		return abs( I0 )	
	
	def set_A(self , dist , left_index , right_index ):
		"""!Estimate or set the value of the coeficent A, this is based on 
		the estimated width of the highest peak
		(2*pi)/(( 2*dist_slits-1 )* width_tallest_peak)
		@param dist distance of the slits
		@param left_index is the index of the first left valley
		@param right_index is the index of the first rigth valley
		@return the estimated value of A, or zero
		"""
		try:
			#error trapping because divide by zero is possible
			A = ( 2*math.pi ) /( ((2 *  dist )-1)*( right_index - left_index ) )
		except:
			print( "ERROR occured in calculating A, return 0")
			A=0
		
		return A	
	
	def set_B( self, B=None  ):	
		"""!Set the value of B parameter, this is just the max_value of the tallest peak
		@param B optionally just set and return the value of B
		@return the value of B
		"""
		if B is not None:
			self.B = B
			return B;
		
		if self.max_index is not None:
			self.B = self.max_index
		else:
			self.B = 0
		return self.B
		
	def calc_sigma_beta_x(self , emit=None , betax=None ):
		"""!calculate the size due to beta and emittance
		@param emit optionally take in and set the emittance twiss parameter
		@param betax optionally take in the beta twiss parameter value
		@return sigma_betax return the beam size due to dispersion
		"""
		if emit is not None:
			self.emit = emit
		if betax is not None:
			self.betax = betax
			 
		sigma_beta = np.sqrt(  self.emitx * self.betax )
		#print("Sigma Beta Size = " + str(sigma_beta))
		return sigma_beta
	
	
	def calc_sigma_dispersion(self , sigma_beam , sigma_emit ):
		"""!calculate the size due to dispersion
		@param sigma_beam the total beam size
		@param sigma_emit claculated size due to beta and the emittance
		@return sigma_dispersion beam size due to dispersion
		"""
		sigma_disp = np.sqrt( abs( sigma_beam**2 - sigma_emit**2 ) )
		#print("Sigma From Dispersion = " + str(sigma_disp) )
		return sigma_disp
	
	def calc_espread(self, sigma_disp , disp):
		"""!Calculate the energy spread
		@param sigma_disp is the beam size due to dispersion
		@param disp is the dispersion or eta twiss parameter
		@return espread the energy spread
		"""
		espread= abs( sigma_disp/disp )
		#print("ESpread = " + str(espread))
		return abs( espread )	
	
	def back_calc_visibility(self , slit_dist, lam , R , sigma):
		"""!Back calculate the visibility based on other parameters. 
		@param slit_dist is the distance of the slits
		@param lam is the wavelength from the bandpass filter
		@param R is the distance from camera sensor to the SR that is imaged
		@param sigma is the total beam size
		@return visibility 
		"""
		visibility = 1 / np.exp( (2 * math.pi**2 * slit_dist**2 * sigma**2) / (lam**2 * R**2  ) )
		#print("Back Calculated Visibilty = " + str(visibility))
		return visibility
		
	
	def back_calc_Imin_from_V(self , Imax , V):
		"""!Back calculate the Imin from the visibility parameter
		@param Imax the max value of the tallest peak or the top of the diffration fit
		@param V is the visibility parameter
		@return Imin
		"""
		Imin = Imax*(1-V)/(1+V)
		#print( "Backcalculated Imin = " + str(Imin) )
		return Imin 
	
	def calc_chi_sq(self , profile_arr , result):
		"""!Calculate the reduced chi squared. This should go into the relibility value
		This needs some work
		@param profile_arr is the profile array of the image
		@param is the result array from the fit
		"""
		 mi = lmfit.minimize(residual, p)
		 dof = len(x) - ndim
		 chi2 = sum(residual(p) ** 2) / dof
		
	
	def fit_diff_model(self , params , profile_arr , xdata  ):
		"""!Use this function to fit the diffraction model using a minimization model
		@param params is the lmfit.Parameter object used for the fit
		@param profile_arr the profile of the image array
		@param xdata is the x value array
		@return the result array
		@return the chi squared 
		@return the new parameter object from optimized parameter 
		"""
		mi = Minimizer( diff_model_min , params, fcn_args=(xdata, profile_arr) )
		
		#get the MinimizerResult object
		result = mi.minimize()
		#set fields to optimized params
				
		#calculate 
		self.dof = len(profile_arr) - len(result.params)
		self.chi2 = sum( result.residual  ** 2) / self.dof
		
		#get chisq 
		chi2_res = result.chisqr
		#get from results
		rchi2_res = result.redchi
		
		r, t, p = self.set_params_to_optimized(result.params)
		
		#~ print( "DOF = " + str(self.dof) )
		#~ print( "chi2 = " + str(self.chi2) )
		#~ print( "chi2 res = " + str(chi2_res) )
		
		#print( "err = " + str( result.residual ) )
		return result , self.chi2 , r
	
	#getters
	def get_espread(self):
		"""!Get the energy spread and assume *10^-5
		"""
		return (self.espread*10**(5))
	
	def get_sigma_total(self):
		"""!get sigma beam"""
		return self.sigma_beam
	
	def get_sigma_twiss(self):
		"""!get sigma emittance"""
		return self.sigma_emit
		
	def get_sigma_disp(self):
		"""!get sigma dispersion"""
		return self.sigma_disp
	
	def get_espread_err(self):
		"""!get energy spread"""
		return self.espread_err
	
	def get_reliability(self):
		"""!get relibility"""
		return self.relibility
	
	def get_visibility(self):
		"""!get visibility"""
		return self.V
	
	def get_visibility_err(self):
		"""!get visibility error"""
		return self.V_err
	
	def get_xdata(self):
		"""!get x values"""
		return self.xdata
		
	def get_profile_array(self):
		"""!get the profile array"""
		return self.profile_array

	def get_result_array(self):
		"""!get the result array"""
		return self.final
	
	def plot(self , save_path=None , title=None , xlabel=None , ylabel=None):
		
		plt.cla()
		#plot label
		#l='fit: V=%5.3f +/- %5.3f' % (self.V, self.V_err)
		
		
		plt.xlabel("Index")
		plt.ylabel("Intensity [Ph/s/.1%bw/mm^2] ")
		
		l="Fit Line:"
		if self.V is not None and self.V_err is not None:
			l+='\nVisibility=%1.2f +/- %1.2e' % (self.V , self.V_err)
		l+='\nEnergy Spread=%1.2e' % (self.espread)
		l+='\nchi2=%1.2e' % (self.chi2)
		
		if self.relibility is False:
			r="BAD"
		else:
			r="GOOD"
			
		l+='\nFIT was ' + r
		
		plt.plot(self.xdata, self.profile_array, label="data" )
		#plt.plot( self.xdata , self.y_est ) 
		#plt.plot( self.xdata , self.result.init , '--' ) 
		#plt.plot( self.xdata[self.left_bound:self.right_bound] , self.final , '--' ) 
		plt.plot( self.xdata , self.final , '--' , label=l ) 
	
		plt.plot(self.peaks, self.profile_array[self.peaks], 'o')
		#plt.plot(self.peaks, self.gauss_max_fit, '-')
		
		plt.plot(self.valleys, self.profile_array[self.valleys], 'o')
		#plt.plot(self.valleys, self.gauss_min_fit, '-')
		
		#plt.plot( [self.xval], [self.n_imin], 'X')
		
		#plt.plot(self.max_index, self.max_val, 'x')
		#plt.plot(self.v_left_index, self.v_left_val, 'x')
		#plt.plot(self.v_right_index, self.v_right_val, 'x')
		#plt.plot([self.xdata[0], self.xdata[-1]], [self.IN, self.IN] ,'--')
		#plt.savefig("./data/xplot_"+str(self.uid)+".png")
		
		
		plt.legend(loc='best')
		
		if title is not None:
			plt.title(title)
		
		if save_path is not None:
			plt.savefig(save_path)
		else:
			plt.show()



def process_sankalp_sims_pass1():
	
	path="./data/sankalp/Pass1/"
	dir_list = os.listdir(path)
	print(dir_list)
	
	for f in dir_list:
		#get info from title
		if f.endswith(".txt"):
			
			df = pd.read_csv( path+"/"+f , delimiter='\t' , header=None )
			
			f = f.strip(".txt")
			sp=f.split(',')
			print(sp)
			
			if len(sp)==4:
				ss=sp[3]
			else:
				ss=''
			slit_dist=int(sp[0])*1000
			slit_width = float(sp[1])
			espread=float(sp[2])
			
			img = ad.ADImage(df.shape[1] , df.shape[0], df.to_numpy().T, 'sankalp_sim_sd-'+str(slit_dist/1000) + '_sw-'+str(slit_width)+'_es-'+str(espread)+'_ss-'+ss)
			sli = SliData( img.x_profile , slit_dist , espread )
			sli.fit()
			sli.set_beam_parameters( sli.emitx , sli.betax , sli.etax , sli.sigx )  
			sli.plot( "./data/sankalp/plots/" + img.uid + ".png" , " Slit Distance=%1.0dmm , Slit Width=%1.1fmm, ES=%1.1e"%(slit_dist/1000,slit_width, espread) )

def process_sankalp_sims_pass2():
	

	#second pass defaults
	etax=-3682094 #desgin dispersion x this will come from CED [m]
	betax=1900104 #desgin beta x this will come from CED [m]
	emitx=0000000.0001337 #10^-10 meters design emittance x from CED [m]
	sigx=114.52
	
	path="./data/sankalp/Pass2/"
	dir_list = os.listdir(path)
	print(dir_list)
	
	for f in dir_list:
		#get info from title
		if f.endswith(".txt"):
			
			df = pd.read_csv( path+"/"+f , delimiter=',' , header=None )
			
			f = f.strip(".txt")
			sp=f.split(',')
			print(sp)
			
			if len(sp)==4:
				ss=sp[3]
			else:
				ss=''
			slit_dist=int(sp[0])*1000
			slit_width = float(sp[1])
			espread=float(sp[2])
			
			img = ad.ADImage(df.shape[1] , df.shape[0], df.to_numpy().T, 'sankalp_sim_sd-'+str(slit_dist/1000) + '_sw-'+str(slit_width)+'_es-'+str(espread)+'_ss-'+ss)
			sli = SliData( img.x_profile , slit_dist , espread )
			sli.fit()
			sli.set_beam_parameters( emitx , betax , etax , sigx )  
			sli.plot( "./data/sankalp/plots2/" + img.uid + ".png" , " Slit Distance=%1.0dmm , Slit Width=%1.1fmm, ES=%1.1e"%(slit_dist/1000,slit_width, espread) )


def test():
	
	
	# Specify the file path
	#file_path = './data/2d_joes_sim.txt'
	#file_path = './data/slice.csv'
	file_path_sim = './data/SIM.txt'
	#file_path = './data/sankalp/7,0.5,3.0e-5.txt'
	
	file_path = './data/sankalp/7,0.6,1.3e-5.txt'
	
	# ~ tif = './data/SRWU16.tif'
	# ~ im = tifffile.imread(tif)
	# ~ im_arr = np.array(im)

	# Read the text file into a DataFrame
	df_sim = pd.read_csv(file_path_sim, delimiter='\t', header=None)
	df = pd.read_csv(file_path, delimiter='\t', header=None)
	#df = pd.read_csv(file_path,  header=None)
	#print( df.head() )

	#data is 32 bit?
	#img = ad.ADImage(df.shape[1] , df.shape[0], df.to_numpy().T, 2)
	img_sim = ad.ADImage(df_sim.shape[1] , df_sim.shape[0], df_sim.to_numpy(), 'sim')
	img = ad.ADImage(df.shape[1] , df.shape[0], df.to_numpy().T, 'sankalp_7_0.5_3e-5')
	#img = ad.ADImage(im_arr.shape[1] , im_arr.shape[0], im_arr, "SRWU16")
	#img.print_image_params()
	#img.save_2d_to_image("./data/")
	
	#gauss_fit(img_sim.x_profile)
	#get_gauss_fit_guess(img_sim.x_profile)
	#fit_gauss(img.x_profile)
	
	#fit_gauss(img_sim.x_profile)
	
	#sli_data_sim = SliData( img_sim.x_profile )
	sli_data = SliData( img.x_profile )
	fit_gauss(sli_data.peaks)
	
	#sli_data_sim.plot("./data/plots/sim.png" , "Gaussian Simulated" , "Intensity [pixels]")
	#sli_data.plot()
	
	#img.xplot()
	#img.yplot()

	
	
#process_sankalp_sims_pass2()
#test()


				
	
	
