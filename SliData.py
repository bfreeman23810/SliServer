#!/home/brian/venv/bin/python3
#!/cs/dvlhome/apps/s/sliServer/dvl/src/venv/bin/python3


import numpy as np
import scipy as sp
import math
from lmfit import Model, Minimizer, Parameters, report_fit
from scipy.optimize import curve_fit
import pandas as pd
import ad_image as ad
import scipy.signal as sig #useful for peak finding 
import matplotlib.pyplot as plt
import tifffile
import configparser
import scipy.stats as stats
import os


###########################################################################
#### Calculate the noise average and standard deviation of 1st N points.
#### 
###########################################################################
def calc_noise(start_pt, YS, N):
	start = start_pt
	end = start_pt + N + 1
	if(end >= len(YS) and len(YS) > 2):
		end = len(YS)-1
		start = end-2
	return np.mean(YS[start:end]), np.sqrt(np.var(YS[start:end], ddof=1))


########################################################################
# Calculate the initial estimate for the 'c' parameter offset based on
# the mean and range of the lower of the two noise levels.  Also returns
# the noise data on which these were based.
# The bounds are 2.5 to 97.5 percentiles of the data.
# YG         subset of the global data that was originally partitioned
#            to contain the peak
# fit_start  index of YG where the data used for fitting starts
# fit_end    index of YG where the data used for fitting ends
# npts_noise number of data points used in noise calculations
########################################################################
def calcCEstimateAndBounds(YG, fit_start, fit_end, npts_noise):
    data_start = 0
    data_end = len(YG)

    # The "halfwidth" or threshold "filters" in findplots may trim the sides of
    # the fitted data range.  If so, take the noise to be from outside of the
    # fitted area without exiting the data.
    startl = max(data_start, fit_start - npts_noise)
    endl = startl + npts_noise
    endr = min(data_end, fit_end + npts_noise)
    startr = endr - npts_noise
    noisel = YG[startl:endl]
    noiser = YG[startr:endr]

    # Take your initial c estimate to be the average of the noise on the side
    # with the lower noise mean, and used the trimmed noise to be the upper
    # and lower bounds on the c value
    meanl = np.mean(noisel)
    meanr = np.mean(noiser)

    mean = lb = ub = noise = None
    if (meanr < meanl):
        mean = meanr
        lb = np.percentile(noiser, 2.5)
        ub = np.percentile(noiser, 97.5)
    else:
        mean = meanl
        lb = np.percentile(noisel, 2.5)
        ub = np.percentile(noisel, 97.5)

    # Sometimes the "noise" has very little variability and the bounds don't make
    # any sense.  In this case, variance calculation may have problems, so
    # use a very simple bounding scheme.
    if (lb >= ub or lb >= mean or ub <= mean):
        lb = mean * 0.9
        ub = mean * 1.1

    return mean, lb, ub, noisel, noiser
    

###############################################################################
#### Fitting using lmfit methods: fits 1 Gaussian to data containing 1 peak.
###############################################################################
def fit_LM(x, y, e, p0, s, ndim, c_lb, c_ub):
    global xv, yv, ev
    [xv, yv, ev] = [x, y, e]
    # -----------------------------------------------------------------------
    # Set up the initial guess and the ranges of each parameter of the fit
    # -----------------------------------------------------------------------
    p = lmfit.Parameters()
    p.add_many(('A', p0[0], True, p0[0] / s, p0[0] * s, None),
               ('mu', p0[1], True, p0[1] / s, p0[1] * s, None),
               ('sigma', p0[2], True, p0[2] / s, p0[2] * s, None),
               ('c', p0[3], True, c_lb, c_ub, None))
    # -----------------------------------------------------------------------
    # Call the routine lmfit to minimize residuals. Comptue chi2
    # -----------------------------------------------------------------------
    mi = lmfit.minimize(residual, p)
    dof = len(x) - ndim
    chi2 = sum(residual(p) ** 2) / dof
    ps = np.zeros(10)
    ps[0] = mi.params['A'].value
    ps[1] = mi.params['A'].stderr
    if (math.isnan(ps[1])):
        ps[1] = 0
    ps[2] = mi.params['mu'].value
    ps[3] = mi.params['mu'].stderr
    if (math.isnan(ps[3])):
        ps[3] = 0
    ps[4] = mi.params['sigma'].value
    ps[5] = mi.params['sigma'].stderr
    if (math.isnan(ps[5])):
        ps[5] = 0
    ps[6] = mi.params['c'].value
    ps[7] = mi.params['c'].stderr
    if (math.isnan(ps[7])):
        ps[7] = 0.0
    ps[8] = chi2
    ps[9] = 0.0

    return ps
    

###############################################################################
#### Fitting using gaussian and line: fits 1 Gaussian to data containing 1 peak.
###############################################################################
def fit_Model(travel, y, p0, fit_xvalues, fit_yvalues):
    def gaussian(x, amp, mu, wid):
        """1-d gaussian: gaussian(x, amp, mu, wid)"""
        return (amp) * np.exp(-(x - mu) ** 2 / (2 * wid ** 2))

    def line(x, slope, intercept):
        """a line"""
        return slope * x + intercept

    # -----------------------------------------------------------------------
    # Define the model and fit it
    # -----------------------------------------------------------------------
    mod = Model(gaussian) + Model(line)
    pars = mod.make_params(amp=p0[0], mu=p0[1], wid=p0[2], slope=0, intercept=p0[3])
    result = mod.fit(y, pars, x=travel)

    ps = np.zeros(10)
    rparams = result.params
    ps[0] = rparams['amp'].value
    ps[1] = rparams['amp'].stderr
    ps[2] = rparams['mu'].value
    ps[3] = rparams['mu'].stderr
    ps[4] = rparams['wid'].value
    ps[5] = rparams['wid'].stderr
#    ps[6] = rparams['intercept'].value
    ps[6] = travel[0]*rparams['slope'] + rparams['intercept']
    ps[7] = rparams['intercept'].stderr
    #ps[8] = rparams['chi-square']
    ps[8] = 0.0
    ps[9] = 0.0

    # Populate the fitvalues arrays
    fit_xvalues.append(travel)
    fit_yvalues.append(list(result.best_fit))
    return ps

# Define the function to fit
def gaussian(x, amp, mu, sigma ):
    return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))


def center_of_mass(arr):
  """Calculates the center of mass of a 1D array.

  Args:
    arr: The 1D array.

  Returns:
    The center of mass of the array.
  """

  return np.sum(np.arange(len(arr)) * arr) / np.sum(arr)


"""Take in a 1d array, and attempt to fit a gaussian to it
	Args:
		array: 1d array
	Returns: 
		the parameters with errors
"""
def get_gauss_fit_guess( arr ):   
	
	#test how gauassian is the data
	# ~ res = stats.normaltest(arr)
	# ~ stat = res.statistic
	# ~ p=res.pvalue
	
	# ~ print( "stat = " + str(stat) + " , p-value = " + str(p) )
	# ~ if p > 0.05:
		# ~ print('Data is likely normally distributed (p-value > 0.05)')
	# ~ else:
		# ~ print('Data is likely not normally distributed (p-value <= 0.05)')

	
	#need to first make initial guesses
	
	#center of mass is our centroid guess
	mu = center_of_mass(arr)
	
	#amp, estimate is the max value
	amp = np.max(arr)
	
	#estimate sigma
	#get mean and variance of data
	mean,noise = calc_noise(0,arr,( len(arr)-2) )
	
	
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
	
	sigma = r_index - l_index
	
	#create an array that is bounded some number of sigma
	bounding_width = 2*sigma
	l_bound = int( (mu - (bounding_width/2)) )
	r_bound = int( (mu + (bounding_width/2)) )
	bounded_arr = arr[ l_bound:r_bound ]
	
	print("left = " + str(l_bound))
	print("right = " + str(r_bound))
	print("bound width = " + str(bounding_width))
	print("arr = " + str(bounded_arr))
	
	#need a new mu, if using bounded array
	bounded_mu = center_of_mass(bounded_arr)
	
	print("mean = " + str( np.mean(arr) ) )
	print("mean = " + str( mean ) )
	print("noise = " + str(noise) )
	
	print( "left index = " + str(l_index)  )
	print( "right index = " + str(r_index)  )
	
	print("center of mass = " + str(mu) )
	print("bounded center of mass = " + str(bounded_mu) )
	print( "sigma = " + str(sigma)  )
	print( "amp = " + str(amp)  )
	
	params = Parameters()
	params.add( "amp" , amp , min=None,max=None )
	params.add( "mu" , mu , min=None,max=None )
	params.add( "sigma" , sigma , min=None,max=None )
	
	#return the initial guess of the three fit parameters amplitude, centroid, and width
	return params, l_bound , r_bound
	

def fit_gauss(arr):
	params , l , r = get_gauss_fit_guess(arr)
	mod = Model(gaussian)
	xdata = np.arange( len(arr[l:r]) )
	ydata=arr[l : r]
	result = mod.fit( ydata , params , x= xdata )
	res = ydata - result.residual
	
	report_fit(result)
	
	plt.plot( xdata , ydata  )
	plt.plot( xdata , res )
	plt.show()


#simple model to minimize, modeled after examples from lmfit website:
def diff_model_min( params , x , data ):
	parvals = params.valuesdict()
	IN = parvals['IN']
	I0 = parvals['I0']
	A = parvals['A']
	B = parvals['B']
	V = parvals['V']
	D = parvals['D']
	F = parvals['F']
	#G = parvals['G']
	
	model = IN + (I0*(np.sinc( A * (x-B)/math.pi )**2))*(1 + V * np.cos(D*(x-B)-F ) )
	#model = IN + (I0*(np.sinc( A * (x-B)/math.pi )**2))*(1 + V * np.cos(D*(x-B)-F )) + G
	return model - data


def diff_model( x , IN , I0 , A , B , D, V , F):
	return ( IN + (I0*(np.sinc( A * (x-B)/math.pi )**2))*(1 + V * np.cos(D*(x-B)-F ) ) )

# ---------------------------------------------------------------------------
############################################################
#### Functions to be fit: Gaussian with a constant offset.
#### and diffraction model function 
############################################################
f_gauss = lambda x, amp, cen, sigma, c: amp * np.exp(-((x - cen) ** 2) / (2.0 * (sigma ** 2))) + c
#  
#  name: f_diff
#  @param
#  @return
#  
f_diff = lambda x, IN, I0, A, B, D, V, F , C : IN + (I0*(np.sinc( A * (x-B)/math.pi )**2))*(1 + V * np.cos(D*(x-B)-F ) ) + C


# A container to keep fit parameters and profiles
class SliData():
	
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
	V = 0 #visibility parameter 
	A = 0 #depends on slit size
	B = 0 # shift of interferogram and image axis, depends on slit size
	IN = 0 #background intensity
	I0 = 0 #background intensity of SR source
	F = 0 #Phi or phase difference of the light reaching the slits, depends on size
	D = dist_slits #depends on slit dist
	
	relibility = False
	
	"""
	dist_slits = the distance between the slits. 
	R = The distance from the SL source and the slits
	probably should get the initial parameters from config file, then perhaps EPICS
	"""
	def __init__(self, profile_array=None, dist=None, expected_es=None  ):
		
			
		#if these were input then set the fields
		if dist is not None:
			self.dist_slits = dist
		
		if(profile_array is not None):
			self.set_array(profile_array)
		
		
		# ~ #print("---------------------------------------------------------------\n")

		# ~ #fit the model
		# ~ self.result , self.chi2 , self.relibility = self.fit_diff_model( params , self.profile_array , self.xdata  )		
		
		# ~ if self.relibility == False:
			# ~ print("Fit was BAD!")
		# ~ else:
			# ~ print("FIT was  GOOD")
		
		
		# ~ report_fit(self.result)
		
	
		# ~ print("Params" + str(self.result.params.pretty_print()) )
		# ~ self.final = self.result.residual + self.profile_array
		

		#self.espread = self.set_beam_parameters(self.emitx , self.betax , self.etax , self.sigx)
		
		# ~ print("-----------------Now use gaussian fit------------------------")
		
		#self.r_max , self.r_min = self.fit_max_min_array_to_gaussian( self.profile_array , self.peaks , self.valleys )
		#~ self.sigma_beam = self.calc_sigma_beam( self.V , self.dist_slits , self.R , self.lam )
		#~ self.sigma_disp = self.calc_sigma_dispersion( self.sigma_beam , self.sigma_emit  )
		#~ #self.sigma_disp = self.calc_sigma_dispersion( 88.03 , self.sigma_emit  ) #using joe's example
		#~ self.espread = self.calc_espread(self.sigma_disp , self.etax)
	

	
	def set_array(self, profile_array, dist_slits=None):
		
		if dist_slits is not None:
			self.dist_slits = dist_slits
		
		dt = self.get_data_type(profile_array)
		print(dt)
		
		#set the profile array for this model
		self.profile_array = np.array( profile_array , dtype=dt  )
		
			
	
	def set_beam_parameters(self , emitx , betax , etax , sigx,  expected_es=None ):
		
		self.emitx = emitx
		self.betax = betax
		self.etax = etax
		self.sigx = sigx
		
		self.sigma_emit = self.calc_sigma_beta_x( emitx , betax  )
		
		#from parameters	
		self.sigma_beam = self.calc_sigma_beam( self.V , self.dist_slits , self.R , self.lam )
		#size from design dispersion
		self.sigma_disp_design = self.calc_sigma_dispersion( sigx , self.sigma_emit  )
		self.sigma_disp = self.calc_sigma_dispersion( self.sigma_beam , self.sigma_emit  )
		
		self.espread = self.calc_espread(self.sigma_disp , etax)
		self.espread_design = self.calc_espread(self.sigma_disp_design , etax)
		
		#need to estimate error
		self.espread_err = 0;
		
		#expected e_spread
		if expected_es is not None:
			self.expected_es = expected_es
		else:
			self.expected_es = self.espread_design
		
		self.error_is_espread = abs(  self.expected_es - self.espread )	
			
		print(" ---------------------------------------")
		print( "optimized V = " + str(self.V)   +" and optimized sigma beam = "+str(self.sigma_beam) )
		print("sigma disp => " + str( self.sigma_disp ))
		print("sigma emit => " + str( self.sigma_emit ))
		print("espread design  => " + str( self.espread_design ))
		print("espread  => " + str( self.espread ))
		print("err from expected  => " + str( self.error_is_espread  ))	
		
		return self.espread
	
	def fit(self):
		
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
		return np.arange(len(profile_arr))		
		
	def get_data_type(self, profile_array):
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
		
	#return the parameter object with initial bounds and guesses
	def set_initial_model_params(self , profile_arr , paramaters=None):
		#first take in parameter, and set my values to the values 
		
		#set the profile array for this model
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
		self.V  = self.calc_visibility( self.Imax , self.Imin )
		self.IN = self.set_background(profile_arr , self.max_index)
		self.I0 = self.set_sr_intensity(  self.Imax , self.IN , self.V  )
		self.A = self.set_A( self.dist_slits , self.v_left_index , self.v_right_index )
		self.B = center_of_mass(self.profile_array)
		self.D = (self.A * self.dist_slits)*2 
		
		#~ print("Imin estimate = " + str(self.Imin) )
		#~ print( "V Estimate = " + str(self.V) )
		#~ print("Background Number is = " + str( self.IN ))
		#~ print("SR intenstity estimate , I0 Number is = " + str( self.I0 ))
		#~ print( "A = " + str(self.A) )
		#~ print("B = " + str(self.B))
		#~ print( "D = " + str( self.D ) )
		
		#F is the phase offset
		self.F = 0 # just set this to zero at first
		#G is the offset of the signal
		self.G = 0 # just set this to zero at first
		
		
		#create parameters
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
	
	def get_signifigance_level(self, alpha , param , param_err , dof ):
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
		
		return fit_good , t_stat , p_value
	
	def set_params_to_optimized(self, params):
		
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
		#find peaks in ydata
		peaks, _ = sig.find_peaks( profile_arr, prominence=0.3, width=3 )
		#max_value in peaks
		max_val = np.max(self.profile_array[peaks])
		
		#index of the max, start at zero 
		max_index = peaks[0]
		
		#iterate through and then retun the max_index
		#save the index off the max pixel	
		for index in peaks:
			if profile_arr[index] == max_val:
				max_index = index 
		
		#return the peaks array , the max value, and the max index
		return peaks, max_val , max_index
		
	
	def set_valleys( self , profile_arr ):
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
		#return [ "MaxParams" : result_max.params ]
		
		
			
	def calc_visibility(self , Imax , Imin):
		#estimate of visibility
		# ~ if Imax is not None:
			# ~ self.Imax = Imax
		# ~ if Imin is not None:
			# ~ self.Imin = Imin
		
		#estimate the visibility 	
		v = (Imax - Imin) / (Imax + Imin)
		
		
		return v
	
	"""
	beam size = [(lambda * R) / (pi* dist_slits)]*sqrt( 0.5*ln(1/V) ) 
	"""	
	def calc_sigma_beam( self , V , dist_slits , R , lam ):
		# ~ if dist_slits is not None:
			# ~ self.D = dist_slits
		# ~ if R is not None:
			# ~ self.R = R
		# ~ if lam is not None:
			# ~ self.lam = lam
		# ~ if V is not None:
			# ~ self.V = V
		
		#self.coeff = (( self.lam * self.R  ) / ( math.pi * self.dist_slits ))*np.sqrt(0.5)
		#self.sigma_est = self.coeff * np.sqrt( np.log(1/self.V) )
		
		sigma_est = ( ( lam * R) / ( math.pi * dist_slits) ) * np.sqrt( 0.5 * np.log( 1 / V ) ) 
		
		return sigma_est
		
	#def set_e_spread(self):
		
	
	def set_background(self , profile_arr , split_val ):
			
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
		
		mean , noise = calc_noise( 0,profile_arr,len(profile_arr) )
		
		
		#print("IN = " + str(IN) )
		#print("noise = " + str(noise) )
		
		return noise
	
	"""
	set the estimate of IN, or just set the value and return
	(max value - background) / (1 + Visibility)
	"""	
	def set_sr_intensity(self , max_val , noise , V  ):
		
		
		I0 = ( ( max_val - noise ) / ( 1 + V) )
		
		return abs( I0 )	
	"""
	estimate or set the value of the coeficent A
	(2*pi)/(( 2*dist_slits-1 )* width_first_peak)
	"""
	def set_A(self , dist , left_index , right_index ):
		
		A=0
		try:
			A = ( 2*math.pi ) /( ((2 *  dist )-1)*( right_index - left_index ) )
		except:
			print( "ERROR occured in calculating A, return 0")
		#A = ( 2*math.pi / dist - 0.5 )*( right_index - left_index ) 
		
		return A	
	
	def set_B( self, B=None  ):	
		if B is not None:
			self.B = B
			return B;
		
		if self.max_index is not None:
			self.B = self.max_index
		else:
			self.B = 0
		return self.B
		
	def calc_sigma_beta_x(self , emit=None , betax=None ):
		if emit is not None:
			self.emit = emit
		if betax is not None:
			self.betax = betax
			 
		sigma_beta = np.sqrt(  self.emitx * self.betax )
		#print("Sigma Beta Size = " + str(sigma_beta))
		return sigma_beta
		
	def calc_sigma_dispersion(self , sigma_beam , sigma_emit ):
		sigma_disp = np.sqrt( abs( sigma_beam**2 - sigma_emit**2 ) )
		#print("Sigma From Dispersion = " + str(sigma_disp) )
		return sigma_disp
	
	def calc_espread(self, sigma_disp , disp):
		espread= abs( sigma_disp/disp )
		#print("ESpread = " + str(espread))
		return abs( espread )	
	
	def back_calc_visibility(self , slit_dist, lam , R , sigma):
		visibility = 1 / np.exp( (2 * math.pi**2 * slit_dist**2 * sigma**2) / (lam**2 * R**2  ) )
		#print("Back Calculated Visibilty = " + str(visibility))
		return visibility
		
	
	def back_calc_Imin_from_V(self , Imax , V):
		Imin = Imax*(1-V)/(1+V)
		#print( "Backcalculated Imin = " + str(Imin) )
		return Imin 
	
	def calc_chi_sq(self , profile_arr , result):
		 mi = lmfit.minimize(residual, p)
		 dof = len(x) - ndim
		 chi2 = sum(residual(p) ** 2) / dof
		
	
	def fit_diff_model(self , params , profile_arr , xdata  ):
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
		return (self.espread*10**(5))
	
	def get_sigma_total(self):
		return self.sigma_beam
	
	def get_sigma_twiss(self):
		return self.sigma_emit
		
	def get_sigma_disp(self):
		return self.sigma_disp
	
	def get_espread_err(self):
		return self.espread_err
	
	def get_reliability(self):
		return self.relibility
	
	def get_visibility(self):
		return self.V
	
	def get_visibility_err(self):
		return self.V_err
	
	def get_xdata(self):
		return self.xdata
		
	def get_profile_array(self):
		return self.profile_array

	def get_result_array(self):
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


				
	
	
