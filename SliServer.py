#!/home/brian/venv/bin/python3
#/cs/dvlhome/apps/s/sliServer/dvl/src/venv/bin/python3


##
# @mainpage SliServer Project
#
# @section description Description:
# This project will be used to take in data from an EPICS Waveform and then use fitting
# algorithms to fit the expected diffraction pattern. Then will send that data
# back to EPICS. 
# 
# This server will also listen on a confirgured host and port, for communication
# to the server. This default host and port will be included in the server.cfg file.
#

##
# @file SliServer.py
# @brief This file will be the main entry point to the server. 
# @author Brian Freeman
# @par Revision History:
# -Februray ??, 2025 Initial Release

#standard imports
import sys
import time
import socket
import argparse
import configparser
import asyncio
import threading as t
import time
import os
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

#import epics 
import epics as e
from epics import ca

#local imports
import SliData as sli
import ADImage as ad

## default config file
config_file = "./server.cfg"
## public lock object
lock = t.Lock()


class ServerState(Enum):
	"""! A class to use to keep track of server states"""
	INIT="INIT"
	START="START"
	CONNECT="CONNECT"
	CHECK = "CHECK"
	GET="GET"
	FIT="FIT"
	FIT_CHECK="FIT_CHECK"
	PUT="PUT"
	DISCONNECT="DISCONNECT"

class SliServer():
	"""! Main Server class, that will provide the server functions. Server will have a thread
	that will wait for client connections and then respond with a status. Listening thread
	will also be able to signal the server to stop, from the client. There will also
	be a thread that will continuesly get and process data from EPICS, then will update 
	EPICS variables. This server will use two EPICS callback functions to use 
	
	"""
	
	
	
	def __init__(self, config, host=None, port=None ):
		"""!Initilize the SliServer object
		@param config - config file
		@param host - host to use, if not using from config file
		@param port - port to use, if not using from config file
		"""
		
		## ConfigParser object
		self.config = config
		
		## sli data object. data holder for model and fitting algorithms
		self.sli_data = sli.SliData()
		
		## server heartbeat. Will use a seperate thread to update this heartbeat
		self.heartbeat = 0
		## boolean value to fit the fit_array
		self.fit_now = False
		## use this boolean as a global signal to run server
		self.running = False
		## number of arrays collected, will use an EPICS variable to set the number of frames until averageing and fitting occurs
		self.array_counter = 0
		## a collection of ADImage objects, and provides convience functions to average
		self.ad_image_collection = ad.ADImageCollection()
		
		
		# check to see if host and port are parameters, if not set them to a default
		if host is not None:
			self.host = host
		else:
			self.host = config.get("server" , "host")
		
		if port is not None:
			self.port = port
		else:
			self.host = config.get("server" , "port")
		
		#set the server state to INIT
		self.state = ServerState.INIT
        
		## server message dictionary, will return a string based on state	
		self.messages = {
		
			ServerState.INIT:"Server is initializing ... Please Wait",
			ServerState.CONNECT:"Now connecting to EPICS channels ... ",
			ServerState.START:"Starting Server ... ",
			ServerState.CHECK:"Server checking EPICS channels still connected ...",
			ServerState.GET:"Server is getting data from EPICS ... Running and OK",
			ServerState.FIT:"Server is using data to fit ... ",
			ServerState.FIT_CHECK:"Server is checking the quality of the fit, and validating ... ",
			ServerState.PUT:"Server is sending data to EPICS ...",
			ServerState.DISCONNECT:"Server is disconnecting ... "
		
				}
						
		
	# Callback function to handle changes in the PV value
	def uid_callback(self, pvname, value, **kwargs):
		"""! EPICS AreaDetector PV ${P}:cam1:UniqueId_RBV callback  
			On update it will release the lock object and in turn get the values from EPICS
			This PV is updated everytime there is a new image array to process
		"""
		#release the lock, which triggers an update in the main loop
		try:
			#if running then release the lock
			if self.running == True:
				if lock.locked(): 
					lock.release()
			#else just remove this as a callback and disconnect		
			else:
				self.ad_pvs["id"].clear_callbacks()
				disconnect()
		except Exception as e:
			print( f"something is wrong {pvname} = {value}")
			print(e)
		#print('PV Changed! ', pvname, char_value, time.ctime())
		
	
	def disconnect_monitor(self, pvname, value, **kwargs):
		"""! EPICS PV ${P}:server:disconnect signal callback, whcih just stops the server"""
		
		#if the value is anything but zero, then disconnect
		if value != 0:
			print(f"Disconnect Signal from EPICS, disconnecting.....{pvname} = {value}")
			self.disconnect()
		
			#if lock is locked then release it, this will ensure that thread is not waiting
			if lock.locked():
				lock.release()
			
	def get_message(self):
		"""!get the message from server state"""
		return self.messages[self.state]
		
	
	
	def start(self):
		"""! Start the server, and just wait for client to ask status
			Also will provide a method to remotely shut down the server
		 """
		self.state = ServerState.START
		
		"""Starts a server that communicates with clients and waits for new connections."""
		with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
			# Bind the server to the specified address and port
			server_socket.bind((self.host, int(self.port)))
			server_socket.listen(1)  # Listen for up to 5 connections
			print(f"Server is running on {self.host}:{self.port} and waiting for connections...")
			
			
			while self.running:
				try:
					# Accept a new client connection
					client_socket, client_address = server_socket.accept()
					print(f"Connection established with {client_address}")

					# Handle communication with the client
					with client_socket:
						client_socket.sendall(b"Welcome to the server!\n")
						client_socket.sendall(self.get_message().encode('utf-8') )
						while self.running:
							# Receive data from the client
							data = client_socket.recv(1024)
							data_string = data.decode().strip()
							
							if not data:
								break  # Exit the loop if the client disconnects
							print(f"Received from {client_address}: {data_string}")
							
							#echo server state
							client_socket.sendall(self.messages[self.state].encode('utf-8') + b"\n")
							
							#wanted a method to remotely check the status and send disconnect signal
							if data_string == 'status':
								if self.running == True:
									client_socket.sendall( b"Running \n"  )
								client_socket.sendall(self.get_message().encode('utf-8') )
							elif data_string == "disconnect":
								
								self.disconnect()
							
							time.sleep(0.2)
							
							# Echo the data back to the client
							#client_socket.sendall(data)
							
							#sleep for a bit
							time.sleep(0.2)
					
					if lock.locked(): 
						lock.release()
					
					client_socket.close()
					print(f"Connection closed with {client_address}")
					
				except KeyboardInterrupt:
					print("\nServer is shutting down...")
					self.server_socket.close()
					client_socket.close()
					self.disconnect()
					if lock.locked(): 
						lock.release()
	
					break						
		
		
	
	def init(self):
		"""! function will do some initial tasks before starting server threads
		
		"""
	
		#initilize the channel access library 
		ca.initialize_libca()
		
		#attempt to set ca_max_array_bytes, this is likely not best done here
		num_bytes = 10000000000000000
		ca.max_array_bytes = num_bytes
		os.environ['EPICS_CA_MAX_ARRAY_BYTES'] = str(num_bytes)
	
		#set main running boolean to True 
		self.running = True
	
	
	def heartbeat_counter(self):
		"""!This function will be started in a new daemon thread and be responsible for 
		incrementing the heartbeat counter and  then updating the EPICS PV  
		
		"""
		while self.running == True:
			#sleep for 1 second
			time.sleep(1)
			#increment the counter
			self.heartbeat = self.heartbeat + 1
			
			try:
				#attempt to set the EPICS PV
				self.output_pvs["heartbeat"].put( int( self.heartbeat ) )
			except Exception as e:
				print("Error in heartbeat update....")
				print(e)
				#if an error was trapped, then disconnect the server
				self.disconnect()
	
		
				
	
	def check(self):
		"""!
			check that PVs are connected. Will attempt to reconnect. Some PVs should disconnect the server
			if they are not connected
		"""
		self.state = ServerState.CHECK
		
		for key,value in self.all_pvs.items():
			if value.connected:
				#print for debug
				#print(f"{value.pvname} is {value.get()}")
				continue
			else:
				print(f"PV {value.pvname} seems disconnectd")
				
			
	def get(self):
		"""! do a get on connected channels and store them in variables
			If the number of arrays matches the nframes PV from EPICS, then average 
			and then fit the data
		"""
		self.state = ServerState.GET
		
		#check twiss values from EPICS, using the scaling factors to match CED order of magnitude
		self.etax = float( self.input_pvs["etax"].get() )*(10**6) #number in meters, convert to microns
		self.betax = float( self.input_pvs["betax"].get() )*(10**6) #number in meters, convert to microns
		self.sigx = float( self.input_pvs["sigx"].get() )*(10**6) #number in meters, convert to microns
		self.emitx = float( self.input_pvs["emitx"].get() )*(10**-6) #number in meters, multiply by 10e-10  
		self.dist_slit = float( self.input_pvs["dist_slit"].get() )*(10**3) #number in millimeters
		self.n_frames = int( self.input_pvs["n_frames"].get() ) 
		
		
		#if number is 0 in EPICS, use the default values
		if self.etax == 0:
			self.etax = float(self.defaults["etax"])
		if self.emitx == 0:
			self.emitx = float( self.defaults["emitx"] )
		if self.sigx == 0:
			self.sigx = float( self.defaults["sigx"] )
		if self.betax == 0:
			self.betax = float( self.defaults["betax"] )
		if self.dist_slit == 0:
			self.dist_slit = float(self.defaults["dist_slits"] )
		if self.n_frames == 0:
			self.n_frames = float(self.defaults["nframes"] )
		
		
		#init an ad_image object
		self.ad_image = ad.ADImage(  self.ad_pvs["x_size"].get() , self.ad_pvs["y_size"].get() , self.ad_pvs["array"].get() )
		#increment array counter
		self.array_counter = self.array_counter+1
		
		#add the image array to the collectcion
		self.ad_image_collection.add(self.ad_image)
		
		
		#set fit boolean to false
		self.fit_now = False
		
		#if the number of arrays is greater than n_frames
		if self.array_counter > self.n_frames:
			#reset the array counter
			self.array_counter=0
			
			#set the fit boolean to true
			self.fit_now = True
			
			#set the array to the average of the arrays
			self.array = self.ad_image_collection.average().get_x_profile()
			#set the fit array to the averaged array
			self.set_array_to_fit( self.array )
			
			#clear the image array from ADImageCollection object
			self.ad_image_collection.clear()
			print(f"++++++++++++++++CLEARED+++++++++++++")
			#print(f"etax = { self.etax} , betax = {self.betax} , emitx = {self.emitx} , sigx = {self.sigx} , dist_slits={self.dist_slit}")
		
		
	def set_array_to_fit(self, array):
		"""! Set the fit array
		@param array - array to be fit
		 """
		self.array_to_fit = array	
	
	def get_fit_array(self):
		"""!fit array getter
		@return array_to_fit
		"""
		return self.array_to_fit
		
	def fit(self):
		"""! Fit the array that was set in the get() function
		"""
		self.state = ServerState.FIT
		
		#set the array in the SliData object
		self.sli_data.set_array( self.get_fit_array()  , self.dist_slit )
		#do the fit
		self.sli_data.fit()
		#set the beam parameters, this function sets the beam size and energy spread, etc...
		self.sli_data.set_beam_parameters(self.emitx , self.betax , self.etax , self.sigx)  
			
		
	def fit_check(self):
		"""!place holder to have a function that will check the fit quality and take some
			action if the fit was bad. Thinking that that we narrow the array and then
			re-fit. If fit is bad then, do not do the put.
		"""
		self.state = ServerState.FIT_CHECK
		
		
	
	def put(self):
		"""! This function is responsible for setting the EPICS PVS from the output_pvs 
		dictionary
		"""
		self.state = ServerState.PUT
		#print(f"Server is in {self.state} state - in put()")
		
		#server output pvs
		self.output_pvs["espread"].put( self.sli_data.get_espread() )
		self.output_pvs["sigma_total"].put(self.sli_data.get_sigma_total()  )
		self.output_pvs["sigma_twiss"].put( self.sli_data.get_sigma_twiss() )
		self.output_pvs["sigma_espread"].put(self.sli_data.get_sigma_disp() )
		self.output_pvs["sigma_espread_err"].put(self.sli_data.get_espread_err() )
		self.output_pvs["rel"].put( self.sli_data.get_reliability() )
		self.output_pvs["vis"].put(self.sli_data.get_visibility() )
		
		#waveforms
		self.output_pvs["xdata"].put( self.sli_data.get_xdata() )
		self.output_pvs["proj"].put( self.sli_data.get_profile_array() )
		self.output_pvs["proj_fit"].put( self.sli_data.get_result_array() )
		#~ self.output_pvs["proj_gaussian_max"].put(self.sli_data)
		#~ self.output_pvs["proj_gaussian_min"].put(self.sli_data)
		
		
		
	def connect(self , config):
		"""! Uses the  global config file to try to connect to EPICS channels
		"""
		self.state = ServerState.CONNECT
		
		global e
		## dictionary to store AreaDetector PVS
		self.ad_pvs = dict()
		## dictionary to store input pvs
		self.input_pvs = dict()
		## dictionary to store output pvs
		self.output_pvs = dict()
		## dictionary to store defaults from config file 
		self.defaults = dict()
		
		## dictionary to store all pvs
		self.all_pvs = dict()
		
		#try to connect EPICS PVs
		try: 
			
			#ad.epics.pvs - area detector pvs
			self.ad_pvs["array"] = e.PV( str( config.get("ad.epics.pvs" , "array") ) )
			self.ad_pvs["x_size"] = e.PV( config.get("ad.epics.pvs" , "x_size") )
			self.ad_pvs["y_size"] = e.PV( config.get("ad.epics.pvs" , "y_size") )
			
			self.ad_pvs["id"] = e.PV( str(config.get("ad.epics.pvs" , "id")) )
			#self.ad_pvs["id"].add_callback(self.uid_callback)
			#self.ad_pvs["id"] = e.PV( "SIM:image1:UniqueId_RBV" , callback=self.uid_callback )
			
			self.ad_pvs["type"] = e.PV( config.get("ad.epics.pvs" , "type") )
			self.ad_pvs["rate"] = e.PV( config.get("ad.epics.pvs" , "rate") )
						
			#server input pvs
			self.input_pvs["betax"] = e.PV( config.get("server.input.pvs" , "beta") )
			self.input_pvs["etax"] = e.PV( config.get("server.input.pvs" , "disp") )
			self.input_pvs["sigx"] = e.PV( config.get("server.input.pvs" , "sig") )
			self.input_pvs["emitx"] = e.PV( config.get("server.input.pvs" , "emit") )
			self.input_pvs["dist_slit"] = e.PV( config.get("server.input.pvs" , "dist_slit") )
			self.input_pvs["n_frames"] = e.PV( config.get("server.input.pvs" , "n_frames") )
			self.input_pvs["disconnect"] = e.PV( config.get("server.input.pvs" , "disconnect") )
			
			#server output pvs
			self.output_pvs["heartbeat"] = e.PV( config.get("server.output.pvs" , "heartbeat") )
			self.output_pvs["espread"] = e.PV( config.get("server.output.pvs" , "espread") )
			self.output_pvs["sigma_total"] = e.PV( config.get("server.output.pvs" , "sigma_total") )
			self.output_pvs["sigma_twiss"] = e.PV( config.get("server.output.pvs" , "sigma_twiss") )
			self.output_pvs["sigma_espread"] = e.PV( config.get("server.output.pvs" , "sigma_espread") )
			self.output_pvs["sigma_espread_err"] = e.PV( config.get("server.output.pvs" , "sigma_espread_err") )
			self.output_pvs["rel"] = e.PV( config.get("server.output.pvs" , "rel") )
			self.output_pvs["vis"] = e.PV( config.get("server.output.pvs" , "vis") )
			self.output_pvs["xdata"] = e.PV( config.get("server.output.pvs" , "xdata") )
			self.output_pvs["proj"] = e.PV( config.get("server.output.pvs" , "proj") )
			self.output_pvs["proj_fit"] = e.PV( config.get("server.output.pvs" , "proj_fit") )
			self.output_pvs["proj_gaussian_max"] = e.PV( config.get("server.output.pvs" , "proj_gaussian_max") )
			self.output_pvs["proj_gaussian_min"] = e.PV( config.get("server.output.pvs" , "proj_gaussian_min") )
			
			#not epics, but these are 1st pass defaults
			self.defaults["dist_slits"] = config.get("sli.defaults" , "dist_slits") 
			self.defaults["R"] = config.get("sli.defaults" , "R") 
			self.defaults["lam"] = config.get("sli.defaults" , "lam") 
			self.defaults["etax"] = config.get("sli.defaults" , "etax") 
			self.defaults["betax"] = config.get("sli.defaults" , "betax") 
			self.defaults["emitx"] = config.get("sli.defaults" , "emitx") 
			self.defaults["sigx"] = config.get("sli.defaults" , "sigx") 
			self.defaults["nframes"] = config.get("sli.defaults" , "nframes") 

			#wait for connect
			e.ca.pend_io(timeout=2.0)
			#wait some additional time
			time.sleep(2)
			
			#now check the for connections of the 2 PVs that we are adding callbacks to.
			if self.ad_pvs["id"].connected:
				
				#Add the callback function to Unique_Id PV from AreaDetector
				self.ad_pvs["id"].add_callback( self.uid_callback ) 
				#print(f"{self.ad_pvs['id'].info}" )
				
			if self.input_pvs["disconnect"].connected:
				#This is the server disconnect PV, so I want to force this to zero
				self.input_pvs["disconnect"].put(0) #init this value to zero
				#add the callback, that monitors for change in this value. 
				self.input_pvs["disconnect"].add_callback( self.disconnect_monitor ) 
		
		#if an error is trapped on conenction, then disconnect the server
		except Exception as e:
			print(f"Something wrong with connecting PVs ... Server is diconnecting:")
			print(e)
			self.state=ServerState.DISCONNECT
		
		#update the dictionary that will hold all the PVs for easy checking of connection	
		self.all_pvs = self.ad_pvs | self.output_pvs | self.input_pvs
				
	
	
	def disconnect(self):
		"""!Disconnect the server"""
		self.state = ServerState.DISCONNECT
		print(f"disconnecting ..... state = {self.state}")
		
		try:
			#set the running flag to False/ 
			self.running=False
			
			#disconnect PVs
			for key,value in self.all_pvs.items():
				value.disconnect()
			
			#clear the callbacks
			self.ad_pvs["id"].clear_callbacks()
			self.input_pvs["disconnect"].clear_callbacks()
		
		#if there are trapped errors, then exit.  
		except Exception as e:
			print(f"Problem disconnecting ... {e}")
			sys.exit(1)
			
	
	def fit_action(self):
		"""!fitting actions"""
		if self.running == True:
			self.fit()
			self.fit_check()
			self.put()
		
	def process(self):	
		"""!Main processing loop, this loop will be the main event loop"""
		#initilize and the connect the PVs from the defined config file
		self.init()
		self.connect(self.config)
		self.check()
		
		#start server thread. 
		self.server_thread = t.Thread(target=self.start)
		#start as daemon thread, so ending main event loop will not wait wait for join
		self.server_thread.daemon = True
		self.server_thread.start()
		
		
		#start a thread to update a heartbeat pv. 
		self.heartbeat_thread = t.Thread(target=self.heartbeat_counter)
		#start as daemon thread, so ending main event loop will not wait wait for join
		self.heartbeat_thread.daemon = True
		self.heartbeat_thread.start()
		
		
		print('Now wait for changes .... ')
		while self.running == True:
			try:
				#lock the thread, until there is an update to the Unique_Id PV
				lock.acquire()
				
				#check the PVS, and then get the values from EPICS				
				try:
					self.check()
					self.get()
				except Exception as e:
					print(f"issue with getting ... \n{e}")
				
				#if the boolean fit now was set, which happens in the get() function, then
				#do the fit action
				if self.fit_now:
					
					try:
						#do fit action in a new thread, so we don't block updates
						update_thread = t.Thread(target=self.fit_action)
						#start as daemon thread, so ending main event loop will not wait for join 
						update_thread.daemon = True
						update_thread.start()
					except Exception as e:
						print(f"probelm fitting ... \n{e}" )
				
				#sleep for a short amount of time, save a little CPU
				time.sleep(0.033)
			
			except KeyboardInterrupt:
				print(f"Error in event loop ... disconnecting .....")
				self.disconnect()
			
		#self.disconnect()
		

def main():
	"""!Main loop
	"""
	# Create an ArgumentParser object
	parser = argparse.ArgumentParser(description="Sli Server ")
	
	# Add arguments
	parser.add_argument('-host', type=str, help='host to start the server', required=False)
	parser.add_argument('-port', type=int, help='port to start the server', required=False)
	parser.add_argument('-c', type=str, help='config file', required=False)
	#parser.add_argument('--greet', action='store_true', help='Flag to greet the user')
	
	# Parse the command-line arguments
	args = parser.parse_args()
	### read config file and parse values
	
	#check to see if config file was set on the command line, if not use the default
	
	# Create an instance of ConfigParser
	config = configparser.ConfigParser()
	# Read from the configuration file
	config.read(config_file)
	
	#if host and port were defined on the command line then use those
	if args.host is not None: 
		host = args.host
	elif config.get('server','host') is not None:
		host = config.get('server','host')
		
	#if port were defined on the command line then use it
	if args.port is not None: 
		port = args.port
	elif config.get('server','port') is not None:
		port = config.get('server','port')	
	
	# check to see if host and port are set
	if host is None or port is None:
		print(f"For some reason host or port is undefined ... check the config file { config_file }" )
	
	print(f"Hello, starting server at host = {host} at port={port}")
	
	#create server object
	server = SliServer(config, host , port)
	
	#start main event loop
	server.process()
	

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nServer shut down.")
