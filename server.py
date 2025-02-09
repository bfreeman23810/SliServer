#!/home/brian/venv/bin/python3
#!/cs/dvlhome/apps/s/sliServer/dvl/src/venv/bin/python3


import sys
import time
import socket
import argparse
import configparser
import asyncio
import threading as t
import time
import os
import epics as e
from epics import ca
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

import SliData as sli
import ad_image as ad

config_file = "./server.cfg"
lock = t.Lock()

"""
-Server will listen on a port, and wait for external triggers
-Setup will be waiting for EPICS events and can also be forced using server messages
"""

class ServerState(Enum):
	INIT="INIT"
	START="START"
	CONNECT="CONNECT"
	LISTEN="LISTEN"
	RESPOND="RESPOND"
	EVENT="EVENT"
	CHECK = "CHECK"
	GET="GET"
	FIT="FIT"
	FIT_CHECK="FIT_CHECK"
	PUT="PUT"
	DISCONNECT="DISCONNECT"

class SliServer():
	
	array_counter=0
	
	def __init__(self, config, host=None, port=None ):
		
		#ConfigParser object
		self.config = config
		
		self.sli_data = sli.SliData()
		
		self.heartbeat = 0
		self.fit_now = False
		self.running = False
		
		self.ad_image_collection = ad.ADImageCollection()
		
		self.threads = list()
		
		#check to see if host and port are parameters, if not set them to a default
		if host is not None:
			self.host = host
		else:
			self.host = "127.0.0.1"
		
		if port is not None:
			self.port = port
		else:
			self.host = "12345"
		
	
		self.state = ServerState.INIT
        
		#define know transitions in a dictionary
		self.transitions = {
		
			ServerState.INIT: ServerState.START,
			ServerState.START: ServerState.CONNECT,
			ServerState.CONNECT: ServerState.LISTEN,
			ServerState.LISTEN: ServerState.RESPOND,
			ServerState.RESPOND:ServerState.LISTEN,
			ServerState.EVENT:ServerState.CHECK,
			ServerState.CHECK:ServerState.GET,
			ServerState.GET:ServerState.FIT,
			ServerState.FIT:ServerState.FIT_CHECK,
			ServerState.FIT_CHECK:ServerState.PUT,
			ServerState.PUT:ServerState.LISTEN,
			ServerState.DISCONNECT:ServerState.DISCONNECT
			
		}
			
		self.messages = {
		
			ServerState.INIT:"Server is initializing ... Please Wait",
			ServerState.CONNECT:"Now connecting to EPICS channels ... ",
			ServerState.START:"Starting Server ... ",
			ServerState.LISTEN:"Server is waiting ...",
			ServerState.RESPOND:"Server is responding ...",
			ServerState.EVENT:"Trigger Event detected ...",
			ServerState.CHECK:"Server checking EPICS channels still connected ...",
			ServerState.GET:"Server is gettign data from EPICS ... ",
			ServerState.FIT:"Server is using data to fit ... ",
			ServerState.FIT_CHECK:"Server is checking the quality of the fit, and validating ... ",
			ServerState.PUT:"Server is sending data to EPICS ...",
			ServerState.DISCONNECT:"Server is disconnecting ... "
		
				}
				
		self.message=self.messages[self.state]
		
		
	# Callback function to handle changes in the PV value
	def uid_callback(self, pvname, value, **kwargs):
		print(f"{pvname} = {value}")
		#self.get()
		
		#release the lock, which triggers an update in the main loop
		try:
			if self.running == True:
				lock.release()
			else:
				self.ad_pvs["id"].remove_callbacks()
		except Exception as e:
			print( f"something is wrong ... {value}")
			print(e)
		#print('PV Changed! ', pvname, char_value, time.ctime())
		
	
	def disconnect_monitor(self, pvname, value, **kwargs):
		print(f"{pvname} = {value}")
		
		if value == 1:
			print(f"Disconnect Signal from EPICS, disconnecting.....")
			lock.release()
			self.disconnect()
			
			
	
	def transition(self):
		"""Transition to a new state."""
		state = self.state
		self.state = self.transitions[self.state]
		self.message = self.messages[self.state]
		print(f"Transitioning to {self.state} from {state}")
		print(f"{self.message}")
	
	def action(self):
	
		if self.state == ServerState.INIT:
			self.init()
			
		elif self.state == ServerState.START:
			#start this in a new thread
			print("start")
			self.transition()
			self.server_thread = threading.Thread( target=self.start )
			#self.server_thread.start()
			
		elif self.state == ServerState.CONNECT:
			self.connect(self.config)
			self.heartbeat_thread = threading.Thread(target=self.heartbeat_counter)
			self.heartbeat_thread.start()
			
		elif self.state == ServerState.CHECK:
			self.check()
	
		elif self.state == ServerState.DISCONNECT:
			self.disconnect()
	
		elif self.state == ServerState.GET:
			self.get()
	
		elif self.state == ServerState.FIT:
			self.fit()
	
		elif self.state == ServerState.FIT_CHECK:
			self.fit_check()
	
		elif self.state == ServerState.PUT:
			self.put()
	
		elif self.state == ServerState.LISTEN:
			self.listen()
	
		elif self.state == ServerState.RESPOND:
			self.respond()
	
		elif self.state == ServerState.EVENT:
			self.event()
		else:
			print(f"No action for server state = {self.state}")
	
	
	def start(self):
		print(f"Server is in {self.state} state - in start()")
		
		"""Starts a server that communicates with clients and waits for new connections."""
		with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
			# Bind the server to the specified address and port
			server_socket.bind((self.host, int(self.port)))
			server_socket.listen(1)  # Listen for up to 5 connections
			print(f"Server is running on {self.host}:{self.port} and waiting for connections...")
			
			#before staring server loop, transition to next state
			self.transition()
			
			while True:
				try:
					# Accept a new client connection
					client_socket, client_address = server_socket.accept()
					print(f"Connection established with {client_address}")

					# Handle communication with the client
					with client_socket:
						client_socket.sendall(b"Welcome to the server!\n")
						client_socket.sendall(self.messages[self.state].encode('utf-8') )
						while True:
							# Receive data from the client
							data = client_socket.recv(1024)
							#this could be where to parse out states
							data_string = data.decode().strip()
							
							if not data:
								break  # Exit the loop if the client disconnects
							print(f"Received from {client_address}: {data_string}")
							
							#echo server state
							client_socket.sendall(self.messages[self.state].encode('utf-8') + b"\n")
							
							#wanted a method to force the server state
							if data_string == 'transition':
								self.transition()
								self.action()
								client_socket.sendall( b"Server is transitioning to the next state\n"  )
							elif data_string == "event":
								self.state = ServerState.EVENT
								self.action()
							elif data_string == "disconnect":
								self.state = ServerState.EVENT
								self.action()
							
							
							client_socket.sendall( self.messages[self.state].encode('utf-8') +b"\n"  )
							
							
							
							# Echo the data back to the client
							client_socket.sendall(data)
					
					client_socket.close()
					print(f"Connection closed with {client_address}")
				except KeyboardInterrupt:
					print("\nServer is shutting down...")
					self.server_socket.close()
					client_socket.close()

					break						
		
		
	
	def init(self):
		print(f"Server is in {self.state} state - in init()")
		
		ca.initialize_libca()
		
		#attempt to set ca_max_array_bytes
		num_bytes = 10000000000000000000000000000000
		ca.max_array_bytes = num_bytes
		os.environ['EPICS_CA_MAX_ARRAY_BYTES'] = str(num_bytes)
		
		try:
			addr_list = os.environ['EPICS_CA_ADDR_LIST']
			os.environ['EPICS_CA_ADDR_LIST'] = f"{addr_list} 127.0.0.1"
		except KeyError:
			os.environ['EPICS_CA_ADDR_LIST'] = "127.0.0.1"
			print(f"EPICS_CA_ADDR_LIST = ")
		
		#for testing I will use my local host to run a soft ioc, so need to update EPICS_CA_ADDR_LIST
		#addr_list = os.environ['EPICS_CA_ADDR_LIST']
		#os.environ['EPICS_CA_ADDR_LIST'] = f"{addr_list} 127.0.0.1"
		
		#print(os.environ['EPICS_CA_ADDR_LIST'])
		
		self.running = True

		
		self.transition()
	
	
	def heartbeat_counter(self):
		while self.state is not ServerState.DISCONNECT and self.running == True:
			time.sleep(1)
			self.heartbeat = self.heartbeat + 1
			try:
				self.output_pvs["heartbeat"].put( int( self.heartbeat ) )
			except Exception as e:
				print("Error in heartbeat update....")
				print(e)
				self.state = ServerState.DISCONNECT
	
	def check(self):
		print(f"Server is in {self.state} state - in check()")
		
		for key, value in self.ad_pvs.items():	
			
			if value.connected:
				print(f"{key}: ")
				print(f"{value.get()}")
			else:
				print( f"{key} is disconnected" )
				
		
		for key, value in self.input_pvs.items():
			if value.connected:
				print(f"{key}: {value.get()}")
			else:
				print( f"{key} is disconnected" )
		
		for key, value in self.output_pvs.items():
			if value.connected:
				print(f"{key}: {value.get()}")
			else:
				print( f"{key} is disconnected" )
		
		for key, value in self.defaults.items():
			print(f"{key}: {value}")
		
		self.transition()
	
	def get(self):
		print(f"Server is in {self.state} state - in get()")
		
		#use image profile to set the SliData object profile array 
		#probably average these base on unique id events
		 
		
		#check twiss values from EPICS, if they are non-zero use them
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
		self.array_counter = self.array_counter+1
		
		
		self.ad_image_collection.add(self.ad_image)
		self.array = self.ad_image_collection.average().get_x_profile()
		
		self.fit_now = False
		
		if self.array_counter > self.n_frames:
			self.array_counter=0
			self.ad_image_collection.clear()
			print(f"++++++++++++++++CLEARED+++++++++++++")
			self.fit_now = True
			self.set_array_to_fit( self.array )
			self.transition()
			
		
		#print(f"etax = { self.etax} , betax = {self.betax} , emitx = {self.emitx} , sigx = {self.sigx} , dist_slits={self.dist_slit}")
		
		
	def set_array_to_fit(self, array):
		self.array_to_fit = array	
		
	def get_fit_array(self):
		return self.array_to_fit
		
	def fit(self):
		print(f"Server is in {self.state} state - in fit()")
		
		self.sli_data.set_array( self.get_fit_array()  , self.dist_slit )
		self.sli_data.fit()
		self.sli_data.set_beam_parameters(self.emitx , self.betax , self.etax , self.sigx)  
		
		self.transition()
	
		
		
		
	def fit_check(self):
		print(f"Server is in {self.state} state - - in fit_check()")
		
		self.transition()
	
	def put(self):
		print(f"Server is in {self.state} state - in put()")
		
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
		
		#self.sli_data.plot()
		
		self.transition()
		
	def listen(self):
		print(f"Server is in {self.state} state - in listen()")
		self.transition()
		
	def respond(self):
		print(f"Server is in {self.state} state - in resond()")
		self.transition()
		
	def event(self):
		print(f"Server is in {self.state} state - in event()")
		self.state = ServerState.EVENT
		self.transition()
		
	#take in the config object and then parse the epics channels
	def connect(self , config):
		global e
		self.ad_pvs = dict()
		self.input_pvs = dict()
		self.output_pvs = dict()
		self.defaults = dict()
		
		self.all_pvs = dict()
		
		try: 
			
			
			#ad.epics.pvs - area detector pvs
			#self.pvs['test'] = e.PV( "SIM:cam1:ArraySizeX_RBV" )
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

			e.ca.pend_io(timeout=2.0)
			time.sleep(2)
			if self.ad_pvs["id"].connected:
				
				self.ad_pvs["id"].add_callback( self.uid_callback ) 
				#print(f"{self.ad_pvs['id'].info}" )
				
			if self.input_pvs["disconnect"].connected:
				self.input_pvs["disconnect"].put(0) #init this value to zero
				self.input_pvs["disconnect"].add_callback( self.disconnect_monitor ) 
		
		except Exception as e:
			print(f"Something wrong with connecting PVs ... Server is diconnecting:")
			print(e)
			self.state=ServerState.DISCONNECT
			
		
		self.all_pvs = self.ad_pvs | self.output_pvs | self.input_pvs
				
		#if we get here then transition to the next state
		self.transition()
	
	
	def disconnect(self):
		self.state = ServerState.DISCONNECT
		print(f"disconnecting ..... state = {self.state}")
		#perhaps will need to do epics CA clean up
		#self.ad_pvs["id"].disconnect()
		
		try:
			self.running=False
			
			for key,value in self.all_pvs.items():
				value.disconnect()
			
			self.ad_pvs["id"].clear_callbacks()
			self.input_pvs["disconnect"].clear_callbacks()
		
			#for thread in self.threads:
				#thread.kill()
		except Exception as e:
			print(f"Problem disconnecting ... {e}")
			sys.exit(1)
		
		
		print("System is disconnecting without error .... ")
		sys.exit(0)
	
	def fit_action(self):
		if self.running == True:
			self.fit()
			self.put()
		
	def process(self):	
		
		#initilize and the connect the PVs from the defined config file
		self.init()
		self.connect(self.config)
		self.check()
		
		#start a thread to update a heartbeat pv. 
		self.heartbeat_thread = t.Thread(target=self.heartbeat_counter)
		self.heartbeat_thread.start()
		self.threads.append( self.heartbeat_thread )
		
		print('Now wait for changes')
		expire_time = time.time() + 10.
		#while time.time() < expire_time:
		
		while self.running == True:
			try:
				lock.acquire()
				
				
				self.event()
				
				try:
					
					self.get()
				except Exception as e:
					print(f"issue with getting ... \n{e}")
				
				
				if self.fit_now:
					#
					try:
						#self.fit_action()
						update_thread = t.Thread(target=self.fit_action)
						update_thread.start()
						self.threads.append( self.heartbeat_thread )
					except Exception as e:
						print(f"probelm fitting ... \n{e}" )
				
				time.sleep(0.05)
			
			except KeyboardInterrupt:
				print(f"Error in event loop ... disconnecting .....")
				self.disconnect()
			
		self.disconnect()
				
		#for x in range(12):
			#time.sleep(1)
			#self.transition()
			#self.action()
			##asyncio.sleep(1)
			#time.sleep(5)
			
			# ~ if(x==5):
				# ~ self.state = ServerState.EVENT
				# ~ print(f"{self.message}")
				
			# ~ print(f"x={x}")
		
		# ~ self.state = ServerState.DISCONNECT
		

def main():
	# Create an ArgumentParser object
	parser = argparse.ArgumentParser(description="SLI Server ")
	
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
	#config = configparser.RawConfigParser( allow_unnamed_section=True )
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
	
	#connect to host and ports
	server = SliServer(config, host , port)
	
	server.process()
	
	# ~ server.init()
	# ~ server.connect(config)
	# ~ server.check()
	
	# ~ print('Now wait for changes')
	# ~ expire_time = time.time() + 5.
	# ~ while time.time() < expire_time:
		# ~ lock.acquire()
		# ~ server.
		# ~ time.sleep(0.05)
	
	# ~ print('Done.')

if __name__ == "__main__":
    try:
        #asyncio.run( main() )
        main()
    except KeyboardInterrupt:
        print("\nServer shut down.")
