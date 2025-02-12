#!/home/brian/venv/bin/python3
import socket


#client needs to get the host and port from the same config file the server does

def start_client(host='127.0.0.1', port=12345):
    """Connects to a server and exchanges messages."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        try:
            # Connect to the server
            client_socket.connect((host, port))
            print(f"Connected to server at {host}:{port}")

            # Receive the welcome message from the server
            welcome_message = client_socket.recv(1024).decode()
            print(f"Server: {welcome_message.strip()}")

            # Send messages to the server
            while True:
                message = input("You: ")
                if message.lower() in ['exit', 'quit']:
                    print("Closing connection...")
                    break

                # Send the message
                client_socket.sendall(message.encode())

                # Receive the server's response
                response = client_socket.recv(1024).decode()
                print(f"Server: {response.strip()}")

        except ConnectionRefusedError:
            print(f"Could not connect to the server at {host}:{port}")
        except KeyboardInterrupt:
            print("\nClient shutting down...")
            client_socket.close()

if __name__ == "__main__":
    start_client()
