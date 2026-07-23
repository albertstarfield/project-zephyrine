import zmq

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("ipc://celestial_timestream_vector_helper.socket")

print("Sending request...")
socket.send(b"Hello DCTD")

message = socket.recv()
print(f"Received reply: {message.decode()}")