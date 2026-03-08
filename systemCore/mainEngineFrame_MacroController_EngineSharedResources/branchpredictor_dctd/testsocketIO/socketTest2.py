import zmq
import time
import json

def test_dctd():
    print("--- DCTD Branch Predictor Client Test ---")
    
    # 1. Create Context & Socket
    context = zmq.Context()
    socket = context.socket(zmq.REQ) # REQ matches the Daemon's REP

    # 2. Connect
    # Use the exact string the daemon printed:
    address = "ipc://celestial_timestream_vector_helper.socket"
    
    print(f"Connecting to {address}...")
    socket.connect(address)

    # 3. Send Request
    # Currently the Ada code hardcodes the input to '42', 
    # so the message content here is just a trigger.
    payload = "Trigger Quantum Prediction"
    print(f"Sending: {payload}")
    socket.send(payload.encode('utf-8'))

    # 4. Receive Reply
    print("Waiting for prediction...")
    reply_bytes = socket.recv()
    reply_str = reply_bytes.decode('utf-8')

    print(f"\n[SUCCESS] Received Raw: {reply_str}")
    
    try:
        data = json.loads(reply_str)
        print(f"Parsed JSON:")
        print(f"  > Next Hash: {data['predicted_lsh_hash']}")
        print(f"  > Horizon:   {data['predicted_time_horizon']}")
    except json.JSONDecodeError:
        print("Error: Could not parse JSON response.")

if __name__ == "__main__":
    test_dctd()