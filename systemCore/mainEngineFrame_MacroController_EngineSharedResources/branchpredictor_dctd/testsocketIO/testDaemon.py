import zmq
import json
import random
import time
import numpy as np

def test_dctd():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("ipc://celestial_timestream_vector_helper.socket")

    # Generate 5 dummy vectors (1024-D)
    # We use numpy just to pretty-print shape, but send list
    vectors_np = np.random.rand(5, 1024)
    vectors_list = vectors_np.tolist()

    print(f"\n[Client] Generated Input:")
    print(f"  > Count: {len(vectors_list)} Vectors")
    print(f"  > Dimension: 1024")
    print(f"  > First Vector (Snippet): {vectors_list[0][:5]}...")

    payload = {
        "vectors": vectors_list,
        "timestamps": [time.time()] * len(vectors_list)
    }
    
    print(f"\n[Client] Sending Request ({len(json.dumps(payload))} bytes)...")
    start_time = time.time()
    
    # Send
    socket.send_string(json.dumps(payload))

    # Receive
    msg = socket.recv_string()
    end_time = time.time()
    
    print(f"[Client] Received Reply in {(end_time - start_time)*1000:.2f}ms")
    
    try:
        data = json.loads(msg)
        print(f"\n[Client] Prediction Result:")
        print(f"  > Predicted Next Hash: {data.get('predicted_lsh_hash')}")
        print(f"  > Time Horizon:        {data.get('predicted_time_horizon')}")
        
        if "debug_input_count" in data:
            print(f"  > Daemon Processed:    {data['debug_input_count']} vectors")
            
    except json.JSONDecodeError:
        print(f"Error Parsing JSON: {msg}")

if __name__ == "__main__":
    test_dctd()