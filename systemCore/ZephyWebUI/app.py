from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import platform
import psutil
import random
import time
from datetime import datetime

# Initialize Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")

# Global variables for session data
username = os.getlogin()
assistant_name = "Adelaide Zephyrine Charlotte"
engine_name = "Adelaide Paradigm Engine"
profile_picture_emotions = "happy"  # Default emotion
responses = []  # To store chat history
gen = 0  # Message ID counter

# Helper functions
def get_system_stats():
    """Fetch system stats like CPU usage, memory, etc."""
    cpu_usage = psutil.cpu_percent(interval=1)
    total_memory = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    free_memory = round(psutil.virtual_memory().available / (1024 ** 3), 1)
    return {
        "cpu_usage": cpu_usage,
        "total_memory": total_memory,
        "free_memory": free_memory,
    }

def generate_response(prompt):
    """Simulate a response from the backend."""
    # Simulate a delay for processing
    time.sleep(random.uniform(0.5, 2.0))
    return f"Processed: {prompt}"

# Routes
@app.route("/")
def index():
    """Serve the main HTML page."""
    return render_template("index.html", username=username, assistant_name=assistant_name)

@app.route("/api/get_params", methods=["GET"])
def get_params():
    """Return configuration parameters."""
    params = {
        "username": username,
        "assistantName": assistant_name,
        "engineName": engine_name,
        "platform": platform.system(),
        "arch": platform.machine(),
    }
    return jsonify(params)

@app.route("/api/system_stats", methods=["GET"])
def system_stats():
    """Return system statistics."""
    stats = get_system_stats()
    return jsonify(stats)

@app.route("/api/send_message", methods=["POST"])
def send_message():
    """Handle user messages and return responses."""
    global gen
    data = request.json
    prompt = data.get("message", "")
    
    # Simulate processing and generate a response
    response_text = generate_response(prompt)
    response_id = f"user{gen}"
    gen += 1

    # Store the response
    responses.append({"id": response_id, "text": response_text})

    # Return the response
    return jsonify({"id": response_id, "response": response_text})

@app.route("/api/get_chat_history", methods=["GET"])
def get_chat_history():
    """Return the chat history."""
    return jsonify(responses)

@app.route("/api/update_emotion", methods=["POST"])
def update_emotion():
    """Update the profile picture emotion."""
    global profile_picture_emotions
    data = request.json
    new_emotion = data.get("emotion", "happy")
    profile_picture_emotions = new_emotion
    return jsonify({"status": "success", "emotion": profile_picture_emotions})

@app.route("/api/clear_chat", methods=["POST"])
def clear_chat():
    """Clear the chat history."""
    global responses, gen
    responses = []
    gen = 0
    return jsonify({"status": "success"})

# Serve static files (CSS, JS, images, etc.)
@app.route("/<path:filename>")
def serve_static(filename):
    """Serve static files from the root directory."""
    if filename.endswith(".js"):
        return send_from_directory("static/js", filename, mimetype="application/javascript")
    elif filename.endswith(".css"):
        return send_from_directory("static/css", filename, mimetype="text/css")
    elif filename.endswith((".png", ".jpg", ".jpeg", ".gif", ".ico")):
        return send_from_directory("static/assets", filename, mimetype="image/*")
    else:
        return send_from_directory("static", filename)

# Run the app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=6969)