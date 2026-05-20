import requests
import json
import time

def test_query(category, query):
    url = "http://localhost:11435/api/chat"
    payload = {
        "model": "qwen3.5:9b",
        "messages": [
            {"role": "user", "content": query}
        ],
        "stream": False
    }
    
    print(f"\n=== Testing Category: {category} ===")
    print(f"Query: {query}")
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=300) # Long timeout as requested
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            content = result.get('message', {}).get('content', 'No content')
            print(f"Response (Time: {end_time - start_time:.2f}s):\n{content}")
            return content
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Request failed: {e}")
        return None

test_cases = [
    ("Facts", "What is the capital of France and what is its current population?"),
    ("Casual", "How are you doing today, Adelaide? Tell me a joke."),
    ("Mathematics", "What is the derivative of x^2 * sin(x) with respect to x?"),
    ("Coding", "Write a Python script to calculate the Fibonacci sequence up to n terms using recursion."),
    ("Philosophy and Ethics", "Discuss the Trolley Problem from the perspective of utilitarianism vs. deontology.")
]

if __name__ == "__main__":
    results = {}
    for category, query in test_cases:
        results[category] = test_query(category, query)
        time.sleep(2) # Small delay between tests
    
    with open("test_results_summary.json", "w") as f:
        json.dump(results, f, indent=2)
