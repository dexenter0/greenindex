import uvicorn
import os
import sys

# Ensure the app module is found in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("==================================================")
    print("🌱 Starting GreenIndex - Chlorophyll Deficiency Detection System")
    print("==================================================")
    print("Server will be available at: http://127.0.0.1:8000")
    print("Press CTRL+C to quit")
    
    # Run the FastAPI application
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
