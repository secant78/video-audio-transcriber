import google.generativeai as genai
import os
import time

# --- CONFIGURATION ---
API_KEY = "YOUR_GOOGLE_API_KEY_HERE"  # Paste your key here
AUDIO_FILE = "your_audio_file.mp3"     # Replace with your file path
# ---------------------

def transcribe_with_gemini(file_path):
    # 1. Configure the API
    genai.configure(api_key=API_KEY)

    print(f"1. Uploading {file_path} to Google (this may take a moment)...")
    
    # Upload the file to Google's server
    audio_file = genai.upload_file(path=file_path)
    
    # Wait for processing (usually instant for small files)
    while audio_file.state.name == "PROCESSING":
        print("   Processing audio...")
        time.sleep(2)
        audio_file = genai.get_file(audio_file.name)

    if audio_file.state.name == "FAILED":
        print("Error: Audio processing failed.")
        return

    print("2. Generating Transcript with Speaker Labels...")
    
    # Load the Gemini 1.5 Flash model (Fast & Cheap/Free)
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    # The Prompt: This is where we tell it to identify speakers
    prompt = """
    Please transcribe this audio file accurately.
    1. Identify different speakers (e.g., Speaker 1, Speaker 2).
    2. Add timestamps [MM:SS] at the start of each turn.
    3. Format the output clearly.
    """

    # Generate content
    response = model.generate_content([prompt, audio_file])
    
    # Print to console
    print("\n--- TRANSCRIPT ---\n")
    print(response.text)

    # Save to file
    with open("transcript_gemini.txt", "w", encoding="utf-8") as f:
        f.write(response.text)
    
    print("\n-----------------------------------")
    print("Saved to 'transcript_gemini.txt'")
    
    # Cleanup: Delete the file from Google's server to save space
    audio_file.delete()

if __name__ == "__main__":
    # Small helper to handle drag-and-drop paths
    if not os.path.exists(AUDIO_FILE):
        # If file path in variable is wrong, ask user
        AUDIO_FILE = input("Drag and drop your audio file here: ").strip().replace('"', '')

    transcribe_with_gemini(AUDIO_FILE)