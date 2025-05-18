import os
import pandas as pd
import librosa
import numpy as np
import sounddevice as sd
import speech_recognition as sr
from scipy.io.wavfile import write
from scipy.spatial.distance import euclidean

# Constants
csv_path = 'voice_data.csv'
duration = 2        # Duration of recording in seconds
sample_rate = 44100   # Sample rate in Hertz (CD quality)

# Load or initialize DataFrame
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    df["mfccs"] = df["mfccs"].apply(lambda x: np.fromstring(x.strip("[]"), sep=' '))
else:
    df = pd.DataFrame(columns=["pitch_hz", "loudness_db", "mfccs"])

def speech_to_text():
    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Use microphone as source
    with sr.Microphone() as source:

        audio = recognizer.listen(source)  # Listen for audio

        try:
            # Recognize speech using Google's free Web API
            text = recognizer.recognize_google(audio)
            print("You said:", text)
            return text
        except sr.UnknownValueError:
            print("Sorry, could not understand the audio.")
        except sr.RequestError:
            print("Could not request results from Google Speech Recognition service.")


#delet the audio file
def deletefile(file_path):         
    os.remove(file_path)

# analyzing the audio file feature
def analyze_audio(file_path):
    #load the audio file
    y_raw, sr = librosa.load(file_path, sr=None)
    #trim the silence portion of the file
    y, _ = librosa.effects.trim(y_raw, top_db=10)

    # --- Pitch (Fundamental Frequency) ---
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    pitch_hz = np.nanmean(f0)

    # --- Loudness (RMS Energy) ---
    rms = librosa.feature.rms(y=y)
    loudness_db = np.mean(librosa.amplitude_to_db(rms))

    # --- Timbre (MFCCs) ---
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    avg_mfccs = np.mean(mfccs, axis=1)

    #return the pitch, loudnedd anf MFCCs
    return {
        "pitch_hz": pitch_hz,
        "loudness_db": loudness_db,
        "mfccs": avg_mfccs
    }

#function to add the dataof user into a csv file
def add_datframe(file_path, user_name):
    global df
    features = analyze_audio(file_path)
    features["mfccs"] = np.array2string(features["mfccs"], separator=' ', precision=2)
    df.loc[user_name] = features
    df.to_csv(csv_path)

#Compare two audio files
def compare_audio_features(feat1, feat2):
    # check if the mfccs value are store as a string and 
    # convert them into numpy array for better calculations
    if isinstance(feat1["mfccs"], str):
        feat1["mfccs"] = np.fromstring(feat1["mfccs"].strip("[]"), sep=' ')
    if isinstance(feat2["mfccs"], str):
        feat2["mfccs"] = np.fromstring(feat2["mfccs"].strip("[]"), sep=' ')
    
    #calculate the difference between pitch , loudness and mfcc of two audio files
    pitch_diff = abs(feat1["pitch_hz"] - feat2["pitch_hz"])
    loudness_diff = abs(feat1["loudness_db"] - feat2["loudness_db"])
    mfcc_distance = euclidean(feat1["mfccs"], feat2["mfccs"])

    return pitch_diff, loudness_diff, mfcc_distance

#calculate how much similar two files are
def similarity_score(pitch_diff, loudness_diff, mfcc_dist):
     # Normalize differences to similarity scores between 0 and 1
    pitch_score = max(0, 1 - (pitch_diff / 30))  # Max 30 Hz
    loudness_score = max(0, 1 - (loudness_diff / 10))  # Max 10 dB
    mfcc_score = max(0, 1 - (mfcc_dist / 100))  # Max 100 MFCC dist
     
    # Weighted sum in percentage
    return (pitch_score * 0.3 + loudness_score * 0.2 + mfcc_score * 0.5) * 100


#comparison
def comparison(features):
    similarity = []
    for index, row in df.iterrows():                     # for df dataframe
        pitch_diff, loudness_diff, mfcc_dist = compare_audio_features(row, features.iloc[0])
        score = similarity_score(pitch_diff, loudness_diff, mfcc_dist)
        similarity.append((index, score))

    
    # Show the most similar rows (you can change the number of results shown)
    print("\nRESULT: similar audio features:")
    for idx, score in similarity:
        print(f"Index: {idx}, Similarity: {score:.2f}%")
        if(score > 50.00):        #check the similarity score
            return "yes",idx
            break
        
    return "no",None


# Main menu
while True:
    print("\n1} Enter a new user")
    print("2} Delete a user")
    print("3} Compare voice")
    print("4} Exit the application\n")

    choice = int(input("Enter your choice: "))
    #add a user
    if choice == 1:

        #creat a new user
        name = input("Enter name of the user: ")

        # record recording of user
        print("Recording...")
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2, dtype='int16')
        sd.wait()           
        file_path = f"{name}.wav"    #make a variable name for file_path
        write(file_path, sample_rate, audio_data) # actual save th efile in File_Path
        
        #save the data of audio file in a csv file
        add_datframe(file_path, name)
        print("User added successfully.")
        #v delete the audio file after saving the data
        deletefile(file_path)

    #detel the user
    elif choice == 2:
        for index in df.index:
            print(index)

        name = input("Enter name to delete: ")
        #check wether the user is recrded or not
        if name in df.index:
            df = df.drop(index=name)   #delete the user
            df.to_csv(csv_path)        #save the new data in csv file
            print(f"User {name} deleted.")
        else:
            print("User not found.")

    #compare the voice from the existing data
    elif choice == 3:
        print("Recording for comparison...")    #record a new file
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2, dtype='int16')
      
        #  run speech to text function
        speech_to_text()
        temp_file = "temp_compare.wav"     #make a path for storing audio file
        print("recorded.")
        write(temp_file, sample_rate, audio_data)       #store the audio file
        features = analyze_audio(temp_file)             #analyze the recorded file
        features["mfccs"] = np.array2string(features["mfccs"], separator=' ', precision=2)
        features_df = pd.DataFrame([features])   #creat a new datafram for comparision
        result,Name= comparison(features_df)     #compare the recorded file wir=th th edata frame
        
        #check wether he recorded file is similar or not
        if (result == "yes"):
             print("\nResult: Voice is Similar")
             print("User: ",Name)
        else:
            print("Different voice")

        #delet the recorded file
        deletefile(temp_file)

    #exit the program
    elif choice == 4:
        print("Exiting application.")
        break

    else:
        print("Invalid command. Try again.")
