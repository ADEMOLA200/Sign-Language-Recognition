## **Overview**  
This project, *Sign Language to Speech Conversion using Machine Learning*, enables the recognition of American Sign Language (ASL) gestures and converts them into spoken words or sentences. It is designed to bridge the communication gap for individuals who use sign language.

## 🎯 Features

- **Real-Time Recognition**: Converts live webcam feed into recognized gestures.
- **Robust Prediction**: Implements stabilization for accurate gesture recognition.
- **Word and Sentence Formation**: Automatically forms words and sentences with proper segmentation (space and full stop gestures).
- **Text-to-Speech Conversion**: Speaks out the recognized text for better accessibility.
- **User-Friendly GUI**: Displays the current alphabet, word, and sentence in real time.

## **Project Highlights**  
- **Custom Dataset Creation**:  
  We built this project from the ground up by capturing and preparing our **custom dataset** of ASL gestures, covering A-Z alphabets, 0-9 digits, a gesture for **space**, and one for **full stop**.
  
  ![ASL Characters](/ReadmeAssets/The-26-letters-and-10-digits-of-American-Sign-Language-ASL.png)
  The 26 letters and 10 digits of American Sign Language (ASL)

- **Pre-Trained Model Included**:  
  A **pre-trained model** is provided to allow users to immediately start using the system.  
- **Flexibility**:  
  Users can also train their own custom models using the provided scripts and instructions.

## 🔧 Tech Stack
- **Languages**: Python
- **Libraries**: MediaPipe, OpenCV, Tkinter, Pyttsx3, Scikit-learn
- **Machine Learning Model**: Random Forest Classifier

---

## **Usage**  
### **Option 1: Use the Pre-Trained Model**  
You can directly use the pre-trained model (`model.p`) provided in this repository:  
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/sign-language-to-speech.git
   cd sign-language-to-speech
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the `main.py` script.  
4. Ensure your webcam is functional, and the `model.p` file is in the same directory.  

### **Option 2: Train Your Own Model**  
To create and train a custom model:  
1. **Collect a Custom Dataset**:  
   - Run `collectImgs.py` to capture gesture images using your webcam.  
   - Modify the `number_of_classes` and `dataset_size` variables in the script if needed.  

2. **Process the Dataset**:  
   - Use `createDataset.py` to extract features from the collected images and save them in a `data.pickle` file.  

3. **Train the Model**:  
   - Train a new model using `trainClassifier.py`. It will replace the old one.

---

## **How the Project Was Built**  
1. **Dataset Collection**  
   - A custom dataset was created using the `collectImgs.py` script.  
   - For each gesture, 100 images were captured, ensuring diverse angles and lighting conditions for robust training.  
   - The dataset includes **38 classes**, representing:  
     - 26 alphabets (A-Z)  
     - 10 digits (0-9)  
     - A gesture for **space**  
     - A gesture for **full stop**  

2. **Feature Extraction and Preprocessing**  
   - The collected images were processed using **MediaPipe**, extracting 21 key landmarks for each hand.  
   - Each landmark was converted into a normalized 2D coordinate (x, y), resulting in **42 features per sample**.  
   - The preprocessed data was saved as a `pickle` file using the `createDataset.py` script.  

3. **Model Training**  
   - A **Random Forest Classifier** was used for training, offering high accuracy with fast training times.  
   - The dataset was split into training and testing sets (80-20 ratio).  
   - Training results showed high accuracy, ensuring reliable gesture recognition.  

4. **Inference and Real-Time Conversion**  
   - The trained model was integrated into a real-time system using `main.py`.  
   - The system uses a **stabilization buffer** to improve gesture detection and avoids misclassifications.  
   - A **Text-to-Speech (TTS)** engine was used to convert recognized gestures into audible speech.  
   - A user-friendly **GUI** was built with **Tkinter** for better interaction.

---

## **Future Enhancements**  
- Expand recognition to include **dynamic gestures**.  
- Support for additional sign languages.  
- Deploy the system as a mobile or web application for greater accessibility.
- Improve model accuracy with advanced deep learning techniques.

---

## **Suggestions and Collaborations**  
I believe that collaboration is key to innovation. If you have suggestions for improving this project, new ideas, or wish to collaborate on expanding its capabilities, I’d love to hear from you! Feel free to reach out via issues or pull requests on GitHub.

---

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
