# lighting_test.py
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
import pickle
import pandas as pd
from tabulate import tabulate

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

# Load trained model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Lighting conditions to test
CONDITIONS = ['optimal', 'low_light', 'backlit', 'mixed_shadow']
results = []

def test_lighting_condition(condition):
    """Test model accuracy under specific lighting condition"""
    base_path = f"./test_data/{condition}"
    correct = 0
    total = 0
    class_correct = [0] * 38
    class_total = [0] * 38
    
    # Walk through all class directories
    for class_dir in os.listdir(base_path):
        class_path = os.path.join(base_path, class_dir)
        if not os.path.isdir(class_path):
            continue
            
        class_idx = int(class_dir)
        
        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_path, img_file)
                img = cv2.imread(img_path)
                
                if img is None:
                    print(f"Warning: Could not read image {img_path}")
                    continue
                    
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results_mp = hands.process(img_rgb)
                
                if results_mp.multi_hand_landmarks:
                    for hand_landmarks in results_mp.multi_hand_landmarks:
                        data_aux = []
                        x_ = []
                        y_ = []
                        
                        # Extract landmarks
                        for landmark in hand_landmarks.landmark:
                            x = landmark.x
                            y = landmark.y
                            x_.append(x)
                            y_.append(y)
                        
                        # Normalize landmarks
                        for landmark in hand_landmarks.landmark:
                            x = landmark.x
                            y = landmark.y
                            data_aux.append(x - min(x_))
                            data_aux.append(y - min(y_))
                        
                        # Make prediction if we have enough features
                        if len(data_aux) == 42:
                            prediction = model.predict([np.asarray(data_aux)])
                            predicted_class = int(prediction[0])
                            
                            if predicted_class == class_idx:
                                correct += 1
                                class_correct[class_idx] += 1
                            
                            total += 1
                            class_total[class_idx] += 1
    
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"{condition.upper()}: {correct}/{total} correct ({accuracy:.2f}% accuracy)")
    
    # Add to results
    results.append({
        'condition': condition,
        'test_samples': total,
        'correct': correct,
        'accuracy': accuracy
    })
    
    return accuracy

def capture_test_images():
    """Capture test images under different lighting conditions"""
    print("\n===== CAPTURING TEST IMAGES =====")
    conditions = ['optimal', 'low_light', 'backlit', 'mixed_shadow']
    samples_per_class = 5  # Number of samples per class per condition
    classes = range(38)  # 0-37
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Create directories if they don't exist
    for condition in conditions:
        for class_idx in classes:
            dir_path = f"./test_data/{condition}/{class_idx}"
            os.makedirs(dir_path, exist_ok=True)
    
    for condition in conditions:
        print(f"\n=== {condition.upper()} LIGHTING ===")
        input(f"Set up '{condition}' lighting, then press Enter to start capturing...")
        
        for class_idx in classes:
            class_name = {
                0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 
                8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 
                16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 
                24: 'Y', 25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 
                32: '6', 33: '7', 34: '8', 35: '9', 36: 'SPACE', 37: 'FULLSTOP'
            }.get(class_idx, str(class_idx))
            
            print(f"\nShow gesture for: {class_name} ({class_idx})")
            
            for sample_num in range(samples_per_class):
                print(f"  Sample {sample_num+1}/{samples_per_class} - Get ready...")
                
                # Countdown
                for i in range(3, 0, -1):
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    display_frame = frame.copy()
                    cv2.putText(display_frame, f"{class_name}: {i}...", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow('Capture', display_frame)
                    cv2.waitKey(1000)
                
                # Capture image
                ret, frame = cap.read()
                if ret:
                    save_path = f"./test_data/{condition}/{class_idx}/sample_{sample_num}.jpg"
                    cv2.imwrite(save_path, frame)
                    print(f"  Saved: {save_path}")
                    
                    # Show captured image briefly
                    display_frame = frame.copy()
                    cv2.putText(display_frame, "CAPTURED!", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Capture', display_frame)
                    cv2.waitKey(500)
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nTest image capture complete!")

def generate_results_table():
    """Generate and save results table as image"""
    # Create DataFrame from results
    df = pd.DataFrame(results)
    
    # Format accuracy with percentage
    df['Accuracy'] = df['accuracy'].apply(lambda x: f"{x:.1f}%")
    
    # Create pretty table
    table = tabulate(df[['condition', 'test_samples', 'correct', 'Accuracy']], 
                    headers=['Lighting Condition', 'Test Samples', 'Correct', 'Accuracy'],
                    tablefmt='pretty', showindex=False)
    
    print("\n===== TEST RESULTS =====")
    print(table)
    
    # Save table as text file
    with open('lighting_results.txt', 'w') as f:
        f.write("Sign Language Recognition - Lighting Condition Test Results\n")
        f.write("="*60 + "\n\n")
        f.write(table)
    
    # Create table image using matplotlib
    plt.figure(figsize=(10, 4))
    plt.axis('off')
    plt.title("Lighting Condition Test Results", fontsize=16)
    
    # Create table from DataFrame
    table_data = [['Condition', 'Samples', 'Correct', 'Accuracy']]
    for _, row in df.iterrows():
        table_data.append([
            row['condition'].capitalize(),
            row['test_samples'],
            row['correct'],
            f"{row['accuracy']:.1f}%"
        ])
    
    plt.table(cellText=table_data[1:], 
              colLabels=table_data[0],
              loc='center', 
              cellLoc='center')
    
    plt.tight_layout()
    plt.savefig('lighting_results_table.png', dpi=150, bbox_inches='tight')
    print("\nSaved results table as 'lighting_results_table.png'")

def plot_accuracy_comparison():
    """Plot accuracy comparison across conditions"""
    plt.figure(figsize=(10, 6))
    
    # Extract data
    conditions = [r['condition'].capitalize() for r in results]
    accuracies = [r['accuracy'] for r in results]
    
    # Create bar plot
    bars = plt.bar(conditions, accuracies, color=[
        '#4CAF50',  # Optimal - green
        '#FFC107',  # Low light - amber
        '#F44336',  # Backlit - red
        '#9C27B0'   # Mixed shadow - purple
    ])
    
    plt.ylim(0, 100)
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Under Different Lighting Conditions', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 1, 
                f'{height:.1f}%', ha='center', fontsize=10)
    
    # Add explanatory note
    plt.figtext(0.5, 0.01, 
               "Note: Tested with 5 samples per gesture (38 gestures) across 4 lighting conditions",
               ha="center", fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig('lighting_accuracy_comparison.png', dpi=300)
    print("Saved accuracy comparison plot as 'lighting_accuracy_comparison.png'")

def main():
    """Main function to run lighting tests"""
    # Create test_data directory structure
    os.makedirs('./test_data', exist_ok=True)
    for condition in CONDITIONS:
        os.makedirs(f'./test_data/{condition}', exist_ok=True)
    
    # Ask user if they want to capture new images
    if not os.listdir('./test_data/optimal'):
        print("No test images found in test_data directory.")
        capture = input("Do you want to capture test images now? (y/n): ").lower()
        if capture == 'y':
            capture_test_images()
        else:
            print("Please place test images in test_data directory and run again.")
            return
    
    # Run tests for each lighting condition
    print("\n===== RUNNING TESTS =====")
    for condition in CONDITIONS:
        test_lighting_condition(condition)
    
    # Generate and save results
    generate_results_table()
    plot_accuracy_comparison()
    
    print("\nTesting complete!")

if __name__ == "__main__":
    main()
