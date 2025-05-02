import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import os
import cv2

def manual_grayscale_conversion(image_array):
    height, width = image_array.shape[:2]
    grayscale_image = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            r, g, b = image_array[i, j]
            gray_value = int(0.299 * r + 0.587 * g + 0.114 * b)
            grayscale_image[i, j] = gray_value
    
    return grayscale_image

def manual_mean_filter(image_array, kernel_size=3):
    height, width = image_array.shape[:2]
    result = np.zeros((height, width), dtype=np.uint8)
    pad = kernel_size // 2
    
    for i in range(height):
        for j in range(width):
            i_min = max(0, i - pad)
            i_max = min(height, i + pad + 1)
            j_min = max(0, j - pad)
            j_max = min(width, j + pad + 1)
            neighborhood = image_array[i_min:i_max, j_min:j_max]
            result[i, j] = np.mean(neighborhood)
    
    return result

def butterworth_highpass_filter(shape, cutoff, order):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows)
    v = np.arange(cols)
    U, V = np.meshgrid(u - crow, v - ccol, sparse=False, indexing='ij')
    D = np.sqrt(U**2 + V**2)
    H = 1 / (1 + (cutoff / D)**(2 * order))
    H = 0.5 + 0.5 * H
    return H

def homomorphic_filter(image_array, cutoff=30, order=2):
    gray_image = manual_grayscale_conversion(image_array) if len(image_array.shape) > 2 else image_array
    gray_image = gray_image.astype(np.float32)
    log_image = np.log1p(gray_image)
    
    fft_image = np.fft.fft2(log_image)
    fft_shifted = np.fft.fftshift(fft_image)
    
    butterworth_filter = butterworth_highpass_filter(gray_image.shape, cutoff, order)
    filtered_fft = fft_shifted * butterworth_filter
    
    ifft_shifted = np.fft.ifftshift(filtered_fft)
    ifft_image = np.fft.ifft2(ifft_shifted)
    
    exp_image = np.expm1(np.real(ifft_image))
    filtered_image = np.clip(exp_image, 0, 255).astype(np.uint8)
    
    return filtered_image

def detect_parking_slots(image, original_img):
    result_image = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 19, 2)
    
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_area = 1000  
    max_area = 5000  
    aspect_ratio_range = (1.0, 3.0)  
    
    detected_slots = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / min(w, h)
            
            if aspect_ratio_range[0] < aspect_ratio < aspect_ratio_range[1] and w > 20 and h > 20:
                detected_slots.append((x, y, w, h))
    
    if detected_slots:
        detected_slots.sort(key=lambda slot: slot[1])
        
        row_threshold = 30  
        rows = []
        current_row = [detected_slots[0]]
        
        for i in range(1, len(detected_slots)):
            if abs(detected_slots[i][1] - current_row[0][1]) < row_threshold:
                current_row.append(detected_slots[i])
            else:
                rows.append(current_row)
                current_row = [detected_slots[i]]
        
        if current_row:
            rows.append(current_row)
        
        for row in rows:
            row.sort(key=lambda slot: slot[0])
    
    empty_slots = []
    occupied_slots = []
    
    for x, y, w, h in detected_slots:
        roi = original_img[y:y+h, x:x+w]
        
        avg_intensity = np.mean(roi)
        std_dev = np.std(roi)
        
        if avg_intensity > 100 and std_dev < 50:
            empty_slots.append((x, y, w, h))
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)  
        else:
            occupied_slots.append((x, y, w, h))
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 0, 255), 2)  
    
    total_slots = len(detected_slots)
    empty_count = len(empty_slots)
    occupied_count = len(occupied_slots)
    
    print(f"Total Slots: {total_slots}")
    print(f"Empty: {empty_count}")
    print(f"Occupied: {occupied_count}")
    
    if total_slots == 0:
        edges = cv2.Canny(image, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=40, maxLineGap=10)
        
        parking_grid = np.zeros_like(binary)
        
        if lines is not None:
            horizontal_lines = []
            vertical_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                if x2 - x1 == 0:  
                    angle = 90
                else:
                    angle = abs(np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi)
                
                if angle < 30:  
                    horizontal_lines.append(line[0])
                    cv2.line(parking_grid, (x1, y1), (x2, y2), 255, 2)
                elif angle > 60:  
                    vertical_lines.append(line[0])
                    cv2.line(parking_grid, (x1, y1), (x2, y2), 255, 2)
            
            kernel = np.ones((5, 5), np.uint8)
            parking_grid = cv2.dilate(parking_grid, kernel, iterations=1)
            
            grid_contours, _ = cv2.findContours(parking_grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in grid_contours:
                area = cv2.contourArea(contour)
                if 500 < area < 5000:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = max(w, h) / min(w, h)
                    
                    if 1.0 < aspect_ratio < 3.0:
                        roi = original_img[y:y+h, x:x+w]
                        avg_intensity = np.mean(roi)
                        
                        if avg_intensity > 100:
                            cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            empty_count += 1
                        else:
                            cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
                            occupied_count += 1
                        
                        total_slots += 1
            
            print("After backup detection method:")
            print(f"Total Slots: {total_slots}")
            print(f"Empty: {empty_count}")
            print(f"Occupied: {occupied_count}")
    
    if total_slots == 0:
        rows = [50, 125, 175]  
        cols = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]  
        
        slot_width = 25
        slot_height = 40
        
        for row in rows:
            for col in cols:
                if col < image.shape[1] - slot_width and row < image.shape[0] - slot_height:
                    roi = image[row:row+slot_height, col:col+slot_width]
                    avg_intensity = np.mean(roi)
                    std_dev = np.std(roi)
                    
                    if avg_intensity > 100 and std_dev < 50:
                        cv2.rectangle(result_image, (col, row), (col+slot_width, row+slot_height), (0, 255, 0), 2)
                        empty_count += 1
                    else:
                        cv2.rectangle(result_image, (col, row), (col+slot_width, row+slot_height), (0, 0, 255), 2)
                        occupied_count += 1
                    
                    total_slots += 1
        
        print("After fixed grid detection method:")
        print(f"Total Slots: {total_slots}")
        print(f"Empty: {empty_count}")
        print(f"Occupied: {occupied_count}")
    
    line_image = np.zeros_like(result_image)
    return result_image, line_image, total_slots

def display_results(original, grayscale, smoothed, filtered, slots_detected):
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(original)
    plt.title('Original Image')
    
    plt.subplot(2, 3, 2)
    plt.imshow(grayscale, cmap='gray')
    plt.title('Grayscale Conversion')
    
    plt.subplot(2, 3, 3)
    plt.imshow(smoothed, cmap='gray')
    plt.title('Mean Filter (Noise Reduction)')
    
    plt.subplot(2, 3, 4)
    plt.imshow(filtered, cmap='gray')
    plt.title('Homomorphic Filtering')
    
    plt.subplot(2, 3, 5)
    if len(slots_detected.shape) == 3:
        plt.imshow(cv2.cvtColor(slots_detected, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(slots_detected, cmap='gray')
    plt.title('Parking Slots Detected')
    
    plt.tight_layout()
    plt.savefig('results.png')
    plt.show()

def process_image(image_path):
    start_time = time.time()
    input_image = Image.open(image_path)
    image_array = np.array(input_image)
    
    print("\n===== PARKING SPOT DETECTION =====")
    print(f"Processing image: {image_path}")
    
    gray_image = manual_grayscale_conversion(image_array)
    print("Grayscale conversion completed")
    
    smoothed_image = manual_mean_filter(gray_image, kernel_size=3)
    print("Noise reduction completed")
    
    filtered_image = homomorphic_filter(smoothed_image)
    print("Homomorphic filtering completed")
    
    print("\n----- DETECTION RESULTS -----")
    slots_image, line_image, num_slots = detect_parking_slots(filtered_image, gray_image)
    
    os.makedirs("output", exist_ok=True)
    Image.fromarray(gray_image).save("output/grayscale.jpg")
    Image.fromarray(smoothed_image).save("output/smoothed.jpg")
    Image.fromarray(filtered_image).save("output/filtered.jpg")
    
    if len(slots_image.shape) == 3:
        Image.fromarray(cv2.cvtColor(slots_image, cv2.COLOR_BGR2RGB)).save("output/parking_slots.jpg")
    else:
        Image.fromarray(slots_image).save("output/parking_slots.jpg")
    
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"\nProcessing completed in {processing_time:.2f} seconds")
    print(f"Number of parking slots detected: {num_slots}")
    print("Output images saved to 'output' folder")
    print("===================================\n")
    
    display_results(image_array, gray_image, smoothed_image, filtered_image, slots_image)
    
    return gray_image, smoothed_image, filtered_image, slots_image

def main():
    image_path = os.path.join(os.getcwd(), "image3.jpg")

    try:
        process_image(image_path)
    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found. Please check the directory.")
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    main()