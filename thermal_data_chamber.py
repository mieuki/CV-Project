#%%
#New calibrated BB inside the chamber result analysis
import cv2
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os

# %%
# Thermal Images

base_path = r"C:\Users\Muskan\Desktop\collimator\data\FLIR_Knife_Tests\FLIR_Slantedge_NewBB"
folders = ["30C", "40C", "50C", "60C", "70C", "80C"]

# Display the original images in one figure
fig_original = plt.figure(figsize=(10, 8))

for i, folder in enumerate(folders):
    folder_path = os.path.join(base_path, folder)
    tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    
    if tif_files:
        first_tif_file = os.path.join(folder_path, tif_files[0])
        image = cv2.imread(first_tif_file, cv2.IMREAD_UNCHANGED)
        
        ax = fig_original.add_subplot(2, 3, i + 1)
        ax.set_title(f"Original Image {folder}")
        ax.imshow(image, cmap='gray')

fig_original.tight_layout()
plt.show()

# Display the cropped images in another figure
fig_cropped = plt.figure(figsize=(10, 8))

for i, folder in enumerate(folders):
    folder_path = os.path.join(base_path, folder)
    tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    
    if tif_files:
        first_tif_file = os.path.join(folder_path, tif_files[0])
        image = cv2.imread(first_tif_file, cv2.IMREAD_UNCHANGED)
        
        # Crop the image
        x1, x2 = 200, 350
        y1, y2 = 150, 300
        cropped_image = image[y1:y2, x1:x2]
        
        ax = fig_cropped.add_subplot(2, 3, i + 1)
        ax.set_title(f"Cropped Image {folder}")
        ax.imshow(cropped_image, cmap='gray')
        ax.set_aspect('equal')  # Ensure the x and y axes have the same scale

fig_cropped.tight_layout()
plt.show()

#Display the roi
fig_roi = plt.figure(figsize=(10, 8))
for i, folder in enumerate(folders):
    folder_path = os.path.join(base_path, folder)
    tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    
    if tif_files:
        first_tif_file = os.path.join(folder_path, tif_files[0])
        image = cv2.imread(first_tif_file, cv2.IMREAD_UNCHANGED)
        
        # Crop the image
        x1, x2 = 200, 350
        y1, y2 = 150, 300
        cropped_image = image[y1:y2, x1:x2]
        
        roi_x1, roi_x2 = 65, 65+41
        roi_y1, roi_y2 = 70, 71
        roi = cropped_image[roi_y1:roi_y2, roi_x1:roi_x2]
        
        ax = fig_roi.add_subplot(6, 1, i + 1)
        ax.set_title(f"ROI {folder}")
        ax.imshow(roi, cmap='gray')
        
        
fig_roi.tight_layout()
plt.show()

# %%
for folder in folders:
    folder_path = os.path.join(base_path, folder)
    tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    
    if tif_files:
        # Initialize an array to accumulate the sum of all images
        sum_image = None
        num_images = len(tif_files)
        
        for tif_file in tif_files:
            file_path = os.path.join(folder_path, tif_file)
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            
            if image is not None:
                if sum_image is None:
                    sum_image = np.zeros_like(image, dtype=np.float64)
                sum_image += image
            else:
                print(f"Warning: Failed to read {file_path}")
        
        # Compute the average image
        avg_image = sum_image / num_images
        
        # Display the averaged image
        plt.figure()
        plt.title(f"Averaged Image for {folder}")
        plt.imshow(avg_image, cmap='gray')
        plt.colorbar()
        plt.show()
        
        
        # Crop the averaged image
        x1, x2 = 200, 350
        y1, y2 = 150, 300
        cropped_image = avg_image[y1:y2, x1:x2]
        
        #Display the cropped image
        plt.figure()
        plt.title(f"Cropped Averaged Image for {folder}")
        plt.imshow(cropped_image, cmap='gray')
        plt.colorbar()
        plt.show()
        
        roi_x1, roi_x2 = 65, 65+41
        roi_y1, roi_y2 = 70, 71
        roi = cropped_image[roi_y1:roi_y2, roi_x1:roi_x2]
        plt.figure(figsize=(8, 0.5))
        plt.title(f"ROI for {folder}")
        plt.imshow(roi, cmap='gray', aspect='auto')
        plt.show()
        
        # Compute the average intensity values along the horizontal axis of the ROI
        edge_spread = (np.mean(roi, axis=0))
        
        # Normalize the edge spread function between 0 and 1
        edge_spread = (edge_spread - np.min(edge_spread)) / (np.max(edge_spread) - np.min(edge_spread))
        # edge_spread = edge_spread / np.max(edge_spread)
        # Store the edge spread function for plotting
        if 'edge_spread_data' not in locals():
            edge_spread_data = {}
        edge_spread_data[folder] = edge_spread

        # After the loop, plot the edge spread functions for all categories
        plt.figure(figsize=(10, 8))
        for folder, edge_spread in edge_spread_data.items():
            plt.plot(edge_spread, label=folder)

        plt.title("Edge Spread Function new BB")
        plt.xlabel("Pixel Position")
        plt.ylabel("Normalized Intensity")
        plt.legend(title="Temperature")
        plt.grid()
        plt.show()
        
        if 'lsf_data' not in locals():
            lsf_data = {}
        
        # Compute the Line Spread Function (LSF) as the derivative of the edge spread function
        lsf = (np.diff(edge_spread))
        
        # Normalize the LSF between 0 and 1
        lsf = (lsf - np.min(lsf)) / (np.max(lsf) - np.min(lsf))
        
        lsf_data[folder] = lsf

    # After the loop, plot the line spread functions for all categories
    plt.figure(figsize=(10, 8))
    
    # Sort the folders to ensure the legend is in order
    sorted_folders = sorted(lsf_data.keys(), key=lambda x: int(x[:1]))
    
    for folder in sorted_folders:
        plt.plot(lsf_data[folder], label=folder)

    plt.title("Line Spread Function new BB")
    plt.xlabel("Pixel Position")
    plt.ylabel("Normalized Intensity")
    plt.legend(title="Temperature")
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(10, 8))

    # Sort the folders to ensure the legend is in order
    sorted_folders = sorted(lsf_data.keys(), key=lambda x: int(x[:-1]))

    for folder in sorted_folders:
        # Compute the MTF using the FFT of the LSF
        lsf = lsf_data[folder]
        mtf = sp.fft.fftshift(sp.fft.fft(lsf))
        mtf_norm = np.abs(mtf) / np.max(np.abs(mtf))
        
        # Extract the positive frequencies
        N = len(lsf)
        d = 1
        freq = sp.fft.fftshift(sp.fft.fftfreq(N, d=d))
        n_half = N // 2  # Half the length of the signal
        freq = freq[int(n_half):]
        mtf_norm = mtf_norm[int(n_half):]
        
        # Plot the MTF curve
        plt.plot(freq, mtf_norm, linestyle='--', marker='.', label=f'MTF {folder}')

    plt.title('Modulation Transfer Function New BB')
    plt.xlabel('Spatial Frequency (cycles/pixel)')
    plt.ylabel('Modulation Transfer')
    plt.legend(title="Temperature")
    plt.grid()
    plt.show()


# %%
