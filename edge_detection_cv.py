#%%
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_laplace

#%%
#Canny Edge Detection
# image_path = r"C:\Users\Muskan\Desktop\collimator\data\100D5600\DSC_0101.JPG"
image_path = rimage_path = r"C:\Users\Muskan\Desktop\collimator\data\100D5600\DSC_0113.JPG"
img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
edges = cv.Canny(img,100,200)
 
plt.subplot(121)
plt.imshow(img,cmap = 'gray')
plt.title('Original Image')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image')
plt.xticks([])
plt.yticks([])
plt.show()


# %%
# Sobel Edge Detection  
def sobel_edge_detector(image_path):
    # Read the image using OpenCV
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    # Apply Sobel operator
    sobel_x = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)
    sobel_y = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)

    # Calculate gradient magnitude
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Apply a threshold to identify edges
    threshold = 50
    edges = magnitude > threshold

    return edges

# Example usage
image_path = r"C:\Users\Muskan\Desktop\collimator\data\100D5600\DSC_0113.JPG"
edge_image = sobel_edge_detector(image_path)

# Display the original and edge-detected images
original_image = cv.imread(image_path)
original_image_rgb = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)

# Plotting the images using matplotlib
plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(original_image_rgb)
plt.title('Original Image')
plt.axis('off')

# Edge-detected Image
plt.subplot(1, 2, 2)
plt.imshow(edge_image, cmap='gray')
plt.title('Edge-detected Image (Sobel)')
plt.axis('off')

# Display the images
plt.show()


# %%
# Prewitt Edge Detection
def prewitt_edge_detector(image_path):
    # Read the image using OpenCV
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    # Define Prewitt kernels
    kernel_x = np.array([[1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1]])
    kernel_y = np.array([[1, 1, 1],
                         [0, 0, 0],
                         [-1, -1, -1]])

    # Apply Prewitt operator
    prewitt_x = cv.filter2D(image, cv.CV_64F, kernel_x)
    prewitt_y = cv.filter2D(image, cv.CV_64F, kernel_y)

    # Calculate gradient magnitude
    magnitude = np.sqrt(prewitt_x**2 + prewitt_y**2)

    # Apply a threshold to identify edges
    threshold = 50
    edges = magnitude > threshold

    return edges

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(original_image_rgb)
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(prewitt_edge_detector(image_path), cmap='gray')
plt.title('Edge-detected Image (Prewitt)')
plt.axis('off')
plt.show()

# %%
#Laplacian of Gausian Edge Detection
def laplacian_of_gaussian_edge_detection(image_path, sigma = 1, threshold = None):
    # Read the image using OpenCV
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    
    log_image = gaussian_laplace(image, sigma=1)

    edge_map = np.zeros_like(log_image,dtype=np.uint8)
    for i in range(log_image.shape[0]-1):
        for j in range(log_image.shape[1]-1):
            if log_image[i,j] > 0 and log_image[i,j+1] < 0 or \
               (log_image[i, j] < 0 and log_image[i, j+1] > 0) or \
               (log_image[i, j] > 0 and log_image[i+1, j] < 0) or \
               (log_image[i, j] < 0 and log_image[i+1, j] > 0):
                edge_map[i, j] = 255

    # Apply threshold
    if threshold is not None:
        edge_map[np.abs(log_image) < threshold] = 0
        
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(original_image_rgb)
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(edge_image, cmap='gray')
plt.title('Edge-detected Image (LoG)')
plt.axis('off')
plt.show()

