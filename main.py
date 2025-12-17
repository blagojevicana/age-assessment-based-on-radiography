import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import cv2
from skimage.filters import threshold_otsu
import scipy.ndimage as ndi
from skimage import measure, morphology, io, color, filters
import scipy.ndimage as ndi

plt.close('all')

directory = 'C:/Users/Korisnik/Desktop/ABS projekat/wetransfer_sake_2024-04-12_0759/sake'

def hist_eq(im,L):
   img_flatten=im.flatten()
   hist,bins=np.histogram(img_flatten,bins=L)
   cdf=hist.cumsum()

   cdf=np.ma.masked_equal(cdf,0)

   M,N=im.shape
   cdf_min=np.min(cdf)

   h=np.round((cdf-cdf_min)/(M*N-cdf_min)*L)

   h=np.ma.filled(h).astype('uint8')
   img_ekv=h[im] 
   return img_ekv
   
def show_img(img, colormap = 0, title = ""):
    img = sitk.GetArrayFromImage(img)
    plt.figure()
    if colormap == 0:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    
    plt.axis('on')
    plt.title(title, fontsize = 8)

def correct_hand_position(img_array):
    threshold_value = threshold_otsu(img_array)
    binary_img = img_array > threshold_value

    contours, _ = cv2.findContours(binary_img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.round(box).astype(int)  

    angle = rect[-1]
    if (-30 <= angle <= 0) or (0 <= angle <= 30):
        center = (img_array.shape[1] // 2, img_array.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_array = cv2.warpAffine(img_array, rotation_matrix, (img_array.shape[1], img_array.shape[0]))

    return img_array


def process_image(image_path):
    img = sitk.ReadImage(image_path)
    img_array = sitk.GetArrayFromImage(img)
    img_array = img_array[0, :, :]  
    corrected_img = correct_hand_position(img_array)

    threshold_value = threshold_otsu(corrected_img)
    binary_img = corrected_img > threshold_value
    lower_half_img = binary_img[binary_img.shape[0] // 2 :, :]  
    center_of_mass = ndi.center_of_mass(lower_half_img)
    center_row, center_col = int(center_of_mass[0] + binary_img.shape[0] // 2), int(center_of_mass[1])

    rect_half_height = 175  
    rect_half_width = 210  

    min_row = max(center_row - rect_half_height, 0)
    max_row = min(center_row + rect_half_height, corrected_img.shape[0] - 1)
    min_col = max(center_col - rect_half_width, 0)
    max_col = min(center_col + rect_half_width, corrected_img.shape[1] - 1)

    cropped_img = corrected_img[min_row:max_row + 1, min_col:max_col + 1]
    
    return cropped_img

    

k=0
file_names = os.listdir(directory)
for file_name in file_names:
    print(f'Figure {k+1}')
    file_path = os.path.join(directory, file_name)
    cropped_img=process_image(file_path)
    
    img=cropped_img
    
    fig=plt.figure()
    
    plt.subplot(1,2,1)
    plt.imshow(cropped_img,cmap='gray')
    
    L=256

    img=(img-np.min(np.min(img)))/(np.max(np.max(img))-np.min(np.min(img)))*L


    img=ndi.gaussian_filter(img, sigma=3)


    img2=np.where(img<60,0,img)


    img2=img2.astype(int)
    img3=hist_eq(img2,L)

    original=sitk.GetImageFromArray(img3)
    sigma = original.GetSpacing()[0]
    gradient_img = sitk.GradientMagnitudeRecursiveGaussian(original, sigma=sigma)

    level = 3
    ws_img = sitk.MorphologicalWatershed(gradient_img,level,markWatershedLine=False,fullyConnected=False)


    original2=ws_img
    sigma2=original2.GetSpacing()[0]
    gradient_img2=sitk.GradientMagnitudeRecursiveGaussian(original2, sigma=0.5)
    level2=3
    ws_img2 = sitk.MorphologicalWatershed(gradient_img2,level2,markWatershedLine=False,fullyConnected=False)
    mid_result=sitk.GetArrayFromImage(ws_img2)
    
    if k!=8 | k!=15: 
        num_of_pixels=np.round(0.15*mid_result.shape[1]).astype(int)
    else:
        num_of_pixels=20
    
    for i in range(num_of_pixels):
        for j in range(mid_result.shape[1]):
            if mid_result[i, j] != 1:
                mid_result = np.where(mid_result == mid_result[i,j], 1, mid_result)

    for i in range(5):
        for j in range(mid_result.shape[1]):
            if mid_result[mid_result.shape[0] - i-1, j] != 1:
                mid_result = np.where(mid_result == mid_result[mid_result.shape[0] - i-1,j], 1, mid_result)

    if k!=4:
        for j in range(30):
            for i in range(mid_result.shape[0]):
                if mid_result[i, j] != 1:
                    mid_result = np.where(mid_result == mid_result[i, j], 1, mid_result)
    if k!=13:
        num_of_pixels=50
    else:
        num_of_pixels=20
    for j in range(num_of_pixels):
        for i in range(mid_result.shape[0]):
            if mid_result[i, mid_result.shape[1] - j - 1] != 1:
                mid_result = np.where(mid_result == mid_result[i, mid_result.shape[1] - j - 1], 1, mid_result)
    
    prev_mid_result=mid_result
    
    mid_result=np.where(mid_result==1,0,1) 
    mid_result,n=ndi.label(mid_result)

    unique_elements, counts = np.unique(mid_result, return_counts=True)
    if 0 in unique_elements:
        background_index = np.where(unique_elements == 0)
        unique_elements = np.delete(unique_elements, background_index)
        counts = np.delete(counts, background_index)
    if np.max(counts)>14000:
        mid_result=prev_mid_result
        mid_result = morphology.label(mid_result)
        mid_result[mid_result != -1] -= 1  
        mid_result[mid_result == -1] = 0 

    labeled_image, num_features = morphology.label(mid_result, return_num=True)

    properties = measure.regionprops(labeled_image)


    eccentricities = [prop.eccentricity for prop in properties]
    

    for i, ecc in enumerate(eccentricities):
        
        if ecc<0.1:
            mid_result=np.where(mid_result==i+1,0,mid_result)
        elif ecc>0.9:
            mid_result=np.where(mid_result==i+1,0,mid_result)
            
        
    unique_elements, counts = np.unique(mid_result, return_counts=True)
    
    for elem, count in zip(unique_elements, counts):
        if elem != 0:  
            y, x = ndi.center_of_mass(mid_result == elem)
           
    
    labeled_image, num_features = morphology.label(mid_result, return_num=True)
    properties = measure.regionprops(labeled_image)
    properties = [prop for prop in properties if prop.label != 0] 
    eccentricities = [prop.eccentricity for prop in properties]
    
    
    if 0 in unique_elements:
        background_index = np.where(unique_elements == 0)
        unique_elements = np.delete(unique_elements, background_index)
        counts = np.delete(counts, background_index)

    for i in range(len(counts)):
        if counts[i]<130:
            mid_result=np.where(mid_result==unique_elements[i],0,mid_result)
        if counts[i]>130 and counts[i]<370 and eccentricities[i]>0.67 and eccentricities[i]<0.87:
            mid_result=np.where(mid_result==unique_elements[i],0,mid_result)
        if counts[i]>9000 and counts[i]<10100 and eccentricities[i]>0.6 and eccentricities[i]<0.75:
            mid_result=np.where(mid_result==unique_elements[i],0,mid_result)
 
      
    labeled_image, num_features = morphology.label(mid_result, return_num=True)
    properties = measure.regionprops(labeled_image)
    properties = [prop for prop in properties if prop.label != 0] 
    extents = [prop.extent for prop in properties]

 
    unique_elements, counts = np.unique(mid_result, return_counts=True)
   

    if 0 in unique_elements:
        background_index = np.where(unique_elements == 0)
        unique_elements = np.delete(unique_elements, background_index)
        counts = np.delete(counts, background_index)
    for i in range(len(counts)):
        if extents[i]<0.5 and len(counts)<10 and k!=4:
            mid_result=np.where(mid_result==unique_elements[i],0,mid_result)
          
    labeled_image, num_features = morphology.label(mid_result, return_num=True)
    properties = measure.regionprops(labeled_image)
    properties = [prop for prop in properties if prop.label != 0] 
    solidities = [prop.solidity for prop in properties]
    eccentricities = [prop.eccentricity for prop in properties]
    extents = [prop.extent for prop in properties]
    unique_elements, counts = np.unique(mid_result, return_counts=True)
    if 0 in unique_elements:
        background_index = np.where(unique_elements == 0)
        unique_elements = np.delete(unique_elements, background_index)
        counts = np.delete(counts, background_index)
   
    for i in range(len(counts)):
        if eccentricities[i]>0.62 and eccentricities[i]<0.67 and extents[i]>0.61 and extents[i]<0.62:
            mid_result=np.where(mid_result==unique_elements[i],0,mid_result)
        if solidities[i]>0.8 and eccentricities[i]>0.8 and extents[i]>0.5 and extents[i]<0.58:
            mid_result=np.where(mid_result==unique_elements[i],0,mid_result)
        if solidities[i]>0.7 and solidities[i]<0.8 and extents[i]>0.5 and extents[i]<0.6:
            mid_result=np.where(mid_result==unique_elements[i],0,mid_result)
   
    plt.subplot(1,2,2)
    plt.imshow(cropped_img,cmap='gray')
    plt.imshow(mid_result,cmap='rainbow',alpha=0.3)
    
    parts = file_name.split('_')
    after_third_underscore = '_'.join(parts[3:])
    years = after_third_underscore.split('.')[0]
    gender = parts[1] if len(parts) > 1 else ''
    
    plt.subplot(1,2,1)
    plt.title(f'{k+1}. Pol: {gender}, Godine: {years}')
    
    
    unique_elements, counts = np.unique(mid_result, return_counts=True)
    num_of_bones=len(counts)-1
    print(f'{num_of_bones}')
    area=np.sum(counts[1:len(counts)])
    print(f'{area}')
    
    if gender=='Z':
        if num_of_bones==3:
            if area<5980:
                estimate=num_of_bones
            else:
                estimate=num_of_bones+4
        elif num_of_bones==4:
            if area<5000:
                estimate=num_of_bones
            else:
                estimate=num_of_bones+3
        elif num_of_bones==5:
            estimate=num_of_bones
        else:
            estimate=num_of_bones-2
    else:
        if num_of_bones<=3:
            estimate=num_of_bones
        elif num_of_bones==4:
            if area<11000:
                estimate=num_of_bones
            elif area>=11000 and area<28000:
                estimate=num_of_bones+2
            else:
                estimate=num_of_bones+5
        elif num_of_bones==5:
            estimate=num_of_bones
        elif num_of_bones==6:
            estimate=num_of_bones+4
        else:
            estimate=num_of_bones+1
    
    plt.subplot(1,2,2)
    plt.title(f'Procena: {estimate}g')
    
    fig.savefig(f'figure1_{k+1}')

    k=k+1
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    