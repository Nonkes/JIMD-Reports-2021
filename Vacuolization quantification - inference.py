"""
Code to quantify lymphocyte vacuolization in peripheral blood smears, as described in Nonkes et al. 2021 (JIMD Reports).
(see https://doi.org/10.1002/jmd2.12191)

Copyright (c) 2021 
Licensed under the MIT License (see LICENSE for details)
Written by Lourens Nonkes
"""

# import general dependencies
import numpy as np
import pandas as pd

# parameter declaration
smooth = 1.

# declaration of directories
dir_src = ' put in filepath ' # directory with lymphocyte images to quantify (should be created beforehand)
dir_exp = ' put in filepath ' # output directory for masked images (should be created beforehand)
dir_maskCheck = ' put in filepath ' # output directory for images of interest with overlayed mask -> for checking accuracy of produced masks (should be created beforehand)


########################
# START OF FUNCTIONS - #
########################
# made it kind of modular, easy to remove/add stuff

# function to obtain filelist
def fileList(dirOI):
    import os
    
    matches = []
    for root, dirnames, filenames in os.walk(dirOI):
        for filename in filenames:
            if filename.endswith(('.jpg','.png')):
                matches.append(os.path.join(filename))
    return matches


# Function to load image files of interest and return those as a numpy array
def read_images(file_ids):
    import numpy as np
    from skimage.transform import resize
    from keras.utils import Progbar 
    from skimage.io import imread
            
    CNN_list = np.zeros((len(file_ids), 128, 128, 3), dtype=np.uint8)
    mask_list = np.zeros((len(file_ids), 358, 352, 3), dtype=np.uint8)
    im_sizes = []
    print('\nGetting and resizing images ... ')
    
    b = Progbar(len(file_ids))
    for n, id_ in enumerate(file_ids):
        path = dir_src + id_
        img = imread(path)[:,:,:3]
        im_sizes.append([img.shape[0], img.shape[1]])
        
        img_net = resize(img, (128, 128), mode='constant', preserve_range=True)
        CNN_list[n] = img_net # holds images for CNN
        
        img_mask = resize(img, (358, 352), mode='constant', preserve_range=True)
        mask_list[n] = img_mask # holds images for CNN
        b.update(n)
        
    return CNN_list, im_sizes, mask_list


# Metric function
def dice_coef(y_true, y_pred):
    from keras import backend as K
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# Loss funtion
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# function to obtain CELL masks
def getCellMasks(image_list, img_sizes, min_pixsize):
    from keras.models import load_model
    import numpy as np
    from skimage.transform import resize
    import cv2
    
    threshold = 200 # threshold used in masking procedure
    kernel = np.ones((5,5), np.uint8) # kernel for erosion operation
    kernelD = np.ones((3,3), np.uint8) # kernel for dilation operation
    
    # load model
    u_net = load_model('model-segmentationC.h5', custom_objects={'dice_coef': dice_coef})
    
    print("\n")
    print("\nPredicting cell mask boundaries ...")
    U_masks = u_net.predict(image_list,verbose=1)
    
    Umasks_resized = []
    export_masks = []
    for i, mask in enumerate(U_masks,0):
        Umasks_resized.append(resize(np.squeeze(mask), 
                                           (img_sizes[i][0],img_sizes[i][1]), 
                                           mode='constant', preserve_range=True))
        
        Umasks_resized[i] = np.array(Umasks_resized[i] * 255, dtype = np.uint8)
        
        # set threshold
        Umasks_resized[i][Umasks_resized[i] >= threshold] = 255
        Umasks_resized[i][Umasks_resized[i] < threshold] = 0
        
        # find all connected components (objects in image)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(Umasks_resized[i], connectivity=8)
        # taking out the background which is also considered a component
        sizes = stats[1:, -1]; nb_components = nb_components - 1
    
        export_masks.append(np.zeros(output.shape, dtype = np.uint8))
        # one round of erosion and dilation to smooth the mask a bit
        export_masks[i] = cv2.erode(export_masks[i], kernel, iterations=1)
        export_masks[i] = cv2.dilate(export_masks[i], kernelD, iterations=1)
     
        # for every component in the image, keep only those above min_size, put in export_masks
        for j in range(0, nb_components):
            if sizes[j] >= min_pixsize:
                export_masks[i][output == j + 1] = 255       

    return export_masks


# function to obtain NUCLEUS masks
def getNucleusMasks(image_list, img_sizes, min_pixsize):
    from keras.models import load_model
    import numpy as np
    from skimage.transform import resize
    import cv2
    
    threshold = 200 # threshold used in masking procedure
    kernelD = np.ones((3,3), np.uint8) # kernel for dilation operation
    
    # load model
    u_net = load_model('model-segmentationN.h5', custom_objects={'dice_coef': dice_coef})
    
    print("\n")
    print("\nPredicting nucleus mask boundaries ...")
    U_masks = u_net.predict(image_list,verbose=1)
    
    Umasks_resized = []
    export_masks = []
    for i, mask in enumerate(U_masks,0):
        Umasks_resized.append(resize(np.squeeze(mask), 
                                           (img_sizes[i][0],img_sizes[i][1]), 
                                           mode='constant', preserve_range=True))
        
        Umasks_resized[i] = np.array(Umasks_resized[i] * 255, dtype = np.uint8)
        
        # set threshold
        Umasks_resized[i][Umasks_resized[i] >= threshold] = 255
        Umasks_resized[i][Umasks_resized[i] < threshold] = 0
        
        # find all connected components (white blobs in image)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(Umasks_resized[i], connectivity=8)
        # taking out the background which is also considered a component
        sizes = stats[1:, -1]; nb_components = nb_components - 1
    
        export_masks.append(np.zeros(output.shape, dtype = np.uint8))
        
        # for every component in the image, keep only those above min_size
        for j in range(0, nb_components):
            if sizes[j] >= min_pixsize:
                export_masks[i][output == j + 1] = 255
        
        export_masks[i] = cv2.dilate(export_masks[i], kernelD, iterations=1)
        
    return export_masks


# Function to obtain CYTOPLASM masks
def getCytoMasks(cellmasks, nucleusmasks, min_pixsize):
    import cv2
    import numpy as np
    
    print("\n")
    print("\nObtaining cytoplasm masks ...")
    
    cytomasks = []
    for i in range(len(cellmasks)):
        cytomasks.append(cellmasks[i]-nucleusmasks[i])
                
        #find all connected components (objects in image)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(cytomasks[i], connectivity=8)
        #taking out the background which is also considered a component
        sizes = stats[1:, -1]; nb_components = nb_components - 1
    
        # minimum size of particles (no. of pixels)
        min_size = min_pixsize
    
        cytomasks[i] = (np.zeros(output.shape, dtype = np.uint8))
        
        # for every component in the image, keep only those above min_size, put in cytomasks
        for j in range(0, nb_components):
            if sizes[j] >= min_size:
                cytomasks[i][output == j + 1] = 255
    
    return cytomasks 


# function to obtain masked images
def getExportMasks(dir_exp, test_img,cellmasks3,nucleusmasks3):
    import os, shutil
    import numpy as np
    from skimage.io import imsave
    
    shutil.rmtree(dir_exp) # delete previous content 
    i = 0;
    maskedImages = []
    print("\nProducing masked images ...")
    os.mkdir(dir_exp) # recreate dir_heatmap
    
    for element in test_img: 
        #print("\nProcessing image: ", )
        
        m = element[:,:,:].astype(np.float32)
        mmask = cellmasks3[i].astype(np.float32)
        m[mmask == 0] = np.nan
        m = np.uint8(m)
        maskedImages.append(m)
        
        # Save to a File
        #filename = ''.join(["image ", str(i)])
        imsave(dir_exp + file_ids[i], m)
        i = i +1
        
    return maskedImages


# function to obtain dataframe for easy output to excel
def makeDataframe(file_ids, prediction):
    import pandas as pd
    
    df = pd.DataFrame(file_ids, columns = ['fileID'])
    df['normal%'] = prediction[:,0]*100
    df['vacs%'] = prediction[:,1]*100
               
    return df


# function to show images of masks as overlay on top of original images
def exportMasks(dir_maskCheck, test_img, nucleusmasks, cytomasks, cellmasks):
    import os, shutil
    import matplotlib.pyplot as plt
    from skimage.io import imshow
    
    shutil.rmtree(dir_maskCheck) # delete previous content 
    print("\nProducing overlay images ...")
    os.mkdir(dir_maskCheck) # recreate directory
     
    for i in range(len(test_img)):
        #plt.figure()
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(test_img[i])
        axarr[0].imshow(cellmasks[i], alpha=0.3)
        axarr[1].imshow(test_img[i])
        axarr[1].imshow(nucleusmasks[i], alpha=0.3)
        plt.savefig(dir_maskCheck + file_ids[i], format='png', dpi=300)
        plt.close()


# function to predict vacuolisation
def neuralpredict_vacs(dir_src):
    import os
    from keras.models import load_model
    import numpy as np
    from PIL import Image
    from keras.applications.resnet50 import preprocess_input
    
    print("\npredicting vacuolisation ...")
    img_size = 224 

    # load model
    model = load_model('model-quantification-21-0.96.hdf5')    
    
    def fileList():
        matches = []
        for root, dirnames, filenames in os.walk(dir_src):
            for filename in filenames:
                if filename.endswith(('.jpg')):
                    matches.append(dir_src + os.path.join(filename))
        return matches
    
    img_paths = fileList()
    img_list = [Image.open(img_path) for img_path in img_paths]
    
    batch = np.stack([preprocess_input(np.array(img.resize((img_size, img_size))))
                                 for img in img_list])
     
    pred_probs = model.predict(batch)       
    
    return pred_probs

    
####################
# END OF FUNCTIONS #
####################    
    
 
# get filelist
file_ids = fileList(dir_src)

# get image_data
Unet_list,img_sizes,im_list = read_images(file_ids)
       
# obtain masks
cellmasks = getCellMasks(Unet_list, img_sizes, 2000) 
nucleusmasks = getNucleusMasks(Unet_list, img_sizes, 1000) 
cytomasks = getCytoMasks(cellmasks, nucleusmasks, 10)

# export masks & mask overlay images to imagefiles
exportMasks(dir_maskCheck, im_list, nucleusmasks, cytomasks, cellmasks) # export images with mask as layer projected on image
maskedImages = getExportMasks(dir_exp,im_list,cytomasks,nucleusmasks) # export actual masked images

#get neural net prediction on vacuolisation
prediction = neuralpredict_vacs(dir_exp)

# put predictions of all images in pandas df and export as excelfile
df = makeDataframe(file_ids, prediction) 
df.to_excel('df_vacsQ.xlsx')

# df_vacs dataframe contains predicted no. lymphocytes >65% vacuolization (threshold level), export to excelfile
index = [0]
df_vacs = pd.DataFrame([], index=index, columns=['prediction: '])
df_vacs['>65%'] = sum(df['vacs%']>65)  
# export as excelfile
df_vacs.to_excel('df_vacsQ 65.xlsx')
    
