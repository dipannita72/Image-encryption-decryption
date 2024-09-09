# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 02:28:50 2023

@author: USER
"""

import cv2
import numpy as np
import matplotlib as plt
import random

img = cv2.imread("Lena.jpeg")
cv2.imshow("rgb",img)

img_con = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
cv2.imshow("YCrCb", img_con)

#--------------------------------divide in blocks
tilesize = (8,8)#(128,128)
def reshape_split(image: np.ndarray, kernel_size: tuple):

    img_height, img_width, channels = image.shape
    tile_height, tile_width = kernel_size

    tiled_array = image.reshape(img_height // tile_height,
                                tile_height,
                                img_width // tile_width,
                                tile_width,
                                channels)
    tiled_array = tiled_array.swapaxes(1, 2)
    return tiled_array
#tiles shape:(4, 4, 128, 128, 3) :::4*4 tiles
tiles = reshape_split(img_con, tilesize)

#----------------------------------shuffle----

def shuffle(image, key):
    
    temp = np.zeros((image.shape[0],image.shape[1],image.shape[2],image.shape[3],image.shape[4]),np.uint8)
    temp2 = np.zeros((image.shape[0],image.shape[1],image.shape[2],image.shape[3],image.shape[4]),np.uint8)
    no_tiles = image.shape[0] *image.shape[1]
    list_ = list(range(0, image.shape[0]))
    random.seed(key)
    random.shuffle(list_)
    #print(image.shape)   
    #print(temp.shape)
    #print(list_)
    random.seed()
    for j in range(len(list_)) :
        temp[j,:,:,:,:]=image[list_[j],:,:,:,:]   
    for j in range(len(list_)) :
        temp2[:,j,:,:,:]=temp[:,list_[j],:,:,:]
    
    return temp2


key1=3
shuffled_img = shuffle(tiles, key1)
print(shuffled_img.shape)
shuffled_img_view = shuffled_img.swapaxes(1, 2)
tiles_merge_s = np.reshape(shuffled_img_view,(512,512,3))
cv2.imshow("SHUFFLE",tiles_merge_s)


#------------------rotation----------

def rotate(image):
    
    key2 = 90
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return rotated_image

rotated_img_view = shuffled_img.swapaxes(1, 2)
rotated_img_view = np.reshape(rotated_img_view,(512,512,3))
rotated_img = rotate(rotated_img_view)
#--------------inversion

key3 = -1
flipped_image = cv2.flip(rotated_img , key3)

cv2.imshow("ROTATE",rotated_img)
cv2.imshow("INVERSION",flipped_image)



#-----------------------color shuffle

channel1= flipped_image[:,:,0]
channel2= flipped_image[:,:,1]
channel3= flipped_image[:,:,2]

# Merge the arrays horizontally
merged_image = np.concatenate((channel1, channel2, channel3), axis=1)
print(merged_image.shape)
#----------------color shuffling-----
def color_shuffle(image,key):
    img_height, img_width = image.shape
    tile_height, tile_width = tilesize

    tiled_array = image.reshape(img_height // tile_height,
                                tile_height,
                                img_width // tile_width,
                                tile_width)
    image = tiled_array.swapaxes(1, 2)
    print("k")
    print(tiled_array.shape)
    
    temp = np.zeros((image.shape[0],image.shape[1],image.shape[2],image.shape[3]),np.uint8)
    temp2 = np.zeros((image.shape[0],image.shape[1],image.shape[2],image.shape[3]),np.uint8)
    
    list_ = list(range(0, image.shape[1]))
    random.seed(key)
    random.shuffle(list_)
    #print(image.shape)   
    print(temp.shape)
    print(len(list_))
    random.seed()
     
    for j in range(len(list_)) :
        temp[:,j,:,:]=image[:,list_[j],:,:]
    
    return temp
cv2.imshow("ENCRYPTED IMAGE before shuffle",merged_image)
key4=3
encrypted_img = color_shuffle(merged_image, key4)
encrypted_img = encrypted_img.swapaxes(1, 2)
encrypted_img = np.reshape(encrypted_img,merged_image.shape)
cv2.imshow("FINAL ENCRYPTED IMAGE",encrypted_img)  
print(encrypted_img.shape)
#-------------Compression--------------------------
def compress_image(image,quality = 50):
    compressed_path = "G:\\1.1-4.2STUDY\\Masters\\network security-assignment\\compressed.jpeg"
    cv2.imwrite(compressed_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])


compress_image(encrypted_img)


#-------------Decompression
def decompress_image():
    compressed_path = "G:\\1.1-4.2STUDY\\Masters\\network security-assignment\\compressed.jpeg"
    decompressed_path = "G:\\1.1-4.2STUDY\\Masters\\network security-assignment\\decompressed.jpeg"
    
    compressed_image = cv2.imread(compressed_path)
    cv2.imwrite(decompressed_path, compressed_image)
    
# Decompress the image
decompress_image()

# Display the decompressed image
decompressed_image = cv2.imread("decompressed.jpeg",0)
cv2.imshow("Decompressed Image", decompressed_image)

#--------------------------------------DECRYPTION--------------------------------------
#------------------------------------------------------------------------------------


def color_shuffle_reverse(image,key):
    img_height, img_width = image.shape
    tile_height, tile_width = tilesize

    tiled_array = image.reshape(img_height // tile_height,
                                tile_height,
                                img_width // tile_width,
                                tile_width)
    image = tiled_array.swapaxes(1, 2)
    print("k")
    print(tiled_array.shape)
    
    temp = np.zeros((image.shape[0],image.shape[1],image.shape[2],image.shape[3]),np.uint8)
    temp2 = np.zeros((image.shape[0],image.shape[1],image.shape[2],image.shape[3]),np.uint8)
    
    list_ = list(range(0, image.shape[1]))
    random.seed(key)
    random.shuffle(list_)
    #print(image.shape)   
    print(temp.shape)
    print(len(list_))
    random.seed()
     
    for j in range(len(list_)) :
        temp[:,list_[j],:,:]=image[:,j,:,:]
    
    return temp

decrypted_img = color_shuffle_reverse(decompressed_image, key4)
decrypted_img = decrypted_img.swapaxes(1, 2)
decrypted_img = np.reshape(decrypted_img,merged_image.shape)

#----------------- seperate channels-------------------

channel_img = np.zeros((512,512,3),np.uint8)
channel_img[:,:,0] = decrypted_img[:,0:1*512]
channel_img[:,:,1] = decrypted_img[:,1*512:2*512]
channel_img[:,:,2] = decrypted_img[:,2*512:3*512]
print()

cv2.imshow("channelED BACK",channel_img)

#--------------------invsersion
flipped_image_back = cv2.flip(channel_img , key3)
#---------------------rotation
rotated_image_back = cv2.rotate(flipped_image_back, cv2.ROTATE_90_COUNTERCLOCKWISE)
cv2.imshow("Rotate Back",rotated_image_back)


#---------------------unshuffle---------
def shuffle_reverse(shuffled_img,key):
    
    unshuffled_img = np.zeros((shuffled_img.shape[0],shuffled_img.shape[1],shuffled_img.shape[2],shuffled_img.shape[3],shuffled_img.shape[4]),np.uint8)
    unshuffled_img2 = np.zeros((shuffled_img.shape[0],shuffled_img.shape[1],shuffled_img.shape[2],shuffled_img.shape[3],shuffled_img.shape[4]),np.uint8)
    list_ = list(range(0, shuffled_img.shape[0]))
    random.seed(key)
    random.shuffle(list_)
    #print(len(list_))
    #print(list_)
    random.seed()
    for j in range(len(list_)):
        #print(j,list_[j])
        unshuffled_img[list_[j], :, :, :] = shuffled_img[j, :, :, :, :]
    
    for j in range(len(list_)):
        #print(j,list_[j])
        unshuffled_img2[:, list_[j], :, :, :] = unshuffled_img[:, j, :, :, :]
    
    return unshuffled_img2

tile_h,tile_w = tilesize
h=rotated_image_back.shape[0]
w=rotated_image_back.shape[1]
blocked_img_shuffled_back = np.reshape(rotated_image_back, (h//tile_h,tile_h,w//tile_w,tile_w,3))
blocked_img_shuffled_back= blocked_img_shuffled_back.swapaxes(1, 2)
#print(blocked_img_shuffled_back.shape)
unshuffle_img_back = shuffle_reverse(blocked_img_shuffled_back, key1)
#print(unshuffle_img_back.shape)

unshuffle_img_back_view = unshuffle_img_back.swapaxes(1, 2)
unshuffle_img_back_view = np.reshape(unshuffle_img_back_view,(512,512,3))
cv2.imshow("unshuffle",unshuffle_img_back_view)



#---------------tiles to image-------------------

#tiles_ = tiles.swapaxes(1, 2)
#tiles_merge = np.reshape(tiles_,(512,512,3))

#---------------------------------
im_rgb = cv2.cvtColor(unshuffle_img_back_view, cv2.COLOR_YCrCb2BGR)
cv2.imshow("DEcrypted RGB",im_rgb)

#--------------------------
diff = cv2.absdiff(img, im_rgb)
cv2.imwrite("output_diff.jpg", diff)
cv2.imshow("difference",diff)

rms = np.sqrt(np.mean(np.square(diff)))
print("Root square error:",rms)
cv2.waitKey(0)
cv2.destroyAllWindows()