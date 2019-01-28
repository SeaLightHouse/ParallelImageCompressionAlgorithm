import findspark
findspark.init()
import pyspark
from pyspark import SparkContext
import cv2
import numpy as np 
import scipy as sp
import struct
from helper_functions import *
from constants import *



###my helper functions
##   convert from rgb to YCrCb
def my_bgr2ycrcb(arr):
    Y,Crf,Cbf=convert_to_YCrCb(arr[1])
    h,w=Y.shape
    #resize the image : 
    Crf= resize_image(Crf, w,h) 
    Cbf= resize_image(Cbf,w,h)
    return ((arr[0],'channel1'),Y) ,((arr[0],'channel2'),Crf) ,((arr[0],'channel3'),Cbf)


## split the three 2D arrays Y,Cr,Cb to 8x8 tiles and build
# an rdd for the tiles
#horizontal direction  first
def my_create_block(arr):
    M=arr[1]
    h,w=M.shape
    B=[]
    n=0  # n is block no
    for p in range(0,h,8):
        for q in range(0,w,8):
           n+=1
           B.append( ((arr[0][0],arr[0][1],h,w),(n, M[p:p+8,q:q+8])))
    return B


# wrapper function over the helper function
def my_quantize_block(arr):
    f=False
    if arr[0][1] == 'channel1' :
       f=True
    return arr[0],(arr[1][0],quantize_block(arr[1][1],f))

# wrapper function over the helper function
def my_inverse_quantize_block(arr):
    f=False
    if arr[0][1] == 'channel1' :
       f=True
    return arr[0],(arr[1][0],quantize_block(arr[1][1],f,99,True))


# combine the 8x8 tiles to again form a HxW array
def my_combine_block(arr):
    blk=[i[1]  for i in sorted(arr[1])]
    h=arr[0][2]
    w=arr[0][3]
    M=np.zeros((h,w),np.uint8)
    n=0
    for p in range(0,h,8):
        for q in range(0,w,8):
            M[p:p+8,q:q+8]=blk[n]
            n+=1
    return arr[0], M


# combine the Y,Cr,Cb 2d arrays to form a 3d array
def my_create_3d_image(arr):
    # sort by channel , channe1, channel2 , channel3
    r= [i[1]  for i in sorted(arr[1])]
    img= np.dstack([r[0],r[1],r[2]])
    return arr[0], img



### WRITE ALL HELPER FUNCTIONS ABOVE THIS LINE ###

def generate_Y_cb_cr_matrices(rdd):
    """
    THIS FUNCTION MUST RETURN AN RDD
    """
    ### BEGIN SOLUTION ###
    rdd=rdd.flatMap(my_bgr2ycrcb)  
    return rdd

def generate_sub_blocks(rdd):
    """
    THIS FUNCTION MUST RETURN AN RDD
    """
    ### BEGIN SOLUTION ###
    rdd=rdd.flatMap(my_create_block)
    return rdd

def apply_transformations(rdd):
    """
    THIS FUNCTION MUST RETURN AN RDD
    """
    ### BEGIN SOLUTION ###
                
    rdd=rdd.map(lambda x : (x[0],(x[1][0],dct_block((x[1][1]+128).astype(np.int8),True))))
    #note above since the data type is uint8 , so (x + 128).astype(np.int8) is actually x - 128    

    rdd=rdd.map(my_quantize_block)

    rdd=rdd.map(my_inverse_quantize_block)

    rdd=rdd.map(lambda x : (x[0],(x[1][0],(dct_block(x[1][1],False) + 128).astype(np.uint8))))
    return rdd


def combine_sub_blocks(rdd):
    """
    Given an rdd of subblocks from many different images, combine them together to reform the images.
    Should your rdd should contain values that are np arrays of size (height, width).

    THIS FUNCTION MUST RETURN AN RDD
    """
    ### BEGIN SOLUTION ###
    rdd=rdd.groupByKey().map(my_combine_block)
    return rdd

def run(images):
    """
    THIS FUNCTION MUST RETURN AN RDD

    Returns an RDD where all the images will be proccessed once the RDD is aggregated.
    The format returned in the RDD should be (image_id, image_matrix) where image_matrix 
    is an np array of size (height, width, 3).
    """
    sc = SparkContext()
    rdd = sc.parallelize(images, 16) \
        .map(truncate).repartition(16)
    rdd = generate_Y_cb_cr_matrices(rdd)
    rdd = generate_sub_blocks(rdd)
    rdd = apply_transformations(rdd)
    rdd = combine_sub_blocks(rdd)

    ### BEGIN SOLUTION HERE ###
    # Add any other necessary functions you would like to perform on the rdd here
    # Feel free to write as many helper functions as necessary
    #create the 3 d matrix
    #first group the 3 channels into a tuple for each image
    rdd=rdd.sortByKey().map(lambda x : (x[0][0], (x[0][1],x[1])))
    #then combine the tuple into a 3d array 
    rdd=rdd.groupByKey().map(my_create_3d_image)
    
    #convert to rbg
    rdd=rdd.map(lambda x : (x[0] , to_rgb(x[1])))
    return  rdd
