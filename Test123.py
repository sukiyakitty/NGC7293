import numpy as np
import cv2
from Lib_Function import any_to_image, image_to_gray, image_treatment_toGray, image_retreatment_toGray, trans_blur

if __name__ == '__main__':
    k = 1

    img_original = image_to_gray(r'C:\Users\Kitty\Desktop\test\Result3_CD13_S0077(G6-G6)_C00_M0012_ORG.jpg')
    cv2.imshow('Source Image', img_original)

    img_blur = cv2.GaussianBlur(img_original, ksize=(13, 13), sigmaX=0, sigmaY=0)
    cv2.imshow('blur Image', img_blur)

    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(16, 16))
    img_CLAHE = clahe.apply(img_original)
    cv2.imshow('CLAHE Enhanceed Image', img_CLAHE)

    img_UM = np.uint8(img_original + k * (img_original - img_blur))
    cv2.imshow('Unsharp Masking Image', img_UM)

    img_CLAHE_blur = cv2.GaussianBlur(img_CLAHE, ksize=(13, 13), sigmaX=0, sigmaY=0)
    cv2.imshow('Unsharp Masking CLAHE Image', np.uint8(img_CLAHE + k * (img_CLAHE - img_CLAHE_blur)))

    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(16, 16))
    cv2.imshow('CLAHE Unsharp Masking Image', clahe.apply(img_UM))

    cv2.waitKey()

# import tensorflow as tf
#
# # tf.reset_default_graph()
#
# var1 = tf.Variable(1.0 , name='firstvar')
# print ("var1:",var1.name)
# var1 = tf.Variable(2.0 , name='firstvar')
# print ("var1:",var1.name)
# var2 = tf.Variable(3.0 )
# print ("var2:",var2.name)
# var2 = tf.Variable(4.0 )
# print ("var1:",var2.name)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print("var1=",var1.eval())
#     print("var2=",var2.eval())
#
#
#
# get_var1 = tf.get_variable("firstvar",[1], initializer=tf.constant_initializer(0.3))
# print ("get_var1:",get_var1.name)
#
# #get_var1 = tf.get_variable("firstvar",[1], initializer=tf.constant_initializer(0.4))
# #print ("get_var1:",get_var1.name)
#
# get_var1 = tf.get_variable("firstvar1",[1], initializer=tf.constant_initializer(0.4))
# print ("get_var1:",get_var1.name)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print("get_var1=",get_var1.eval())

    img_UM = np.uint8(img_original + k * (img_original - img_blur))
    cv2.imshow('Unsharp Masking Image', img_UM)

# largest = None
# smallest = None
# while True:
#     num = input()
#
#     if num == "done":
#         break
#
#     try:
#         value = int(num)
#     except:
#         print("Invalid input")
#         continue
#
#     if largest is None:
#         largest = value
#     elif value > largest:
#         largest = value
#
#     if smallest is None:
#         smallest = value
#     elif value < smallest:
#         smallest = value
#
# print("Maximum is ", largest)
# print("Minimum is ", smallest)
