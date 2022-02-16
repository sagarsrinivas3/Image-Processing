import cv2
import os
import numpy as np

def getImageArrayColor(image):
  color = cv2.imread(image, 1) # 1 for color 0 for b&w
  return color

def getImageArrayGray(image):
  color = cv2.imread(image, 0) # 1 for color 0 for b&w
  return color

def createGreyImage(image, rename):
  color = cv2.imread(image, 0)
  cv2.imwrite(rename, color)
  
def getImagesList(dir):
  images = os.listdir(dir)
  return images

def convertMultipleImagesToGray(src, dest):
  images = getImagesList(src)
  for image in images:
    createGreyImage(f'{src}/{image}', f'{dest}/gray-{image}')
    #createGreyImage(src+"/"+image, dest+"/"+f'gray-{image}')
    pass

def getDimension(img):
  image = cv2.imread(img)
  return image.shape

def getImage(img):
  image = cv2.imread(img)
  return image

def getNewHeightAndWidth(scale_percentage, width, height):
  new_height = int(height * scale_percentage / 100)
  new_width = int(width * scale_percentage / 100)
  return (new_width, new_height)

def resize(img_path, scale_percent, resized_path):
  shape = getDimension(img_path)
  image = getImage(img_path)
  new_shape = getNewHeightAndWidth(scale_percent, shape[1], shape[0])
  resize_image = cv2.resize(image, new_shape)
  cv2.imwrite(resized_path, resize_image)

def DetectObjects(image_path, object_xml_path, newfilepath):
  image = getImageArrayColor(image_path)
  face_cascade = cv2.CascadeClassifier(object_xml_path)
  faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=4)
  # If faces len is 0 then faces are not detected
  COLOR = (255,255,255)
  WIDTH = 4
  for (x,y,w,h) in faces:
    cv2.rectangle(image, (x,y), (x+w, y+h), COLOR , WIDTH )
  if len(faces) > 0 :
    print(f"{len(faces)} Faces are not detected!!")
    cv2.imwrite(newfilepath, image)
    print("Output Image created!!")
  else:
    print("Faces are not detected!!")

def addWaterMark(image_path, watermark_path, outputfile_path):
  watermark = getImageArrayColor(watermark_path)
  image = getImageArrayColor(image_path)
  
  x = image.shape[1] - watermark.shape[1]
  y = image.shape[0] - watermark.shape[0]

  watermark_place = image[y:, x:]
  cv2.imwrite("watermark/watermark_place.jpeg", watermark_place)

  blend = cv2.addWeighted(src1=watermark_place, alpha=0.5, src2=watermark, beta=0.5, gamma=0)

  cv2.imwrite("watermark/blend.jpeg", blend)

  image[y:, x:] = blend
  cv2.imwrite(outputfile_path, image)

def removeGreenBackground(image_path, backimage_path):
  foreground = getImageArrayColor(image_path)
  background = getImageArrayColor(backimage_path)

  print(foreground[40,40]) # get the rgb values of green pixel
  
  width = foreground.shape[1]
  height = foreground.shape[0]
  
  for i in range(width):
    for j in range(height):
      pixel = foreground[j,i]
      pixel_list = list(pixel)

      #use resize_back for correct size of background and replace background with resize_back
      #resize_back = cv2.resize(background, (width, height ))
      
      if np.any(pixel == [54, 254,28]) | np.any(pixel == [53, 253,31]):
        foreground[j,i] = background[j,i]   # as prescribed
      if ( (pixel_list[0] < 90) & (pixel_list[0] > 55) & (pixel_list[1] < 280) & (pixel_list[1] > 250) & (pixel_list[2] < 55)):    # trail
        foreground[j,i] = background[j,i]
        

  cv2.imwrite("output.jpeg", foreground)
  print("file generated!!")
      
    
  
  
  
#case 1 : get image array
#print(getImageArray('galaxy.jpeg'))

#case 2 : convert to gray image
#createGreyImage("pictures/galaxy.jpeg", "gray-pictures/pictures/galaxy-gray.jpeg")

#case 3 :convert multipe images to gray
#convertMultipleImagesToGray("pictures", "gray-pictures")

#case 4 : Resize the image
#resize("pictures/galaxy.jpeg", 90, "resize-image.jpeg")

#case 5 : Detect Faces
#DetectObjects("faces.jpg", "config/faces.xml", "humanfaces.png")

#case 6 : Add watermark to image
#addWaterMark("watermark/faces.jpg", "watermark/sign.png", "watermark/imagewithwatermark.jpeg")

#case 7 : Remove Background
#removeGreenBackground("background/foreground.jpg", "background/background.jpeg")


