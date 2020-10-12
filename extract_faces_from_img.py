from PIL import Image
import face_recognition
import cv2,os,glob
img_dir="raw_img/"

for img in glob.glob(img_dir+"*.jpg"):
    im_name=img.split("\\")[1].split(".")[0]
    image = face_recognition.load_image_file(img)
    
    
    # Find all the faces in the image using the default HOG-based model.
    # This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
    # See also: find_faces_in_picture_cnn.py
    face_locations = face_recognition.face_locations(image,number_of_times_to_upsample=0, model="cnn")
    
    print("I found {} face(s) in this photograph.".format(len(face_locations)))
    
    for i, face_location in enumerate(face_locations):
    
        # Print the location of each face in this image
        top, right, bottom, left = face_location
        
        print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
    
        # You can access the actual face itself like this:
        face_image = image[top:bottom, left:right]
        
        f_im2=cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        #pil_image = Image.fromarray(face_image)
        #pil_image.show()
        cv2.imwrite("faces/ex_face{}-{}.jpg".format(i,im_name),f_im2)
    
