import face_recognition as f_rec
import cv2,PIL,os,glob
import pickle
known_face_dir=r"faces/"
known_peeps=["montu/","chanchal/"]
unk_face_dir=r"raw_imgs/"
thresh=0.5
MODEL="cnn"
FRAME_THICKNESS = 2
FONT_THICKNESS = 1


def find_element(element, list_element):
    try:
        index_element = list_element.index(element)
        return index_element
    except ValueError:
        return None
def calculate_known_encoding(known_face_dir):
    known_faces=[]
    known_names=[]
    for i,img_path in enumerate( glob.glob(known_face_dir+"*.jpg")):
        image=f_rec.load_image_file(img_path)
        height, width, _ = image.shape
        face_location = (0, width, height, 0)
        encoding=f_rec.face_encodings(image, known_face_locations=[face_location],model=MODEL)[0]
        known_faces.append(encoding)
        known_names.append(img_path.split("/")[1].split("\\")[0])
    return known_faces,known_names
def compare_single_face(unk_face_dir):
    unk_faces=[]
    for i,img_path in enumerate( glob.glob(unk_face_dir+"*.jpg")):
        image=f_rec.load_image_file(img_path)
        height, width, _ = image.shape
        face_location = (0, width, height, 0)
        encoding=f_rec.face_encodings(image, known_face_locations=[face_location],model=MODEL)[0]
        comp_list=f_rec.compare_faces(known_faces,encoding,tolerance=thresh)
        print(find_element(True,comp_list))
        if(find_element(True,comp_list)!=None):
            img=cv2.imread(img_path)
            #img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imshow(known_names[0]+"!!!!!",img)
            cv2.waitKey(0)
        unk_faces.append(encoding)
# Returns (R, G, B) from name
def name_to_color(name):
    # Take 3 first letters, tolower()
    # lowercased character ord() value rage is 97 to 122, substract 97, multiply by 8
    color = [(ord(c.lower())-97)*8 for c in name[:3]]
    return color
def compare_multi_face(img_path,known_faces,known_names):
    image=f_rec.load_image_file(img_path)
    height, width, _ = image.shape
    locations=f_rec.face_locations(image)
    encodings=f_rec.face_encodings(image, locations)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for face_encoding, face_location in zip(encodings, locations):
        results = f_rec.compare_faces(known_faces, face_encoding, thresh)
        match=None
        if find_element(True,results)!=None:
            match = known_names[results.index(True)]
             # Each location contains positions in order: top, right, bottom, left
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])

            # Get color by name using our fancy function
            color = name_to_color(match)

            # Paint frame
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

            # Now we need smaller, filled grame below for a name
            # This time we use bottom in both corners - to start from bottom and move 50 pixels down
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)

            # Paint frame
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

            # Wite a name
            cv2.putText(image, match+"!!!!", (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)

    # Show image
    image = cv2.resize(image, (768,600), interpolation = cv2.INTER_AREA)
    cv2.imshow(img_path, image)
    cv2.waitKey(0)
    cv2.destroyWindow(img_path)

def compare_multi_face_dir(unk_face_dir,known_faces,known_names):
    unk_faces=[]
    for i,img_path in enumerate( glob.glob(unk_face_dir+"*.jpeg")):
        compare_multi_face(img_path,known_faces,known_names)


montu_face,montu_name=calculate_known_encoding(known_face_dir+known_peeps[0])
print(len(montu_face))
print("known face embedding calculated")
compare_multi_face_dir(unk_face_dir,montu_face, montu_name)

    