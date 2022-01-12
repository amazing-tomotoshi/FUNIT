import cv2
from matplotlib import pyplot as plt
from retinaface.pre_trained_models import get_model
from retinaface.utils import vis_annotations
import subprocess
from PIL import Image

model = get_model("resnet50_2020-07-20", max_size=2048)
model.eval()

path = 'images/input.jpg'
image = cv2.imread(path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

results = model.predict_jsons(image)

bboxes = []

height, width, ch = image.shape

def draw_res(image,results):
    for r in results:
        bbox = r['bbox']
        if not bbox:continue
        cv2.rectangle(image,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),thickness=10)
    return image

def set_bboxes(image,results,bboxes,height, width):
    for r in results:
        bbox = r['bbox']
        if not bbox:continue
        left = int(bbox[0] - (bbox[2] - bbox[0]) * 0.5)
        if left < 0 :
            left = 0
        right = int(bbox[2] + (bbox[2] - bbox[0]) * 0.5)
        if right > width :
            right = width
        top = int(bbox[1] - (bbox[3] - bbox[1]) * 0.3)
        if top < 0 :
            top = 0
        bottom = int(bbox[3] + (bbox[3] - bbox[1]) * 0.3)
        if bottom > height :
            bottom = height
        bboxes.append([left, top, right, bottom])
    return image

set_bboxes(image,results,bboxes,height, width);
print(bboxes)
for i,bbox in enumerate(bboxes):
    image_face = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    face_path = "images/face/face-{}.jpg".format(i)
    cv2.imwrite(face_path, image_face)
    command = "python3 test_k_shot.py --config configs/funit_animals.yaml --ckpt pretrained/animal149_gen.pt --input {} --class_image_folder {} --output images/output-{}.jpg".format(face_path, 'images/n02086646', i)
    subprocess.run(command, shell=True)

image1 = Image.open(path)
back_image = image1.copy()

for i,bbox in enumerate(bboxes):
    face_path = "images/output-{}.jpg".format(i)
    image2 = Image.open(face_path)
    image2_resize = image2.resize((bbox[2]-bbox[0], bbox[3]-bbox[1]))
    back_image.paste(image2_resize, (bbox[0], bbox[1]))

back_image.save('images/output.jpg', quality=95)

cv2.imshow('output', back_image)
cv2.waitKey(0)
cv2.destroyAllWindows()