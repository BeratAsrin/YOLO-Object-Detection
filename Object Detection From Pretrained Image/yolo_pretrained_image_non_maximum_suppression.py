import cv2 as cv
import numpy as np

img = cv.imread("Object Detection From Pretrained Image\contents\\traffic-608.jpg")

img_width = img.shape[0]
img_height = img.shape[1]

img_blob = cv.dnn.blobFromImage(img, 1/255, (608,608), swapRB=True, crop=False)

labels = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
        "trafficlight","firehydrant","stopsign","parkingmeter","bench","bird","cat",
        "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
        "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sportsball",
        "kite","baseballbat","baseballglove","skateboard","surfboard","tennisracket",
        "bottle","wineglass","cup","fork","knife","spoon","bowl","banana","apple",
        "sandwich","orange","broccoli","carrot","hotdog","pizza","donut","cake","chair",
        "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
        "remote","keyboard","cellphone","microwave","oven","toaster","sink","refrigerator",
        "book","clock","vase","scissors","teddybear","hairdrier","toothbrush"]

colors = ["0,255,255", "0,0,255", "0,255,0", "255,0,0", "255,255,0"]
colors = [np.array(color.split(",")).astype("int") for color in colors]
colors = np.array(colors)
colors = np.tile(colors, (18, 1))

model = cv.dnn.readNetFromDarknet("Pretrained Model\yolov3.cfg", "Pretrained Model\yolov3.weights")
layers = model.getLayerNames()
output_layer = [layers[layer[0]-1] for layer in model.getUnconnectedOutLayers()]

model.setInput(img_blob)

detection_layer = model.forward(output_layer)

ids_list = []
boxes_list = []
confidences_list = []

for detection in detection_layer:
    for object_detection in detection:
        
        scores = object_detection[5:] 
        predicted_id = np.argmax(scores)
        confidence = scores[predicted_id]

        label = labels[predicted_id]
        bounding_box = object_detection[0:4] * np.array([img_width, img_height, img_width, img_height])
        (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int")
        
        start_x = int(box_center_x - (box_width/2))
        start_y = int(box_center_y - (box_height/2))
        
        ids_list.append(predicted_id)
        confidences_list.append(float(confidence))
        boxes_list.append([start_x, start_y, int(box_width), int(box_height)])

            

max_ids = cv.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)

for max_id in max_ids:
    
    max_class_id = max_id[0]
    box = boxes_list[max_class_id]

    start_x = box[0]
    start_y = box[1]
    box_width = box[2]
    box_height = box[3]    

    predicted_id = ids_list[max_class_id]
    label = labels[predicted_id]
    confidence = confidences_list[max_class_id]

    end_x = start_x + box_width
    end_y = start_y + box_height

    box_color = colors[predicted_id]
    box_color = [int(each) for each in box_color]

    label = "{} :{:.1f}%".format(label, confidence*100)

    cv.rectangle(img, (start_x, start_y), (end_x, end_y), box_color, 1)
    cv.putText(img, label, (start_x, start_y-10), cv.FONT_HERSHEY_PLAIN, 1, box_color, 1)

cv.imshow("Detection Frame", img)

cv.waitKey(0)
cv.destroyAllWindows()
