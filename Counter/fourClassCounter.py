import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("../Videos/k1.mp4")  # For Video
iconSize=(50,50)

model = YOLO("../YoloWeights/best1.pt")

classNames =  ['Ambulance', 'AutoRikshaw', 'Bicycle', 'Bus', 'Car', 'Motorcycle', 'Truck']
# classNames = ["person", "Bicycle", "Car", "Motorcycle", "aeroplane", "Bus", "train", "Truck", "boat",
#               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
#               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
#               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
#               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
#               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
#               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
#               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
#               "teddy bear", "hair drier", "toothbrush"
# ]

mask = cv2.imread("maskK1.png")

# Tracking
trackerCar = Sort(max_age=20, min_hits=0, iou_threshold=0.5)
trackerTruck = Sort(max_age=20, min_hits=0, iou_threshold=0.5)
trackerBus = Sort(max_age=20, min_hits=0, iou_threshold=0.5)
trackerMotorcycle = Sort(max_age=20, min_hits=0, iou_threshold=0.5)
trackerBicycle = Sort(max_age=20, min_hits=0, iou_threshold=0.5)
trackerAutorikshaw = Sort(max_age=20, min_hits=0, iou_threshold=0.5)
trackerAmbulance = Sort(max_age=20, min_hits=0, iou_threshold=0.5)


limits = [343, 637, 872, 637] #for k1
# limits = [531, 426, 904, 637]
# limits = [400, 297, 673, 297]
carCount = []
busCount = []
truckCount = []
motorcycleCount = []
bicycleCount = []
autorikshawCount = []
ambulanceCount = []


while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)

    # imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    # img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
    # results = model.predict(imgRegion, stream=True, classes=(2,3,5,7))
    results = model.predict(imgRegion, stream=True)

    carDetections = np.empty((0, 5))
    truckDetections = np.empty((0, 5))
    busDetections = np.empty((0, 5))
    motorcycleDetections = np.empty((0, 5))
    bicycleDetections = np.empty((0, 5))
    autorikshawDetections = np.empty((0, 5))
    ambulanceDetections = np.empty((0, 5))

    for r in results:
        print('Bounding')
        boxes = r.boxes
        for box in boxes:
            print('BOX')
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            print(box.cls)
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            print("Class  --  ",currentClass)
            if currentClass == "Autorikshaw" and conf > 0.3:
                print('Autorikshaw')
                currentArray = np.array([x1, y1, x2, y2, conf])
                autorikshawDetections = np.vstack((autorikshawDetections, currentArray))
            elif currentClass == "Bus" and conf > 0.3:
                print('Bus')
                currentArray = np.array([x1, y1, x2, y2, conf])
                busDetections = np.vstack((busDetections, currentArray))
            elif currentClass == "Motorcycle" and conf > 0.3:
                print('Motorcycle')
                currentArray = np.array([x1, y1, x2, y2, conf])
                motorcycleDetections = np.vstack((motorcycleDetections, currentArray))
            elif currentClass == "Bicycle" and conf > 0.3:
                print('Bicycle')
                currentArray = np.array([x1, y1, x2, y2, conf])
                bicycleDetections = np.vstack((bicycleDetections, currentArray))
            elif currentClass == "Car" and conf > 0.3:
                print('Car')
                currentArray = np.array([x1, y1, x2, y2, conf])
                carDetections = np.vstack((carDetections, currentArray))
            elif currentClass == "Ambulance" and conf > 0.3:
                print('Ambulance')
                currentArray = np.array([x1, y1, x2, y2, conf])
                ambulanceDetections = np.vstack((ambulanceDetections, currentArray))
            elif currentClass == "Truck" and conf > 0.3:
                print('Truck')
                currentArray = np.array([x1, y1, x2, y2, conf])
                truckDetections = np.vstack((truckDetections, currentArray))

    print('CAR DETECTIONS - ', carDetections)
    print('TRUCK DETECTIONS - ', truckDetections)
    print('BUS DETECTIONS - ', busDetections)
    print('MOTORCYCLE DETECTIONS - ', motorcycleDetections)
    print('BICYCLE DETECTIONS - ', motorcycleDetections)
    print('AUTORIKSHAW DETECTIONS - ', motorcycleDetections)
    print('AMBULANCE DETECTIONS - ', motorcycleDetections)

    carTracker = trackerCar.update(carDetections)
    truckTracker = trackerTruck.update(truckDetections)
    busTracker = trackerBus.update(busDetections)
    motorcycleTracker = trackerMotorcycle.update(motorcycleDetections)
    bicycleTracker = trackerBicycle.update(bicycleDetections)
    autorikshawTracker = trackerAutorikshaw.update(autorikshawDetections)
    ambulanceTracker = trackerAmbulance.update(ambulanceDetections)

    print('CAR TRACKER - ',carTracker)
    print('TRUCK TRACKER - ',truckTracker)
    print('BUS TRACKER - ',busTracker)
    print('MOTORCYCLE TRACKER - ',motorcycleTracker)
    print('BICYCLE TRACKER - ',bicycleTracker)
    print('AUTORIKSHAW TRACKER - ',autorikshawTracker)
    print('AMBULANCE TRACKER - ',ambulanceTracker)

    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    # CAR
    for result in carTracker:
        print('Car tracking')
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),scale=2, thickness=3, offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if carCount.count(id) == 0:
                carCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # TRUCK
    for result in truckTracker:
        print('Truck tracking')
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if truckCount.count(id) == 0:
                truckCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # BUS
    for result in  busTracker:
        print('Bus tracking')
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if busCount.count(id) == 0:
                busCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # MOTORCYCLE
    for result in motorcycleTracker:
        print('Motorcycle tracking')
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if motorcycleCount.count(id) == 0:
                motorcycleCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # BICYCLE
    for result in bicycleTracker:
        print('Bicycle tracking')
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if motorcycleCount.count(id) == 0:
                motorcycleCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # AUTORIKSHAW
    for result in autorikshawTracker:
        print('autorikshaw tracking')
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if autorikshawCount.count(id) == 0:
                autorikshawCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # AMBULANCE
    for result in ambulanceTracker:
        print('ambulance tracking')
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if carCount.count(id) == 0:
                ambulanceCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)


    # cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))
    overCount = "Four Wheeler : "+str(len(carCount))+"\nBus : "+str(len(busCount))+"\nTruck : "+str(len(truckCount))+"\nMotorcycle : "+str(len(motorcycleCount))+"\n Auto Rikshaw : "+str(len(autorikshawCount))
    overCount = """Four Wheeler : {}
    Bus : {}
    Truck : {}
    Motorcycle : {}
    AutoRikshaw : {}
    """.format(len(carCount),len(busCount),len(truckCount),len(motorcycleCount),len(autorikshawCount))
    bgIcon = cv2.imread("icons/bg1.png", cv2.IMREAD_UNCHANGED)
    autoIcon = cv2.resize(cv2.imread("icons/autoIcon.png", cv2.IMREAD_UNCHANGED), iconSize, interpolation= cv2.INTER_LINEAR)
    busIcon = cv2.resize(cv2.imread("icons/busIcon.png", cv2.IMREAD_UNCHANGED), iconSize, interpolation= cv2.INTER_LINEAR)
    truckIcon = cv2.resize(cv2.imread("icons/truckIcon.png", cv2.IMREAD_UNCHANGED), iconSize, interpolation= cv2.INTER_LINEAR)
    bikeIcon = cv2.resize(cv2.imread("icons/bikeIcon.png", cv2.IMREAD_UNCHANGED), iconSize, interpolation= cv2.INTER_LINEAR)
    carIcon = cv2.resize(cv2.imread("icons/carIcon.png", cv2.IMREAD_UNCHANGED), iconSize, interpolation= cv2.INTER_LINEAR)
    img = cvzone.overlayPNG(img, bgIcon, (0, 0))
    img = cvzone.overlayPNG(img, carIcon, (0, 0))
    img = cvzone.overlayPNG(img, bikeIcon, (200, 0))
    img = cvzone.overlayPNG(img, truckIcon, (400, 0))
    img = cvzone.overlayPNG(img, busIcon, (600, 0))
    img = cvzone.overlayPNG(img, autoIcon, (800, 0))
    cv2.putText(img, str(len(carCount)), (90, 40), cv2.FONT_HERSHEY_PLAIN, 2, (50, 50, 255), 3)
    cv2.putText(img, str(len(motorcycleCount)), (290, 40), cv2.FONT_HERSHEY_PLAIN, 2, (50, 50, 255), 3)
    cv2.putText(img, str(len(truckCount)), (490, 40), cv2.FONT_HERSHEY_PLAIN, 2, (50, 50, 255), 3)
    cv2.putText(img, str(len(busCount)), (690, 40), cv2.FONT_HERSHEY_PLAIN, 2, (50, 50, 255), 3)
    cv2.putText(img, str(len(autorikshawCount)), (890, 40), cv2.FONT_HERSHEY_PLAIN, 2, (50, 50, 255), 3)
    print("Cars  :", carCount)
    print("Trucks  :", truckCount)
    print("Bus  :", busCount)
    print("MotorCycle  :", motorcycleCount)
    print("BiCycle  :", bicycleCount)
    print("AutoRikshaw  :", autorikshawCount)
    print("Ambulance  :", ambulanceCount)
    cv2.imshow("Image", img)
    # cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)
