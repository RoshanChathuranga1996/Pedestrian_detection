# limit the number of cpus used by high performance libraries
import os

import matplotlib
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov5')
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

from tracemalloc import start
from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams, VID_FORMATS
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

crosswalk_counter = 0
road_counter = 0
point_matrix_crosswalk = [(0, 0), (0, 0), (0, 0), (0, 0)]
point_matrix_road = [(0, 0), (0, 0), (0, 0), (0, 0)]
name = []
INT_MAX = 10000


def mousePoints_crosswalk(event, x, y, flags, params):
    global crosswalk_counter
    if event == cv2.EVENT_LBUTTONDOWN:
        point_matrix_crosswalk[crosswalk_counter] = (x, y)
        print("Crosswalk Point " + str(crosswalk_counter + 1) + " = " + str(x) + ", " + str(y))
        crosswalk_counter = crosswalk_counter + 1


def mousePoints_road(event, x, y, flags, params):
    global road_counter
    if event == cv2.EVENT_LBUTTONDOWN:
        point_matrix_road[road_counter] = (x, y)
        print("Road Point " + str(road_counter + 1) + " = " + str(x) + ", " + str(y))
        road_counter = road_counter + 1


def select_area(source):
    # colors
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    fonts = cv2.FONT_HERSHEY_COMPLEX

    camera = cv2.VideoCapture(source)
    capture = False

    # counter = 0
    # def mousePoints(event,x,y,flags,params):
    # global counter
    # Left button mouse click event
    # if event == cv.EVENT_LBUTTONDOWN:
    # point_matrix[counter] = (x,y)
    # counter = counter + 1

    while True:
        ret, frame = camera.read()

        img = frame.copy()

        if capture:
            disp_msg = ""
            cv2.putText(frame, disp_msg, (30, 30), fonts, 0.5, GREEN, 2)
        else:
            disp_msg = "Press 'C' to capture the frame"
            cv2.putText(frame, disp_msg, (30, 30), fonts, 0.5, RED, 2)

        cv2.imshow("selected_frame_for_crosswalk", frame)
        key = cv2.waitKey(1)

        if key == ord('c'):
            capture = True

            cv2.imwrite(f'selected_frame_for_crosswalk_roi.png', img)

            break

        if key == ord('q'):
            break

    disp_msg = ""
    cv2.putText(img, disp_msg, (30, 30), fonts, 0.5, RED, 2)

    # roi = cv2.selectROI(img_save)
    # cropped_img = img_save[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
    # cv2.imwrite(f'selected_roi.png', cropped_img)

    while crosswalk_counter < 5:

        disp_msg = "Select four corners of pedestrian cross, then select four corners of road"

        cv2.imshow("line_selector", img)
        cv2.putText(img, disp_msg, (30, 30), cv2.FONT_HERSHEY_COMPLEX
                    , 0.4, (0, 0, 255), 1)
        cv2.setMouseCallback("line_selector", mousePoints_crosswalk)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if len(point_matrix_crosswalk) > 0:
            for x in range(0, 4):
                cv2.circle(img, point_matrix_crosswalk[x], 3, (0, 255, 0), cv2.FILLED)

        if crosswalk_counter == 4:
            # Draw line for area selected area
            cv2.line(img, point_matrix_crosswalk[0], point_matrix_crosswalk[1], (0, 255, 0), 3)
            cv2.line(img, point_matrix_crosswalk[1], point_matrix_crosswalk[2], (0, 255, 0), 3)
            cv2.line(img, point_matrix_crosswalk[2], point_matrix_crosswalk[3], (0, 255, 0), 3)
            cv2.line(img, point_matrix_crosswalk[3], point_matrix_crosswalk[0], (0, 255, 0), 3)
            break

    while road_counter < 5:

        cv2.imshow("line_selector", img)
        cv2.setMouseCallback("line_selector", mousePoints_road)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if len(point_matrix_road) > 0:
            for x in range(0, 4):
                cv2.circle(img, point_matrix_road[x], 3, (0, 255, 0), cv2.FILLED)

        if road_counter == 4:
            # Draw line for area selected area
            cv2.line(img, point_matrix_road[0], point_matrix_road[1], (0, 255, 0), 3)
            cv2.line(img, point_matrix_road[1], point_matrix_road[2], (0, 255, 0), 3)
            cv2.line(img, point_matrix_road[2], point_matrix_road[3], (0, 255, 0), 3)
            cv2.line(img, point_matrix_road[3], point_matrix_road[0], (0, 255, 0), 3)
            break

    cv2.destroyAllWindows()


def calculateTriangleArea(p1, p2, p3):
    # a = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    # b = ((p2[0] - p3[0]) ** 2 + (p2[1] - p3[1]) ** 2) ** 0.5
    # c = ((p3[0] - p1[0]) ** 2 + (p3[1] - p1[1]) ** 2) ** 0.5
    x1 = p1[0]
    x2 = p2[0]
    x3 = p3[0]
    y1 = p1[1]
    y2 = p2[1]
    y3 = p3[1]

    a = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    b = ((x2 - x3) ** 2 + (y2 - y3) ** 2) ** 0.5
    c = ((x3 - x1) ** 2 + (y3 - y1) ** 2) ** 0.5

    # calculate the semi-perimeter
    s = (a + b + c) / 2
    # calculate the area
    area = (s * (s - a) * (s - b) * (s - c)) ** 0.5
    return area


def calculateQuadrilateralArea():
    tri1 = calculateTriangleArea(point_matrix_road[0], point_matrix_road[1], point_matrix_road[2])
    tri2 = calculateTriangleArea(point_matrix_road[2], point_matrix_road[3], point_matrix_road[0])
    totalAreaOfRoad = tri1 + tri2
    # print("Actual Area of Quadrilateral = " + str(totalAreaOfCrosswalk))
    # print("***************************")
    tri3 = calculateTriangleArea(point_matrix_crosswalk[0], point_matrix_crosswalk[1], point_matrix_crosswalk[2])
    tri4 = calculateTriangleArea(point_matrix_crosswalk[2], point_matrix_crosswalk[3], point_matrix_crosswalk[0])
    totalAreaOfCrosswalk = tri3 + tri4

    return totalAreaOfRoad, totalAreaOfCrosswalk


def sumOfCalculatedArea(center):
    sumOfRoadQuadrilateral = 0
    sumOfCrosswalkQuadrilateral = 0

    for cycle in range(3):
        partOfRoadQuadrilateral = calculateTriangleArea(point_matrix_road[cycle], point_matrix_road[cycle + 1], center)
        # print("Triangle No: " + str(cycle + 1))
        # print("Area for this part = " + str(partOfCrosswalkQuadrilateral))
        sumOfRoadQuadrilateral = sumOfRoadQuadrilateral + partOfRoadQuadrilateral
        # print("Total Area = " + str(sumOfCrosswalkQuadrilateral))
        cycle = cycle + 1

        if cycle == 3:
            partOfRoadQuadrilateral = calculateTriangleArea(point_matrix_road[3], point_matrix_road[0], center)
            sumOfRoadQuadrilateral = sumOfRoadQuadrilateral + partOfRoadQuadrilateral
            # print("Triangle No: 4")
            # print("Area for this part = " + str(partOfCrosswalkQuadrilateral))
            # print("Total Area = " + str(sumOfCrosswalkQuadrilateral))
            # print("***************************")
            # print("Total area  = " + str(sumOfCrosswalkQuadrilateral))
            # print("***************************")
            break

    for cycle in range(3):
        partOfCrosswalkQuadrilateral = calculateTriangleArea(point_matrix_crosswalk[cycle],
                                                             point_matrix_crosswalk[cycle + 1], center)
        # print("Triangle No: " + str(cycle + 1))
        # print("Area for this part = " + str(partOfCrosswalkQuadrilateral))
        sumOfCrosswalkQuadrilateral = sumOfCrosswalkQuadrilateral + partOfCrosswalkQuadrilateral
        # print("Total Area = " + str(sumOfCrosswalkQuadrilateral))
        cycle = cycle + 1

        if cycle == 3:
            partOfCrosswalkQuadrilateral = calculateTriangleArea(point_matrix_crosswalk[3], point_matrix_crosswalk[0],
                                                                 center)
            sumOfCrosswalkQuadrilateral = sumOfCrosswalkQuadrilateral + partOfCrosswalkQuadrilateral
            # print("Triangle No: 4")
            # print("Area for this part = " + str(partOfCrosswalkQuadrilateral))
            # print("Total Area = " + str(sumOfCrosswalkQuadrilateral))
            # print("***************************")
            # print("Total area  = " + str(sumOfCrosswalkQuadrilateral))
            # print("***************************")
            break
    return sumOfRoadQuadrilateral, sumOfCrosswalkQuadrilateral


def detectViolation(fullArea, sumOfArea):
    # print("Qua Area = " + str(fullArea))
    # print("Sum of Area = " + str(sumOfArea))

    # print("full area " + str(fullArea) + " sum area " + str(sumOfArea))
    if sumOfArea <= fullArea:  # Inside the ROI
        return True
    else:
        return False


def onSegment(p: tuple, q: tuple, r: tuple) -> bool:
    if ((q[0] <= max(p[0], r[0])) &
            (q[0] >= min(p[0], r[0])) &
            (q[1] <= max(p[1], r[1])) &
            (q[1] >= min(p[1], r[1]))):
        return True

    return False


def orientation(p: tuple, q: tuple, r: tuple) -> int:
    val = (((q[1] - p[1]) *
            (r[0] - q[0])) -
           ((q[0] - p[0]) *
            (r[1] - q[1])))

    if val == 0:
        return 0
    if val > 0:
        return 1  # Collinear
    else:
        return 2  # Clock or counterclock


def doIntersect(p1, q1, p2, q2):
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if (o1 != o2) and (o3 != o4):
        return True

    # Special Cases
    if (o1 == 0) and (onSegment(p1, p2, q1)):
        return True

    # p1, q1 and p2 are collinear and
    # q2 lies on segment p1q1
    if (o2 == 0) and (onSegment(p1, q2, q1)):
        return True

    # p2, q2 and p1 are collinear and
    # p1 lies on segment p2q2
    if (o3 == 0) and (onSegment(p2, p1, q2)):
        return True

    # p2, q2 and q1 are collinear and
    # q1 lies on segment p2q2
    if (o4 == 0) and (onSegment(p2, q1, q2)):
        return True

    return False


def is_inside_polygon(points: list, p: tuple) -> bool:
    n = len(points)

    if n < 3:
        return False

    extreme = (INT_MAX, p[1])
    count = i = 0

    while True:
        nxt = (i + 1) % n

        if (doIntersect(points[i], points[nxt], p, extreme)):

            if orientation(points[i], p, points[nxt]) == 0:
                return onSegment(points[i], p, points[nxt])

            count += 1

        i = nxt

        if i == 0:
            break

    # Return true if count is odd
    return count % 2 == 1


def colorLightStatus():
    # if status is 0, Green light...
    # if status is 1, Yellow light...
    # if status is 2, Red light...

    status = 2
    return status


def prePro():
    image_path = "data/Dataset_Annotated/train/images"

    for image in image_path:
        img = cv2.imread(image)
        # apply Gaussians smooth
        # blur = cv2.GaussianBlur(img, (5,5), 0)
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
        # apply segmentation
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        cv2.imwrite(image, thresh)
        cv2.imgshow(image, thresh, "img_org", "img_pre")


def detect(opt):
    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, \
    project, exist_ok, update, save_crop = \
        opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.exist_ok, opt.update, opt.save_crop
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = select_device(opt.device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Directories
    if type(yolo_model) is str:  # single yolo model
        exp_name = yolo_model.split(".")[0]
    elif type(yolo_model) is list and len(yolo_model) == 1:  # single models after --yolo_model
        exp_name = yolo_model[0].split(".")[0]
    else:  # multiple models after --yolo_model
        exp_name = "ensemble"
    exp_name = exp_name + "_" + deep_sort_model.split('/')[-1].split('.')[0]
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run if project name exists
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)

    # Create as many trackers as there are video sources
    deepsort_list = []
    for i in range(nr_sources):
        deepsort_list.append(
            DeepSort(
                deep_sort_model,
                device,
                max_dist=cfg.DEEPSORT.MAX_DIST,
                max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
            )
        )
    outputs = [None] * nr_sources

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # select ROI for crosswalk & road
    select_area(source)

    # calculate quadrilateral area
    # area = calculateQuadrilateralArea()
    # roadArea = area[0]
    # print("road area " + str(roadArea))
    # crosswalkArea = area[1]
    # print("crosswalk area " + str(crosswalkArea))
    # print("===========================")
    # print(crosswalkArea)

    # Run tracking

    model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if opt.visualize else False
        pred = model(im, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                   max_det=opt.max_det)
        dt[2] += time_sync() - t3

        # violation count
        outCount = 0
        crosswalkCount = 0
        GREEN = (0, 255, 0)
        RED = (0, 0, 255)
        fonts = cv2.FONT_HERSHEY_COMPLEX

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # nr_sources >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=2, pil=not ascii)
            # Get current color light status
            status = colorLightStatus()

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                t4 = time_sync()
                outputs[i] = deepsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    for j, (output) in enumerate(outputs[i]):

                        bboxes = output[0:4]
                        idp = output[4]
                        cls = output[5]
                        conf = output[6]
                        lbl = str(int(idp))

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                        if save_vid or save_crop or show_vid:  # Add bbox to image
                            # Select bounding box
                            x1 = int(output[0])
                            y1 = int(output[1])
                            x2 = int(output[2])
                            y2 = int(output[3])

                            detectPointX = (x2 + x1) / 2
                            detectPointY = (y2 + y1) / 2
                            center = (int(detectPointX), int(y2))
                            # centerOfPed = center
                            im0 = cv2.circle(im0, center, radius=0, color=(0, 0, 255), thickness=-1)
                            # print("Bounding box is ", x1, y1, x2, y2)
                            # print("p = " + str(centerOfPed))
                            # Sum of triangle
                            # sumArea = sumOfCalculatedArea(centerOfPed)
                            # sumRoad = sumArea[0]
                            # print("sum of road " + str(sumRoad))
                            # sumCrosswalk = sumArea[1]
                            # print("sum of crosswalk " + str(sumCrosswalk))

                            # check for violation
                            # detectionStatusRoad = detectViolation(roadArea, sumRoad)
                            # detectionStatusCrosswalk = detectViolation(crosswalkArea, sumCrosswalk)

                            c = int(cls)  # integer class

                            # check pedestrian on the crosswalk or not
                            if status == 0:
                                if is_inside_polygon(points=point_matrix_road, p=center):
                                    if is_inside_polygon(points=point_matrix_crosswalk, p=center):
                                        crosswalkCount += 1
                                        # annotator.box_label(bboxes, lbl, color=colors(6, True))
                                    else:
                                        outCount += 1
                                        annotator.box_label(bboxes, lbl, color=colors(c, True))

                            if status == 1:
                                if is_inside_polygon(points=point_matrix_road, p=center):
                                    if is_inside_polygon(points=point_matrix_crosswalk, p=center):
                                        crosswalkCount += 1
                                        annotator.box_label(bboxes, lbl, color=colors(6, True))
                                    else:
                                        outCount += 1
                                        annotator.box_label(bboxes, lbl, color=colors(c, True))

                            if status == 2:
                                if is_inside_polygon(points=point_matrix_road, p=center):
                                    if is_inside_polygon(points=point_matrix_crosswalk, p=center):
                                        crosswalkCount += 1

                                        annotator.box_label(bboxes, lbl, color=colors(6, True))
                                    else:
                                        outCount += 1
                                        annotator.box_label(bboxes, lbl, color=colors(c, True))

                            # if status == 0:

                            # if detectionStatusRoad:
                            # if detectionStatusCrosswalk:
                            # print("--No violation detected--")
                            # disp_msg = "--No violation detected--"
                            # cv2.putText(im0, disp_msg, (30, 60), fonts, 0.5, GREEN, 2)
                            # annotator.box_label(bboxes, lbl, color=colors(6, True))
                            # else:
                            # print("--Out of crosswalk violation detected--")
                            # disp_msg = "Out of crosswalk violation detected--"
                            # cv2.putText(im0, disp_msg, (30, 30), fonts, 0.5, RED, 2)
                            # annotator.box_label(bboxes, lbl, color=colors(c, True))

                            # elif status == 1:
                            # if detectionStatusRoad:
                            # if detectionStatusCrosswalk:
                            # print("-- Yellow light violation detected--")
                            # disp_msg = "--Yellow light violation detected--"
                            # cv2.putText(im0, disp_msg, (30, 60), fonts, 0.5, GREEN, 2)
                            # annotator.box_label(bboxes, lbl, color=colors(6, True))
                            # else:
                            # print("--Out of crosswalk violation detected---")
                            # disp_msg = "--Out of crosswalk violation detected---"
                            # cv2.putText(im0, disp_msg, (30, 30), fonts, 0.5, RED, 2)
                            # annotator.box_label(bboxes, lbl, color=colors(c, True))

                            # elif status == 2:
                            # if detectionStatusRoad:
                            # if detectionStatusCrosswalk:
                            # print("--Red light violation detected--")
                            # disp_msg = "--Red light violation detected--"
                            # cv2.putText(im0, disp_msg, (30, 60), fonts, 0.5, GREEN, 2)
                            # annotator.box_label(bboxes, lbl, color=colors(6, True))
                            # break
                            # else:
                            # print("--Out of crosswalk violation detected--")
                            # disp_msg = "--Out of crosswalk violation detected--"
                            # cv2.putText(im0, disp_msg, (30, 30), fonts, 0.5, RED, 2)
                            # annotator.box_label(bboxes, lbl, color=colors(c, True))

                            if save_crop:
                                txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[
                                    c] / f'{id}' / f'{p.stem}.jpg', BGR=True)

                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')

            else:
                deepsort_list[i].increment_ages()
                LOGGER.info(f' No detections')

            # print detection counts according to violation
            if status == 0:
                disp_msg1 = "Green light"
                cv2.putText(im0, disp_msg1, (30, 30), fonts, 0.5, GREEN, 1)
                print("Green light")
                disp_msg2 = str(outCount) + " Out of crosswalk violation detected"
                cv2.putText(im0, disp_msg2, (30, 60), fonts, 0.5, RED, 1)
                print(str(outCount) + " Out of crosswalk violation detected")
            if status == 1:
                disp_msg1 = str(crosswalkCount) + " Yellow light violation detected"
                cv2.putText(im0, disp_msg1, (30, 30), fonts, 0.5, GREEN, 1)
                print(str(crosswalkCount) + " Yellow light violation detected")
                disp_msg2 = str(outCount) + " Out of crosswalk violation detected"
                cv2.putText(im0, disp_msg2, (30, 60), fonts, 0.5, RED, 1)
                print(str(outCount) + " Out of crosswalk violation detected")
            if status == 2:
                disp_msg1 = str(crosswalkCount) + " Red light violation detected"
                cv2.putText(im0, disp_msg1, (30, 30), fonts, 0.5, GREEN, 1)
                print(str(crosswalkCount) + " Red light violation detected")
                disp_msg2 = str(outCount) + " Out of crosswalk violation detected"
                cv2.putText(im0, disp_msg2, (30, 60), fonts, 0.5, RED, 1)
                print(str(outCount) + " Out of crosswalk violation detected")

            # Stream results
            im0 = annotator.result()
            if show_vid:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
        per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_model)  # update model (to fix SourceChangeWarning)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    with torch.no_grad():
        detect(opt)
