import cv2 as cv

crosswalk_counter = 0
point_matrix = [(0, 0), (0, 0), (0, 0), (0, 0)]


def mousePoints(event, x, y, flags, params):
    global crosswalk_counter
    if event == cv.EVENT_LBUTTONDOWN:
        point_matrix[counter] = (x, y)
        print('x---------', x)
        print('y---------', y)
        counter = counter + 1


img = cv.imread('selected_frame_for_roi.png')

while crosswalk_counter < 5:

    disp_msg = "Select four corners of pedestrian cross  --exit to q"

    cv.imshow("line_selector", img)
    cv.putText(img, disp_msg, (30, 30), cv.FONT_HERSHEY_COMPLEX
               , 0.6, (0, 0, 255), 2)
    cv.setMouseCallback("line_selector", mousePoints)

    key = cv.waitKey(1)
    if key == ord('q'):
        break
    if len(point_matrix) > 0:
        for x in range(0, 4):
            cv.circle(img, point_matrix[x], 3, (0, 255, 0), cv.FILLED)

    if crosswalk_counter == 4:
        # Draw line for area selected area
        cv.line(img, point_matrix[0], point_matrix[1], (0, 255, 0), 3)
        cv.line(img, point_matrix[1], point_matrix[2], (0, 255, 0), 3)
        cv.line(img, point_matrix[2], point_matrix[3], (0, 255, 0), 3)
        cv.line(img, point_matrix[3], point_matrix[0], (0, 255, 0), 3)


