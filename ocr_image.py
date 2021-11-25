from typing import Counter
import cv2
import numpy as np
from numpy.lib.type_check import imag
import pytesseract
from PIL import Image, ImageOps
import requests
import win32gui
from PIL import ImageGrab


def send_request(metric_dict):

    # ----- 1. get the token ----- #
    LOGIN_URL = "https://mmj.advancedcare.com/api/sign/in"
    body = {
        "login": "dev@miqare.com",
        "password": "2021Time@AC"
    }

    # sending post request and saving the response as response object
    response = requests.post(url = LOGIN_URL, json = body)

    # extracting data in json format
    data = response.json()

    # extract token info from response data
    token = data['result']['token']
    print('token : ', token)


    # ----- 2. Fill in the values and send request ----- #
    URL = "https://mmj.advancedcare.com/api/patient/vital"
    header = {
        "authorization": token,
        "content-type": "application/json"
    }

    body = {
        "items":[
            {"vital_type":"pulse","pai_pi":13.949999809265137,"patient_id":6979,"heart_rate":78,"oxygen":98},
            {"steps":24,"patient_id":6979,"cadence":1,"activity":0.38671875,"vital_type":"accelerometer"},
            {"height":170,"height_type":"centimeters","vital_type":"height", "patient_id":6979},
            {"weight":"89","vital_type":"weight","weight_type":"kg", "patient_id":6979},
            {"temperature_type":"celsius","temperature":"36","vital_type":"temperature", "patient_id":6979},
            {"blood_pressure_systolic":106,"heart_rate":80,"vital_type":"blood_pressure","blood_pressure_diastolic":78, "patient_id":6979},
            {"respiratory":"15","vital_type":"respiratory", "patient_id":6979},
            {"value":"0.0", "unit": "L/min","vital_type":"ventilation", "patient_id":6979},
            {"value":"0.0", "unit": "L","vital_type":"tidal_volume", "patient_id":6979}
        ]
    }

    body['items'][0]['heart_rate'] = metric_dict['Heart Rate']
    body['items'][0]['oxygen'] = metric_dict['Oxygen Saturation']
    body['items'][1]['steps'] = metric_dict['Steps']
    body['items'][1]['cadence'] = metric_dict['Cadence']
    body['items'][4]['temperature'] = metric_dict['Temperature']
    body['items'][5]['blood_pressure_diastolic'] = metric_dict['Blood Pressure']
    # body['items'][5]['blood_pressure_systolic'] = metric_dict['Blood Pressure']
    # body['items'][5]['heart_rate'] = metric_dict['Heart Rate']
    body['items'][6]['respiratory'] = metric_dict['Breathing Rate']    
    body['items'][7]['value'] = metric_dict['Ventilation']
    body['items'][8]['value'] = metric_dict['Tidal Volume']

    # sending post request and saving the response as response object
    response = requests.post(url=URL, headers=header, json=body)

    # extracting data in json format
    data = response.json()

def recognize_device(image, cntr_device):

    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    options = " --psm 1 --oem 3"

    serial_contour = cntr_device[-1]
    (x, y, w, h) = cv2.boundingRect(serial_contour)
    serial_roi = image[
        y:y + h,
        x:x + w
    ]
    # # show the serial number region    
    # cv2.imshow('Serial ROI', serial_roi)
    # cv2.waitKey(5)

    lpText = pytesseract.image_to_string(serial_roi, config=options)
    serial_number = lpText.split()[0]
    serial_number = serial_number.replace(',', '')

    power_contour = cntr_device[-2]
    (x, y, w, h) = cv2.boundingRect(power_contour)
    horz_center = int(x + w / 2)
    power_roi = image[
        y + 2 : y + h - 2,
        horz_center + 2 : x + w - 2
    ]
    # # show the power info region
    # cv2.imshow('Power ROI', power_roi)
    # cv2.waitKey(5)

    lpText = pytesseract.image_to_string(power_roi, config=options)
    power_value = lpText.split()[-1].replace('%', '')

    return serial_number, power_value

def recognition_metrics(image, cntr_metrics):

    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    options = "--psm 8 --oem 3"
    # options = "--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789"

    metrics = []
    for contour in cntr_metrics:
        (x, y, w, h) = cv2.boundingRect(contour)
        # print('left : {}, top : {}, width : {}, height : {}'.format(x, y, w, h))
        roi = image[
            y:y + h,
            x:x + w
        ]

        horiz_center = int(len(roi) / 2)
        roi = roi[horiz_center - 5:, :]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        roi_bin = cv2.threshold(roi_gray, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        cv2.imshow('ROI_bin', roi_bin)
        cv2.waitKey(500)

        candi_contours =cv2.findContours(roi_bin,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
        boundingBoxes = [cv2.boundingRect(c) for c in candi_contours]

        (candi_contours, boundingBoxes) = zip(*sorted(zip(candi_contours, boundingBoxes),
                                                      key=lambda b: b[1][0], reverse=True))

        max_height = np.max([item[3] for item in boundingBoxes])
        candi_character = []
        pot = True
        for cntr in candi_contours:
            (x, y, w, h) = cv2.boundingRect(cntr)
            candi_character.append((x, y, w, h))
            # if (h > max_height * 0.75):
            #     candi_character.append((x, y, w, h))
            # elif (h < max_height * 0.2):
            #     if len(candi_character) > 0 and pot:
            #         candi_character.append((x, y, w, h))
            #         pot = False
        
        cv2.rectangle(
            roi, 
            (candi_character[-1][0], candi_character[-1][1]), 
            (candi_character[-1][0] + candi_character[-1][2], candi_character[-1][1] + candi_character[-1][3]),
            (0, 0, 255), 2)
        cv2.imshow('ROI', roi)
        cv2.waitKey(0)

        del candi_character[-1]
        if len(candi_character) == 0:
            metrics.append(-1.0)
        else:
            # show the serial number region
            cv2.rectangle( roi, 
                (candi_character[-1][0] - 5, 0),
                (candi_character[0][0] + candi_character[0][2] + 5, len(roi)),
                (0, 255, 0), 2)
            cv2.imshow('ROI', roi)
            cv2.waitKey(1000)

            roi = roi[
                0 : len(roi) - 5,
                candi_character[-1][0] - 5 : candi_character[0][0] + candi_character[0][2] + 5
            ]

            lpText = pytesseract.image_to_string(roi, config=options)
            metrics.append(float(lpText.split()[0]))

    metrics = metrics[::-1]
    return metrics

def analysis_image(input_image):

    HEIGHT, WIDTH = input_image.shape[:2]
    # # show the original image
    # cv2.imshow('original image', input_image)
    # cv2.waitKey(0)

    # convert image to HSV color space
    hsv_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2HSV)
    # # show the hsv image
    # cv2.imshow('hsv image', hsv_image)
    # cv2.waitKey(0)

    # make the color filter mask
    mask = cv2.inRange(hsv_image, (0, 0, 230), (255, 100, 255))
    # # show the color filter mask
    # cv2.imshow('mask', mask)
    # cv2.waitKey(500)

    # perform mask to original image
    image = cv2.bitwise_and(input_image, input_image, mask=mask)
    # # show the masked image
    # cv2.imshow('image', image)
    # cv2.waitKey(0)

    # find the rectangular reiongs from the image
    contours =cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
    cntr_device = []
    cntr_metrics = []
    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        epsilon = 0.005*cv2.arcLength(contour,True)
        approx = cv2.approxPolyDP(contour,epsilon,True)
        if len(approx) == 4 and ((w > WIDTH / 4) and (w < WIDTH / 3)):
            cntr_metrics.append(approx)
            cv2.drawContours(input_image,cntr_metrics,-1,(0,255,0),2)
        elif len(approx) == 4 and (w > WIDTH / 2):
            cntr_device.append(approx)
            cv2.drawContours(input_image,cntr_device,-1,(0,0,255),2)

    # print('number of found metrics regions : ', len(cntr_metrics))
    # print('number of found device info regions : ', len(cntr_device))

    # cv2.imshow('original image', input_image)
    # cv2.waitKey(0)

    # recognition device infos
    try:
        serial_number, power_value = recognize_device(image, cntr_device)
    except:
        serial_number, power_value = -1, -1
    print('serial number : {}, power value : {}'.format(serial_number, power_value))

    # recognition metrics
    metrics_label = ['Heart Rate', 'Steps', 'Cadence', 'Breathing Rate', 'Ventilation',
                    'Tidal Volume', 'Oxygen Saturation', 'Temperature', 'Blood Pressure']
    try:
        metrics = recognition_metrics(image, cntr_metrics)
        metric_dict = dict(zip(metrics_label, metrics))
        print(metric_dict)
        # send_request(metric_dict)
    except:
        print('metrics recognition failed')


if __name__ == '__main__':

    # input the image
    input_image = cv2.imread('./image/1.png')
    input_image = cv2.resize(input_image, (685, 945))
    analysis_image(input_image)