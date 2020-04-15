#!/usr/bin/env python3
import argparse
import calendar
import io
import json
import os
import subprocess as sp
import sys
from decimal import Decimal
from math import exp, log, sqrt
from pathlib import Path
from threading import Thread

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

from logzero import logger


'''
Base Class for FLIR thermal images
'''

class CFlir():
    line_flag = False
    contour = []
    drawing = False
    scale_moving = False
    measurement_moving = False
    rect_moving = False
    spots_moving = False
    xo,yo = 0,0
    xdisp,ydisp = None,None
    measurement_index = None
    rect_index = None
    spots_index = None

    def __init__(self,image_path, color_map = 'jet'):
        self.image_path = image_path
        self.cmap = color_map
        self.thermal_np, self.raw_thermal_np, self.meta = self.extract_temperatures()
        self.thermal_image = CFlir.get_temp_image(self.thermal_np, colormap=self.cmap)
        self.global_min_temp = np.min(self.thermal_np)
        self.global_max_temp = np.max(self.thermal_np)
        self.scale_contours = []
        self.measurement_contours = []
        self.measurement_rects = []
        self.spots = []

    @staticmethod
    def raw2temp(raw, E=0.9,OD=1,RTemp=20,ATemp=20,IRWTemp=20,IRT=1,RH=50,PR1=21106.77,PB=1501,PF=1,PO=-7340,PR2=0.012545258):
        """ convert raw values from the flir sensor to temperatures in °C """
        # this calculation has been ported to python from https://github.com/gtatters/Thermimage/blob/master/R/raw2temp.R
        # a detailed explanation of what is going on here can be found there

        # constants
        ATA1=0.006569; ATA2=0.01262; ATB1=-0.002276; ATB2=-0.00667; ATX=1.9; #RH=0
        # transmission through window (calibrated)
        emiss_wind = 1 - IRT
        refl_wind = 0
        # transmission through the air
        h2o = (RH/100)*exp(1.5587+0.06939*(ATemp)-0.00027816*(ATemp)**2+0.00000068455*(ATemp)**3)
        tau1 = ATX*exp(-sqrt(OD/2)*(ATA1+ATB1*sqrt(h2o)))+(1-ATX)*exp(-sqrt(OD/2)*(ATA2+ATB2*sqrt(h2o)))
        tau2 = ATX*exp(-sqrt(OD/2)*(ATA1+ATB1*sqrt(h2o)))+(1-ATX)*exp(-sqrt(OD/2)*(ATA2+ATB2*sqrt(h2o)))        
        # radiance from the environment
        raw_refl1 = PR1/(PR2*(exp(PB/(RTemp+273.15))-PF))-PO
        raw_refl1_attn = (1-E)/E*raw_refl1 # Reflected component

        raw_atm1 = PR1/(PR2*(exp(PB/(ATemp+273.15))-PF))-PO # Emission from atmosphere 1
        raw_atm1_attn = (1-tau1)/E/tau1*raw_atm1 # attenuation for atmospheric 1 emission

        raw_wind = PR1/(PR2*(exp(PB/(IRWTemp+273.15))-PF))-PO # Emission from window due to its own temp
        raw_wind_attn = emiss_wind/E/tau1/IRT*raw_wind # Componen due to window emissivity

        raw_refl2 = PR1/(PR2*(exp(PB/(RTemp+273.15))-PF))-PO # Reflection from window due to external objects
        raw_refl2_attn = refl_wind/E/tau1/IRT*raw_refl2 # component due to window reflectivity

        raw_atm2 = PR1/(PR2*(exp(PB/(ATemp+273.15))-PF))-PO # Emission from atmosphere 2
        raw_atm2_attn = (1-tau2)/E/tau1/IRT/tau2*raw_atm2 # attenuation for atmospheric 2 emission

        raw_obj = (raw/E/tau1/IRT/tau2-raw_atm1_attn-raw_atm2_attn-raw_wind_attn-raw_refl1_attn-raw_refl2_attn)
        val_to_log = PR1/(PR2*(raw_obj+PO))+PF
        if any(val_to_log.ravel()<0):
            logger.warning('Image seems to be corrupted')
            val_to_log = np.where(val_to_log<0, sys.float_info.min, val_to_log)
        # temperature from radiance
        temp_C = PB/np.log(val_to_log)-273.15

        return temp_C

    @staticmethod
    def parse_temp(temp_str):
        # TODO: do this right
        # we assume degrees celsius
        return (float(temp_str.split()[0]))

    @staticmethod
    def parse_length(length_str):
        # TODO: do this right
        # we assume meters
        return (float(length_str.split()[0]))

    @staticmethod
    def parse_percent(percentage_str):
        return (float(percentage_str.split()[0]))

    def generate_colorbar(self, min_temp=None, max_temp=None, cmap=cv.COLORMAP_JET, height=None):
        if min_temp is None:
            min_temp = self.global_min_temp
        if max_temp is None:
            max_temp = self.global_max_temp
        cb_gray = np.arange(255,0,-1,dtype=np.uint8).reshape((255,1))
        if cmap is not None:
            cb_color = cv.applyColorMap(cb_gray, cmap)
        else:
            cb_color = cv.cvtColor(cb_gray, cv.COLOR_GRAY2BGR)
        for i in range(1,6):
            cb_color = np.concatenate( (cb_color, cb_color), axis=1 )
        
        if height is None:
            append_img = np.zeros( (self.thermal_image.shape[0], cb_color.shape[1]+30, 3), dtype=np.uint8 )
        else:
            append_img = np.zeros( (height, cb_color.shape[1]+30, 3), dtype=np.uint8 )

        append_img[append_img.shape[0]//2-cb_color.shape[0]//2  : append_img.shape[0]//2 - (cb_color.shape[0]//2) + cb_color.shape[0] , 10 : 10 + cb_color.shape[1] ] = cb_color
        cv.putText(append_img, str(min_temp), (5, append_img.shape[0]//2 - (cb_color.shape[0]//2) + cb_color.shape[0] + 30), cv.FONT_HERSHEY_PLAIN, 1, (255,0,0) , 1, 8)
        cv.putText(append_img, str(max_temp), (5, append_img.shape[0]//2-cb_color.shape[0]//2-20) , cv.FONT_HERSHEY_PLAIN, 1, (0,0,255) , 1, 8 )
        return append_img

    def extract_temperatures(self):
        """ extracts the thermal image as 2D numpy array with temperatures in degC """

        # read image metadata needed for conversion of the raw sensor values
        # E=1,SD=1,RTemp=20,ATemp=RTemp,IRWTemp=RTemp,IRT=1,RH=50,PR1=21106.77,PB=1501,PF=1,PO=-7340,PR2=0.012545258
        try:
            meta_json = sp.Popen(f'exiftool "{self.image_path}" -Emissivity -ObjectDistance -AtmosphericTemperature -ReflectedApparentTemperature -IRWindowTemperature -IRWindowTransmission -RelativeHumidity -PlanckR1 -PlanckB -PlanckF -PlanckO -PlanckR2 -j', shell=True, stdout=sp.PIPE).communicate()[0]
        except:
            meta_json = sp.Popen(f'exiftool.exe "{self.image_path}" -Emissivity -ObjectDistance -AtmosphericTemperature -ReflectedApparentTemperature -IRWindowTemperature -IRWindowTransmission -RelativeHumidity -PlanckR1 -PlanckB -PlanckF -PlanckO -PlanckR2 -j', shell=True, stdout=sp.PIPE).communicate()[0]

        meta = json.loads(meta_json)[0]

        #exifread can't extract the embedded thermal image, use exiftool instead
        # sp popen can't handle bytes
        try:
            thermal_img_bytes = sp.check_output(['exiftool','-RawThermalImage','-b', f'{self.image_path}'])
        except:
            thermal_img_bytes = sp.check_output(['exiftool.exe','-RawThermalImage','-b', f'{self.image_path}'])

        thermal_img_stream = io.BytesIO(thermal_img_bytes)

        thermal_img = Image.open(thermal_img_stream)
        raw_thermal_np = np.array(thermal_img)

        # raw values -> temperature E=meta['Emissivity']
        raw2tempfunc = (lambda x: CFlir.raw2temp(x, E=0.9, OD=CFlir.parse_length(meta['ObjectDistance']), RTemp=CFlir.parse_temp(meta['ReflectedApparentTemperature']), ATemp=CFlir.parse_temp(meta['AtmosphericTemperature']), IRWTemp=CFlir.parse_temp(meta['IRWindowTemperature']), IRT=meta['IRWindowTransmission'], RH=CFlir.parse_percent(meta['RelativeHumidity']), PR1=meta['PlanckR1'], PB=meta['PlanckB'], PF=meta['PlanckF'], PO=meta['PlanckO'], PR2=meta['PlanckR2']))
        thermal_np = raw2tempfunc(raw_thermal_np)

        return thermal_np,raw_thermal_np,meta

    @staticmethod
    def normalize(thermal_np):
        num = thermal_np - np.amin(thermal_np)
        den = (np.amax(thermal_np)-np.amin(thermal_np))
        thermal_np = num/den
        return thermal_np

    @staticmethod
    def get_temp_image(thermal_np, colormap=cv.COLORMAP_JET):
        thermal_np_norm = CFlir.normalize(thermal_np)
        thermal_image = np.array(thermal_np_norm*255, dtype=np.uint8)
        if colormap != None:
            thermal_image = cv.applyColorMap(thermal_image, colormap)
        return thermal_image

    @staticmethod
    def draw_contour_area( event, x, y, flags, params):
        thermal_image = params[0]
        contours = params[1]

        is_rect = params[2][0]
        point1 = params[2][1]
        point2 = params[2][2]

        if event == cv.EVENT_LBUTTONDOWN:
            if CFlir.drawing == False:
                CFlir.drawing=True
                if is_rect:
                    point1[0] = ((x, y))
    
        elif event==cv.EVENT_MOUSEMOVE:
            if CFlir.drawing==True:
                if not is_rect:
                    cv.circle(thermal_image, (x,y), 1, (0,0,0), -1)
                    CFlir.contour.append((x,y))
                else:
                    point2[0] = ((x,y))

        elif event==cv.EVENT_LBUTTONUP:
            CFlir.drawing=False
            CFlir.contour = np.asarray(CFlir.contour, dtype=np.int32)
            if len(CFlir.contour) > 0:
                contours.append(CFlir.contour)
            CFlir.contour=[]


    @staticmethod
    def draw_spots(event, x, y, flags, params):
        point = params[0]
        flag = params[1]
        point.clear()

        if event == cv.EVENT_MOUSEMOVE:  
            if CFlir.drawing == True:
                point.append(x)
                point.append(y)
        
        elif event ==  cv.EVENT_LBUTTONDOWN:
            CFlir.drawing == False
            point.append(x)
            point.append(y)
            flag[0] = False


    def get_spots(self, thermal_image):
        CFlir.drawing = True
        image_copy = thermal_image.copy()
        original_copy = image_copy.copy()
        if len(original_copy.shape) < 3:
            cmap_copy = cv.applyColorMap(original_copy, cv.COLORMAP_JET)

        point = []
        spot_points = []
        flag = [True]
        cv.namedWindow('Image')
        cv.setMouseCallback('Image', CFlir.draw_spots, (point, flag) )
        while(1):         
            image_copy = original_copy.copy()
            for i in range(0,len(spot_points)):
                cv.circle(image_copy, spot_points[i] , 5, 0, -1)
                try:
                    cv.circle(cmap_copy, spot_points[i] , 5, 0, -1)
                except:
                    cv.circle(original_copy, spot_points[i] , 5, 0, -1)

            if len(point) > 0:
                cv.circle(image_copy, tuple(point) , 5, 0, -1)


            if flag[0] == False:
                spot_points.append(tuple(point))
                flag[0] = True


            cv.imshow('Image', image_copy)
            k = cv.waitKey(1) & 0xff
            
            if k == 13 or k == 141 :
                break

        CFlir.drawing = False
        cv.destroyAllWindows()
        # origi_copy = cv.UMat(origi_copy)
        if len(original_copy.shape) == 3:
            gray = cv.cvtColor(original_copy, cv.COLOR_BGR2GRAY)
        else:
            gray = cv.cvtColor(cmap_copy, cv.COLOR_BGR2GRAY)

        ret,thresh = cv.threshold(gray, 10, 255, cv.THRESH_BINARY_INV)
        contours, hierarchy = cv.findContours(thresh,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE )
        self.spots = contours 


    def get_spots_values(self, thermal_image, thermal_np, raw_thermal_np, contours ):
        spots_measurement_values = []
        for i in range(0,len(contours)):
            spots_measurement_values.append(CFlir.get_roi(thermal_image, thermal_np, raw_thermal_np, self.spots, i)[1] )
        
        return spots_measurement_values
            


    @staticmethod
    def draw_line(event, x ,y ,flags, params):

        lp1,lp2 = params[0],params[1]
        thermal_image = params[2]

        if len(lp1)<=2 and len(lp2) <2:        
            if event == cv.EVENT_LBUTTONDOWN:
                CFlir.line_flag = not(CFlir.line_flag)
                if CFlir.line_flag == True:
                    lp1.append(x)
                    lp1.append(y)
                
                else:
                    lp2.append(x)
                    lp2.append(y)
                    lp1 = tuple(lp1)
                    lp2 = tuple(lp2)
                    cv.line(thermal_image, lp1, lp2, (0,0,0), 2, 8 )
        

    @staticmethod
    def get_line(image):
        point1 = []
        point2 = []

        cv.namedWindow('image')
        cv.setMouseCallback('image',CFlir.draw_line, (point1, point2, image) )
        
        while(1):
            cv.imshow('image', image)
            k=cv.waitKey(1) & 0xFF

            if k==13 or k==141:
                break

        cv.destroyWindow('image')  
        
        thresh = 15
        line = []
        p1x,p1y = point1[0],point1[1]
        p2x,p2y = point2[0],point2[1]

        if  abs((p1x-p2x))>thresh and abs((p1y-p2y))>thresh :
            #Using y = mx + c
            m = (p2y - p1y) / (p2x-p1x) 
            c = p2y - (m*p2x) 
            if p1x > p2x:
                for x in range(p1x,p2x-1,-1):
                    y = int( (m*x) + c)
                    line.append((x,y))
            else:
                for x in range(p1x,p2x+1):
                    y = int( (m*x) + c )
                    line.append((x,y))

        elif abs(p1x-p2x) <= thresh:
            if p1y>p2y:
                for y in range(p1y,p2y-1,-1):
                    line.append((p1x,y))
            else:
                for y in range(p1y,p2y+1):
                    line.append( (p1x,y) )
        
        else:
            if p1x > p2x:
                for x in range(p1x,p2x-1,-1):
                    line.append((x,p1y))
            else:
                for x in range(p1x,p2x+1):
                    line.append((x,p1y))

        return line, (p1x,p1y) , (p2x,p2y)


    def line_measurement(self, image, thermal_np, cmap=cv.COLORMAP_JET):
        img = image.copy()
        line, point1, point2 = CFlir.get_line(img)
        line_temps = np.zeros(len(line))
    
        if len(img.shape) == 3:
            gray_values = np.arange(256, dtype=np.uint8)
            color_values = map(tuple, cv.applyColorMap(gray_values, cmap).reshape(256, 3))
            color_to_gray_map = dict(zip(color_values, gray_values))
            img = np.apply_along_axis(lambda bgr: color_to_gray_map[tuple(bgr)], 2, image)
        
        for i in range(0,len(line)):
            line_temps[i] = thermal_np[ line[i][1], line[i][0] ]
            
        cv.line(img, point1, point2, 255, 2, 8)
        
        plt.subplot(1, 5, (1,2) )
        plt.imshow(img, cmap='jet')
        plt.title('Image')
        plt.subplot(1, 5, (4,5) )
        plt.plot(line_temps)
        plt.title('Distance vs Temperature')
        plt.show() 
        
        logger.info(f'\nMin line: {np.amin(line_temps)}\nMax line: {np.amax(line_temps)}' )
    
    @staticmethod
    def is_in_rect(rectangle, point):
        tlx,tly,w,h = rectangle
        px, py = point
        is_inside = False
        if px>tlx and px<tlx+w:
            if py>tly and py<tly+h:
                is_inside = True
        return is_inside

    @staticmethod
    def move_contours(event, x, y, flags, params): # scale contour,emissivity contours

        CFlir.xdisp = None
        CFlir.ydisp = None
        measurement_contours = params[0]
        measurement_rects = params[1]
        scale_contours = params[2]
        spot_contours = params[3]
        img = params[4]
        vals = params[5]
        spot_vals = params[6]
        scale_contour = []

        if len(scale_contours)>0:
            scale_contour = scale_contours[0]

        if CFlir.measurement_moving == True:
            measurement_cont = measurement_contours[CFlir.measurement_index]

        if CFlir.rect_moving == True:
            measurement_rect = measurement_rects[CFlir.rect_index]

        if CFlir.spots_moving == True:
            spot_cont = spot_contours[CFlir.spots_index]    

        if event == cv.EVENT_RBUTTONDOWN:
            for i in range(0, len(measurement_contours) ):
                if cv.pointPolygonTest(measurement_contours[i], (x,y), False) == 1:
                    CFlir.measurement_index = i 
                    CFlir.xo = x
                    CFlir.yo = y
                    CFlir.measurement_moving = True
                    break
            
            for i in range(0,len(measurement_rects)):
                if x >= measurement_rects[i][0] and x <= (measurement_rects[i][0]+measurement_rects[i][2]):
                    if y >= measurement_rects[i][1] and y <= (measurement_rects[i][1]+measurement_rects[i][3]):
                        CFlir.rect_index = i
                        CFlir.xo = x
                        CFlir.yo = y
                        CFlir.rect_moving = True
                        break

            if len(scale_contours)>0:
                if cv.pointPolygonTest(scale_contour, (x,y), False) == 1:
                    CFlir.xo = x
                    CFlir.yo = y
                    CFlir.scale_moving = True


            for i in range(0, len(spot_contours)):
                if cv.pointPolygonTest(spot_contours[i], (x,y), False) == 1:
                    CFlir.spots_index = i 
                    CFlir.xo = x
                    CFlir.yo = y
                    CFlir.spots_moving = True
                    break



        elif event == cv.EVENT_MOUSEMOVE:
            if CFlir.measurement_moving == True:
                measurement_cont[:,0] += (x-CFlir.xo)
                measurement_cont[:,1] += (y-CFlir.yo)
                
                if np.max(measurement_cont[:,0]) >= img.shape[1] or np.amax(measurement_cont[:,1]) >= img.shape[0] or np.amin(measurement_cont[:,0]) <= 0 or np.amin(measurement_cont[:,1]) <= 0:
                    measurement_cont[:,0] -= (x-CFlir.xo)
                    measurement_cont[:,1] -= (y-CFlir.yo)
                    logger.warning("Could not move to intended location. Check if points are exceeding image boundary")
                else:
                    CFlir.xo = x
                    CFlir.yo = y

            if CFlir.rect_moving is True:
                x_new = measurement_rect[0] + (x-CFlir.xo)
                y_new = measurement_rect[1] + (y-CFlir.yo)
                
                if x_new >= img.shape[1]-measurement_rect[2]:
                    x_new = img.shape[1]-measurement_rect[2]-1
                if x_new <= 0:
                    x_new = 1
                if y_new >= img.shape[0]-measurement_rect[3]:
                    y_new = img.shape[0]-measurement_rect[3]-1
                if y_new <= 0:
                    y_new = 1
                measurement_rects[CFlir.rect_index] = x_new,y_new, measurement_rect[2], measurement_rect[3]
                CFlir.xo = x
                CFlir.yo = y

            if CFlir.scale_moving == True:
                scale_contour[:,0] += (x-CFlir.xo)
                scale_contour[:,1] += (y-CFlir.yo)

                if np.max(scale_contour[:,0]) >= img.shape[1] or np.amax(scale_contour[:,1]) >= img.shape[0] or np.amin(scale_contour[:,0]) <= 0 or np.amin(scale_contour[:,1]) <= 0:
                    scale_contour[:,0] -= (x-CFlir.xo)
                    scale_contour[:,1] -= (y-CFlir.yo)
                    logger.warning("Could not move to intended location. Check if points are exceeding image boundary")
                else:
                    CFlir.xo = x
                    CFlir.yo = y

            if CFlir.spots_moving == True:
                spot_cont[:,0,0] += (x-CFlir.xo)
                spot_cont[:,0,1] += (y-CFlir.yo)

                if np.max(spot_cont[:,0,0]) >= img.shape[1] or np.amax(spot_cont[:,0,1]) >= img.shape[0] or np.amin(spot_cont[:,0,0]) <= 0 or np.amin(spot_cont[:,0,1]) <= 0:
                    spot_cont[:,0,0] -= (x-CFlir.xo)
                    spot_cont[:,0,1] -= (y-CFlir.yo)
                    logger.warning("Could not move to intended location. Check if points are exceeding image boundary")
                else:
                    CFlir.xo = x
                    CFlir.yo = y

        elif event== cv.EVENT_RBUTTONUP:
            CFlir.scale_moving = False
            CFlir.measurement_moving = False
            CFlir.spots_moving = False
            CFlir.rect_moving = False

        elif event == cv.EVENT_LBUTTONDBLCLK:
            for i in range(0,len(measurement_contours)):
                if cv.pointPolygonTest(measurement_contours[i], (x,y), False) == 1:
                    logger.info(f'\nMaximum temp: {np.amax(vals[i])}\nMinimum temp: {np.amin(vals[i])}\nAvg: {np.average(vals[i])}\n' )
            
            for i in range(len(measurement_rects)):
                if CFlir.is_in_rect(measurement_rects[i], (x,y)):
                    logger.info(f'\nMaximum temp: {np.amax(vals[len(measurement_contours) + i])}\nMinimum temp: {np.amin(vals[len(measurement_contours) + i])}\nAvg: {np.average(vals[len(measurement_contours) + i])}\n' ) # vals stores free hand values first, and then rects; hence the 'len(measurement_contours) + i'

            for i in range( 0, len(spot_contours)):
                if cv.pointPolygonTest(spot_contours[i], (x,y), False) == 1:
                    logger.info(f'\nMaximum temp: {np.amax(spot_vals[i])}\nMinimum temp: {np.amin(spot_vals[i])}\nAvg: {np.average(spot_vals[i])}\n' )
    
        elif event == cv.EVENT_MBUTTONDOWN:
            CFlir.xdisp = x
            CFlir.ydisp = y 


    @classmethod
    def get_contours(cls, thermal_image, contours,is_rect=False):
        temp_image = thermal_image.copy()
        point1, point2 = [[]],[[]]
        cv.namedWindow('image')
        cv.setMouseCallback('image', cls.draw_contour_area, (temp_image, contours, [is_rect, point1, point2]) )
        
        while(1):
            cv.imshow('image', temp_image)
            if is_rect:
                if len(point1[0])>0 and len(point2[0])>0:
                    temp_image  = cv.rectangle(thermal_image.copy(), point1[0], point2[0], (0,0,255))
            k=cv.waitKey(1) & 0xFF

            if k==13 or k==141:
                redraw=None
                if is_rect is True and (len(point1[0])==0 or len(point2[0])==0):
                    logger.warning('No rectangle has been drawn. Do you want to continue?')
                    redraw = input('1-Yes\t0-No,draw rectangle again\n')

                if redraw is not None and redraw == 0:
                    logger.info('Draw a rectangle')
                else:
                    if is_rect is True and redraw is not None:
                        logger.warning('Exiting function without drawing a rectangle')
                        is_rect = False
                    break
        cv.destroyWindow('image')   
        if is_rect:
            area_rect = point1[0][0], point1[0][1], abs(point1[0][0] - point2[0][0]), abs(point1[0][1] - point2[0][1])    
            return area_rect
        else:
            return None

    @staticmethod
    def get_roi(thermal_image, thermal_np, raw_thermal_np, Contours, index, area_rect = None ):
        raw_roi_values = []
        thermal_roi_values = []
        indices = []
        if area_rect is None:
            img2 = np.zeros( (thermal_image.shape[0], thermal_image.shape[1],1), np.uint8)
            cv.drawContours(img2 , Contours , index, 255 , -1 )
            x,y,w,h = cv.boundingRect(Contours[index])

            indices = np.arange(w*h)
            ind = np.where(img2[:, :, 0] == 255)
            indices = indices[np.where(img2[y:y+h,x:x+w,0].flatten() == 255)]
            raw_roi_values = raw_thermal_np[ind]
            thermal_roi_values = thermal_np[ind]

        else:
            x,y,w,h =  area_rect
            raw_roi_values = raw_thermal_np[y:y+h, x:x+w]
            thermal_roi_values = thermal_np[y:y+h, x:x+w]

        return raw_roi_values, thermal_roi_values, indices


    @staticmethod
    def scale_with_roi(thermal_np, thermal_roi_values):
        temp_array = thermal_np.copy()

        roi_values = thermal_roi_values.copy()
        maximum = np.amax(roi_values)
        minimum = np.amin(roi_values)
        #opt = int(input(f'Temp difference in selected area: {temp_diff}C. Proceed with scaling? 1-Yes 0-No: ' ))
        opt=1
        if opt==1:
            #print(f'New maximum Temp: {maximum}',f'\nNew minimum Temp: {minimum}\n')
            temp_array[temp_array>maximum] = maximum
            temp_array[temp_array<minimum] = minimum
        else:
            logger.warning('Returning unscaled temperature image')
        return temp_array

    def get_measurement_contours(self, image, is_rect=False):
        CFlir.contour = []
        img = image.copy()
        area_rect = CFlir.get_contours(img, self.measurement_contours, is_rect=is_rect)
        if area_rect is not None:
            self.measurement_rects.append(area_rect)

    def get_measurement_areas_values(self, image, thermal_np, raw_thermal_np, is_rect=False):
        measurement_areas_thermal_values = []
        measurement_area_indices = []
        
        for i in range( 0,len(self.measurement_contours)):
            raw_vals, thermal_vals, indices = CFlir.get_roi(image, thermal_np, raw_thermal_np, self.measurement_contours, i)
            measurement_areas_thermal_values.append(thermal_vals)
            measurement_area_indices.append(indices)
        
        # measurement_area_indices = None
        for i in range(0, len(self.measurement_rects)):
            measurement_areas_thermal_values.append(CFlir.get_roi(image, thermal_np, raw_thermal_np, self.measurement_contours, i, area_rect=self.measurement_rects[i])[1] )
        return measurement_areas_thermal_values, measurement_area_indices


    def get_scaled_image(self, img, thermal_np, raw_thermal_np, cmap=cv.COLORMAP_JET, is_rect=False ) :
        self.scale_contours = []
        CFlir.contour=[]
        CFlir.get_contours(img, self.scale_contours)
        flag = False
        
        if len (self.scale_contours) > 0:
            
            if len(self.scale_contours[0]) > 15:
                flag = True
                thermal_roi_values = CFlir.get_roi(img, thermal_np, raw_thermal_np, self.scale_contours, 0)[1]
                temp_scaled = CFlir.scale_with_roi(thermal_np, thermal_roi_values)
                temp_scaled_image = CFlir.get_temp_image(temp_scaled, colormap=cmap)

        if flag == False:
            temp_scaled = thermal_np.copy()
            temp_scaled_image = CFlir.get_temp_image(temp_scaled, colormap=cmap)

        return temp_scaled , temp_scaled_image


    def default_scaling_image(self, array, cmap=cv.COLORMAP_JET):
        thermal_np = array.copy()
        mid_thermal_np = thermal_np[ 10:thermal_np.shape[0]-10 , (int)(thermal_np.shape[1]/2)]
        maximum = np.amax(mid_thermal_np)
        minimum = np.amin(mid_thermal_np)

        thermal_np[thermal_np>maximum+10] = maximum+10
        thermal_np[thermal_np<minimum-5] = minimum-5
        image = CFlir.get_temp_image(thermal_np, colormap=cmap)
        
        return image, thermal_np
        
    def save_thermal_image(self, output_path):
        cv.imwrite(output_path, self.thermal_image)


def get_thermal_image_from_file(thermal_input, thermal_class=CFlir, colormap=None):
    '''
    Function to get the image associated with each RJPG file using the FLIR Thermal base class CFlir
    Saves the thermal images in the same place as the original RJPG
    '''
    CThermal = thermal_class

    inputpath = Path(thermal_input)
    if Path.is_dir(inputpath):
        rjpg_img_paths = list(Path(input_folder).glob('*R.JPG'))
        fff_file_paths = list(Path(input_folder).glob('*.fff'))
        if len(rjpg_img_paths)>0:
            for rjpg_img in tqdm(rjpg_img_paths, total=len(rjpg_img_paths)):
                thermal_obj = CThermal(rjpg_img, color_map=colormap)
                path_wo_ext = str(rjpg_img).replace('_R'+rjpg_img.suffix,'') 
                thermal_obj.save_thermal_image(path_wo_ext+'.jpg')

        elif len(fff_file_paths)>0:
            for fff in tqdm(fff_file_paths, total=len(fff_file_paths)):
                save_image_path = str(fff).replace('.fff','.jpg')
                thermal_obj = CThermal(fff, color_map=colormap)
                thermal_obj.save_thermal_image(save_image_path)
        else:
            logger.error('Input folder contains neither fff or RJPG files')

    elif Path.is_file(inputpath):
        thermal_obj = CThermal(thermal_input, color_map=colormap)
        path_wo_ext = Path.as_posix(inputpath).replace(inputpath.suffix,'')
        thermal_obj.save_thermal_image(path_wo_ext+'.jpg')

    else:
        logger.error('Path given is neither file nor folder. Please check')


class CSeqVideo():
    '''
    Base class for splitting SEQ files into multiple fff and jpg files
    refer: https://exiftool.org/forum/index.php?topic=5279.0
    @purpose:
      Read .seq files from Flir IR camera and write each frame to temporary binary file.
    
    @usage:
      seqToBin.py _FILE_NAME_.seq
    
    @note:
      When first using this code for a new camera, it might need find the bits separating
      each frame, which is possibly IR camera specific. Please run:
        hexdump -n16 -C _FILE_NAME_.seq 
    
      @@Example
        >$ hexdump -n16 -C Rec-000667_test.seq 
        00000000  46 46 46 00 52 65 73 65  61 72 63 68 49 52 00 00  |FFF.ResearchIR..|
        00000010
        So, for this camera, the separation patten is:
        \x46\x46\x46\x00\x52\x65\x73\x65\x61\x72\x63\x68\x49\x52
        which == FFFResearchIR
    
    P.S. Runs much faster when writing data to an empty folder rather than rewriting existing folder's files
    '''

    def __init__(self, input_video):
        self.split_thermal(input_video)
        
    def get_hex_sep_pattern(self,input_video):
        '''
        Function to get the hex separation pattern from the seq file automatically. 
        The split, and replace functions might have to be modified. This hasn't been tried with files other than from the Zenmuse XT2
        Information on '\\x':
        https://stackoverflow.com/questions/2672326/what-does-a-leading-x-mean-in-a-python-string-xaa
        https://www.experts-exchange.com/questions/26938912/Get-rid-of-escape-character.html
        Python eval() function:
        https://www.geeksforgeeks.org/eval-in-python
        '''
        pat = sp.check_output(['hexdump', '-n16', '-C', str(input_video)])
        pat = pat.decode('ascii')
        #Following lines are to get the marker (pattern) to the appropriate hex form
        pat  = pat.split('00000000 ')[1]
        pat = pat.split('  |')[0]
        pat = pat.replace('  ',' ')
        pat = pat.replace(' ','\\x')
        pat = f"'{pat}'"
        pat = eval(pat)  #eval is apparently risky to use. Change later
        return pat

    # def split_by_marker(f, marker = pat, block_size = 10240):
    def split_by_marker(self, f, marker = "", block_size = 10240):
        current = ''
        bolStartPos = True
        while True:
            block = f.read(block_size)
            if not block: # end-of-file
                yield marker+current
                return    
            block = block.decode('latin-1')
            # exit()
            current += block
            while True:
                markerpos = current.find(marker)
                if bolStartPos ==True:
                    current = current[markerpos +len(marker):]
                    bolStartPos = False
                    continue
                elif markerpos <0:
                    break
                else:
                    yield marker+current[:markerpos]
                    current = current[markerpos+ len(marker):]

    def split_thermal(self, input_video, output_folder=None,path_to_base_thermal_class_folder='.'):
        '''
        Splits the thermal SEQ file into separate 'fff' frames by its hex separator pattern (TO DO: Find out more about how exactly this is done)
        Inputs: 'input_video':thermal SEQ video, 'output_folder': Path to output folder (Creates folder if it doesn't exist)  
        The Threading makes all the cores run at 100%, but it gives ~x4 speed-up.
        '''

        if output_folder == None:
            output_folder = Path(input_video).with_suffix("")
        
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True)
        
        sys.path.insert(0,path_to_base_thermal_class_folder)

        idx=0
        inputname=input_video
        pat = self.get_hex_sep_pattern(input_video)
        for line in tqdm(self.split_by_marker(open(inputname, 'rb'), marker=pat)):
            outname = output_folder/f"frame_{idx}.fff"
            with open( outname ,"wb") as output_file:
                line = line.encode('latin-1')
                output_file.write( line )
            Thread(target=get_thermal_image_from_file, kwargs={"thermal_class":CFlir, "thermal_input":outname} ).start()
            idx=idx+1
            if idx % 100000 == 0:
                print(f'running index : {idx} ')
                break
        return True

    def split_visual(self,visual_video, fps, fps_ratio, output_folder='visual_frames'):
        '''
        Splits video into frames based on the actual fps, and time between frames of the thermal sequence.
        There is a sync issue where the thermal fps, and visual fps don't have an integer LCM/if LCM is v large. Have to try motion interpolation to fix this
        '''

        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True)
        vid = cv.VideoCapture(visual_video)
        skip_frames = round(fps_ratio)
        total_frames = vid.get(cv.CAP_PROP_FRAME_COUNT)
        current_frame = 0
        thermal_fps = fps * (1/fps_ratio)
        thermal_time = 1/thermal_fps
        logger.info(f'Time between frames for Thermal SEQ: {thermal_time}')
        # Uncomment below lines if you need total time of visual video
        # vid.set(cv.CAP_PROP_POS_AVI_RATIO,1)
        # total_time = vid.get(cv.CAP_PROP_POS_MSEC)
        last_save_time = -1*thermal_time #So that it saves the 0th frame
        idx=0
        while current_frame < total_frames:
            current_frame = vid.get(cv.CAP_PROP_POS_FRAMES)  
            try:
                current_time = (1/fps)*current_frame
            except:
                current_time = 0
            ret,frame = vid.read()
            if ret:
                if (current_time - last_save_time)*1000 >= ((thermal_time*1000)-5):
                    # logger.info(f'Current Time: {current_time}  Last save time: {last_save_time}')
                    cv.imwrite(str(output_folder/f"{idx}.jpg"), frame)
                    idx+=1
                    last_save_time=current_time        
        return True