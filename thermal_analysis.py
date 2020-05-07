#!/usr/bin/env python3
import cv2
import sys
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import cm
import subprocess
import io
import json
import calendar
from math import sqrt,exp,log
from decimal import Decimal
try:
    import CThermal.CFlir as CFlir
except ImportError:
    from CThermal import CFlir
from logzero import logging, logger

if __name__ == "__main__":
    flir_image_path = sys.argv[1]

    cmap = cv2.COLORMAP_JET
    obj = CFlir (flir_image_path,color_map=cv2.COLORMAP_JET)
    raw_np = obj.raw_thermal_np
    original_array= obj.thermal_np
    array = original_array.copy()
    corrected_array = original_array.copy()
    default_scaled_image, default_scaled_array = obj.default_scaling_image( array, cmap )
    image = default_scaled_image.copy()
    original_default_scaled_array = default_scaled_array.copy()
    default_scale = True

    while(1):
        os.system('cls' if os.name == 'nt' else 'clear')
        logger.info('Enter option: \n1. ROI Scaling \n2. Draw Areas \n3. Draw Line \n4. Draw Spots \n5. Change Parameters \n6. Change Color Map\n7. Invert Image Scale\n8. Refresh \n9. Continue\nS. Save thermal image\n0. Exit\n')
        
        cv2.namedWindow('Enter Input',0) #Scalable Window
        cv2.resizeWindow('Enter Input', (image.shape[1],image.shape[0]))
        cv2.imshow('Enter Input',image)
        k = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

        opt = int(k) - 48
            
        os.system('cls' if os.name == 'nt' else 'clear')

        vals = []

        if opt == 1:
            logger.info('ROI Scaling')
            array, image = obj.get_scaled_image(image, array, raw_np , cmap )
            default_scaled_array = array.copy()
            # image = changed_image.copy()
   
        elif opt == 2:
            logger.info('Area Measurement Tool')
            aopt = int(input('1. Free Hand\n2. Rectangle\n'))
            if aopt==1:
                obj.get_measurement_contours(image)
            else:
                obj.get_measurement_contours(image, is_rect=True)

        elif opt == 3:
            logger.info('Line Tool')
            obj.line_measurement(image, corrected_array, cmap )


        elif opt == 4:
            obj.get_spots(image)

        elif opt == 5:
            try:
                OD
            except:
                OD = CFlir.parse_length(obj.meta['ObjectDistance'])
            
            try:
                RH
            except:
                RH = CFlir.parse_percent(obj.meta['RelativeHumidity']) 
            
            try:
                RAT 
            except:
                RAT = CFlir.parse_temp(obj.meta['ReflectedApparentTemperature'])

            try:
                AT 
            except:
                AT = CFlir.parse_temp(obj.meta['AtmosphericTemperature'])
            
            try:
                E 
            except:
                E = obj.meta['Emissivity']
            
            logger.info('\n1.Object Distance: {}\n2.Relative Humidity: {}\n3.Reflected Apparent Temperature: {}\n4.Atmospheric Temperature: {}\n5.Emissivity of image: {}\n'.format(OD,RH,RAT,AT,E))
            
            cv2.imshow('Parameter Change',image)
            k = cv2.waitKey(100000) & 0xFF
            cv2.destroyAllWindows()
            popt = None
            popt = int(k) - 48
            cv2.destroyAllWindows()
            if popt==1:
                try:
                    OD = float(input('Enter new OD\n'))
                except:
                    OD = float(input('Enter new OD again\n'))

            elif popt==2:
                try:
                    RH = float(input('Enter new RH%\n'))
                except:
                    RH = float(input('Enter new RH% \again\n'))

            elif popt==3:
                try:
                    RAT = float(input('Enter new Reflected Apparent Temperature\n'))
                except:
                    RAT = float(input('Enter new Reflected Apparent Temperature again\n'))

            elif popt==4:
                try:
                    AT = float(input('Enter new Atmospheric Temp\n'))
                except:
                    AT = float(input('Enter new Atmospheric Temp again\n'))
            
            elif popt==5:
                try:
                    E = float(input('Enter new Emissivity\n'))
                except:
                    E = float(input('Enter new Emissivity again\n'))

            else:
                logger.error('Invalid Option...No change in parameters\n')
            
            if popt>=1 and popt<=5:
                raw2tempfunc = (lambda x: CFlir.raw2temp(x, E=E, OD=OD,  RTemp=RAT, ATemp=AT, IRWTemp=CFlir.parse_temp(obj.meta['IRWindowTemperature']), IRT=obj.meta['IRWindowTransmission'], RH=RH, PR1=obj.meta['PlanckR1'], PB=obj.meta['PlanckB'], PF=obj.meta['PlanckF'], PO=obj.meta['PlanckO'], PR2=obj.meta['PlanckR2']))
                corrected_array = raw2tempfunc(raw_np)
                default_scaled_array = obj.default_scaling_image( corrected_array, cmap )[1]

        elif opt == 6:
            copt = int(input('Enter Colormap: \n1. Jet(Default)\n2. Gray(No false color map)\n3. Rainbow\n4. Hot\n'))
            
            if copt==1:
                cmap = cv2.COLORMAP_JET
            elif copt==2:
                cmap = None
            elif copt==3:
                cmap = cv2.COLORMAP_RAINBOW            
            elif copt==4:
                cmap = cv2.COLORMAP_HOT           

            if copt>0 and copt<=4:
                image = CFlir.get_temp_image(default_scaled_array, colormap=cmap)
            else:
                logger.error("\nInvalid Option\n")


        elif opt == 7:
            logger.info("Changing the scale of the image")
            if default_scale is True:
                default_scale = False
                image = obj.thermal_image.copy()
                array = original_array.copy()
                default_scaled_array = array.copy()
            else:
                default_scale = True
                image = default_scaled_image.copy()
                default_scaled_array = original_default_scaled_array.copy()
            continue

        elif opt == 8:
            array = original_array.copy()
            image = default_scaled_image.copy()
            corrected_array = original_array.copy()
            default_scaled_array = original_default_scaled_array.copy()
            obj.scale_contours.clear()
            obj.measurement_contours.clear()
            obj.measurement_rects.clear()
            obj.spots.clear()
            cmap = cv2.COLORMAP_JET
            continue
        
        elif opt==9:
            logger.warning('Continuing Without Change')
        
        elif opt==115-48:
            logger.info('Saving image')
            cv2.imwrite(flir_image_path.split('.')[0]+'_formatted.jpg', image) #Change to class object function call

        elif opt==0 or opt==65 or opt==110: #65+18=113, which is `q`
            logger.warning('Exiting...')
            exit(0)
            
        vals, measurement_indices = obj.get_measurement_areas_values(image, corrected_array, raw_np)  
        spot_vals = obj.get_spots_values(image, corrected_array, raw_np, obj.spots) 

        cv2.namedWindow('Main Window', 0)
        cv2.resizeWindow('Main Window', (image.shape[1], image.shape[0]))
        cv2.setMouseCallback('Main Window', CFlir.move_contours, ( obj.measurement_contours, obj.measurement_rects, obj.scale_contours, obj.spots, image, vals, spot_vals ) )
        
        original_image = image.copy()

        temp_min, temp_max = round(np.amin(default_scaled_array),2), round(np.amax(default_scaled_array),2)

        while(1):
            append_img = obj.generate_colorbar(temp_min, temp_max, cmap)
            
            if len(image.shape)==2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            image = np.concatenate( (image, append_img), axis=1)

            if len(vals)>0:
                for i in range(0,len(vals)):
                    vals[i] = obj.get_measurement_areas_values(image, corrected_array, raw_np)[0][i] # list assignment will have to be done this way so that 'vals' remains the same list which is passed to the mouse callback
            
            if len(spot_vals)>0:
                for i in range(0, len(spot_vals)):
                    spot_vals[i] = obj.get_spots_values(image, corrected_array, raw_np, obj.spots)[i]
            
            if len(obj.scale_contours) > 0:
                cv2.drawContours(image, obj.scale_contours,-1, (0,0,0), 1 , 8)            
            
            if len(obj.measurement_contours) > 0:
                cv2.drawContours(image, obj.measurement_contours, -1, (0,0,255), 1, 8 )

            if len(obj.measurement_rects) > 0:
                for i in range(len(obj.measurement_rects)):
                    cv2.rectangle(image, obj.measurement_rects[i], (0,0,255))

            if len(obj.spots) > 0:
                cv2.drawContours(image, obj.spots, -1, (255,255,255), -1)

            if CFlir.xdisp != None and CFlir.ydisp != None:
                temp = Decimal(corrected_array[CFlir.ydisp][CFlir.xdisp])
                temp = round(temp,2)
                cv2.putText(image, str(temp)+'C', (CFlir.xdisp,CFlir.ydisp), cv2.FONT_HERSHEY_PLAIN, 1, 0, 2, 8  )

            cv2.imshow('Main Window',image)

            if len(obj.scale_contours) > 0 and len(obj.scale_contours[0]) > 15:
                roi_vals = CFlir.get_roi(image, corrected_array, raw_np, obj.scale_contours, 0 )[1]            
                scaled = CFlir.scale_with_roi(corrected_array , roi_vals)     
                image = CFlir.get_temp_image(scaled, colormap=cmap)
                temp_min, temp_max = round(np.amin(roi_vals),2), round(np.amax(roi_vals),2)
            else:
                image = original_image.copy()
                

            k=cv2.waitKey(1) & 0xFF
        
            if k==13 or k==141:
                break
            
        cv2.destroyWindow('Main Window')   