#!/usr/bin/env python3
import os
import sys
from decimal import Decimal
from pathlib import Path

import cv2 as cv
import numpy as np
from logzero import logger, logging
from PyInquirer import prompt

if __name__ == "__main__":
    from CThermal import CFlir
else:
    from .CThermal import CFlir


def flush_input():
    try:
        import msvcrt

        while msvcrt.kbhit():
            msvcrt.getch()
    except ImportError:
        import sys, termios  # for linux/unix

        termios.tcflush(sys.stdin, termios.TCIOFLUSH)


if __name__ == "__main__":
    input_image_path = sys.argv[1]

    cmap = cv.COLORMAP_JET
    therm_obj = CFlir(input_image_path, color_map=cv.COLORMAP_JET)
    raw_np = therm_obj.raw_thermal_np
    original_array = therm_obj.thermal_np
    array = original_array.copy()
    corrected_array = original_array.copy()
    default_scaled_image, default_scaled_array = therm_obj.default_scaling_image(array, cmap)
    image = default_scaled_image.copy()
    original_default_scaled_array = default_scaled_array.copy()
    default_scale = True

    OD, RH, RAT, AT, E = [None] * 5

    while True:
        os.system("cls" if os.name == "nt" else "clear")
        flush_input()
        cv.namedWindow("Main Window", 0)  # Scalable Window
        cv.imshow("Main Window", image)
        questions = [
            {
                "type": "list",
                "name": "main_operation",
                "message": "What operation would you like to perform",
                "choices": [
                    "ROI Temperature Scaling",
                    "Area Highlighting Tool",
                    "Line Measurement Tool",
                    "Spot Measurement Tool",
                    "Change Parameters",
                    "Change Color Map",
                    "Invert Image Scale",
                    "Reset",
                    "Continue",
                    "Save thermal image",
                    "Exit",
                ],
            }
        ]

        cv.waitKey(1)
        opt = prompt(questions)["main_operation"]
        cv.destroyAllWindows()

        os.system("cls" if os.name == "nt" else "clear")

        vals = []

        logger.info(f"Selected {opt}")
        if opt == "ROI Temperature Scaling":
            logger.info("Draw a region in the image, then hit Enter")
            array, image = therm_obj.get_scaled_image(image, array, raw_np, cmap)
            default_scaled_array = array.copy()
            # image = changed_image.copy()

        elif opt == "Area Highlighting Tool":
            questions_2 = [
                {
                    "type": "list",
                    "name": "area_type",
                    "message": "What type of area would you like to draw?",
                    "choices": ["Free Hand", "Rectangle"],
                }
            ]
            aopt = prompt(questions_2)["area_type"]
            logger.info(f"Draw a region in the image, then hit Enter")
            if aopt == "Free Hand":
                therm_obj.get_measurement_contours(image)
            elif aopt == "Rectangle":
                therm_obj.get_measurement_contours(image, is_rect=True)

        elif opt == "Line Measurement Tool":
            logger.info(
                "Please draw the extremes of the line along which you would like to measure temperature, then hit Enter"
            )
            therm_obj.line_measurement(image, corrected_array, cmap)

        elif opt == "Spot Measurement Tool":
            logger.info("Please select the points where you would like to measure temperature")
            therm_obj.get_spots(image)

        elif opt == "Change Parameters":
            if OD is None:
                OD = CFlir.parse_length(therm_obj.meta["ObjectDistance"])
            if RH is None:
                RH = CFlir.parse_percent(therm_obj.meta["RelativeHumidity"])
            if RAT is None:
                RAT = CFlir.parse_temp(therm_obj.meta["ReflectedApparentTemperature"])
            if AT is None:
                AT = CFlir.parse_temp(therm_obj.meta["AtmosphericTemperature"])
            if E is None:
                E = therm_obj.meta["Emissivity"]

            cv.imshow("Parameter Change", image)
            cv.waitKey(1)

            logger.info(
                f"""Existing values of parameters:
                    1.Object Distance: {OD}
                    2.Relative Humidity: {RH}
                    3.Reflected Apparent Temperature: {RAT}
                    4.Atmospheric Temperature: {AT}
                    5.Emissivity of image: {E}"""
            )
            logger.info("Please press enter new values:")

            def is_float(value):
                try:
                    float(value)
                    return True
                except ValueError:
                    return False

            questions_2 = [
                {
                    "type": "input",
                    "name": "OD",
                    "message": "Object Distance",
                    "default": str(OD),
                    "validate": is_float,
                    "filter": lambda val: float(val),
                },
                {
                    "type": "input",
                    "name": "RH",
                    "message": "Relative Humidity",
                    "default": str(RH),
                    "validate": is_float,
                    "filter": lambda val: float(val),
                },
                {
                    "type": "input",
                    "name": "RAT",
                    "message": "Reflected Apparent Temperature",
                    "default": str(RAT),
                    "validate": is_float,
                    "filter": lambda val: float(val),
                },
                {
                    "type": "input",
                    "name": "AT",
                    "message": "Atmospheric Temperature",
                    "default": str(AT),
                    "validate": is_float,
                    "filter": lambda val: float(val),
                },
                {
                    "type": "input",
                    "name": "E",
                    "message": "Emissivity of image",
                    "default": str(E),
                    "validate": is_float,
                    "filter": lambda val: float(val),
                },
            ]

            popt = prompt(questions_2)
            OD, RH, RAT, AT, E = popt["OD"], popt["RH"], popt["RAT"], popt["AT"], popt["E"]
            cv.destroyAllWindows()

            raw2tempfunc = lambda x: CFlir.raw2temp(
                x,
                E=E,
                OD=OD,
                RTemp=RAT,
                ATemp=AT,
                IRWTemp=CFlir.parse_temp(therm_obj.meta["IRWindowTemperature"]),
                IRT=therm_obj.meta["IRWindowTransmission"],
                RH=RH,
                PR1=therm_obj.meta["PlanckR1"],
                PB=therm_obj.meta["PlanckB"],
                PF=therm_obj.meta["PlanckF"],
                PO=therm_obj.meta["PlanckO"],
                PR2=therm_obj.meta["PlanckR2"],
            )
            corrected_array = raw2tempfunc(raw_np)
            default_scaled_array = therm_obj.default_scaling_image(corrected_array, cmap)[1]

        elif opt == "Change Color Map":
            questions_2 = [
                {
                    "type": "list",
                    "name": "cmap",
                    "message": "What Colourmap would you like to use?",
                    "choices": ["Jet(Default)", "Gray(No false color map)", "Rainbow", "Hot"],
                }
            ]
            copt = prompt(questions_2)["cmap"]

            if copt == "Jet(Default)":
                cmap = cv.COLORMAP_JET
            elif copt == "Gray(No false color map)":
                cmap = None
            elif copt == "Rainbow":
                cmap = cv.COLORMAP_RAINBOW
            elif copt == "Hot":
                cmap = cv.COLORMAP_HOT

            image = CFlir.get_temp_image(default_scaled_array, colormap=cmap)

        elif opt == "Invert Image Scale":
            logger.info("Changing the scale of the image")
            if default_scale is True:
                default_scale = False
                image = therm_obj.thermal_image.copy()
                array = original_array.copy()
                default_scaled_array = array.copy()
            else:
                default_scale = True
                image = default_scaled_image.copy()
                default_scaled_array = original_default_scaled_array.copy()
            continue

        elif opt == "Reset":
            array = original_array.copy()
            image = default_scaled_image.copy()
            corrected_array = original_array.copy()
            default_scaled_array = original_default_scaled_array.copy()
            therm_obj.scale_contours.clear()
            therm_obj.measurement_contours.clear()
            therm_obj.measurement_rects.clear()
            therm_obj.spots.clear()
            cmap = cv.COLORMAP_JET
            continue

        elif opt == "Continue":
            logger.warning("Continuing Without Change")

        elif opt == "Save thermal image":
            logger.info("Saving image")
            cv.imwrite(f"{Path(input_image_path).name}_formatted.jpg", image)

        elif opt == "Exit":
            logger.warning("Exiting...")
            break

        vals, measurement_indices = therm_obj.get_measurement_areas_values(image, corrected_array, raw_np)
        spot_vals = therm_obj.get_spots_values(image, corrected_array, raw_np, therm_obj.spots)

        cv.namedWindow("Main Window", 0)
        cv.resizeWindow("Main Window", (image.shape[1], image.shape[0]))
        cv.setMouseCallback(
            "Main Window",
            CFlir.move_contours,
            (
                therm_obj.measurement_contours,
                therm_obj.measurement_rects,
                therm_obj.scale_contours,
                therm_obj.spots,
                image,
                vals,
                spot_vals,
            ),
        )

        original_image = image.copy()

        temp_min, temp_max = round(np.amin(default_scaled_array), 2), round(np.amax(default_scaled_array), 2)

        while True:
            append_img = therm_obj.generate_colorbar(temp_min, temp_max, cmap)

            if len(image.shape) == 2:
                image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
            image = np.concatenate((image, append_img), axis=1)

            if len(vals) > 0:
                for i in range(0, len(vals)):
                    vals[i] = therm_obj.get_measurement_areas_values(image, corrected_array, raw_np)[0][
                        i
                    ]  # list assignment will have to be done this way so that 'vals' remains the same list which is passed to the mouse callback

            if len(spot_vals) > 0:
                for i in range(0, len(spot_vals)):
                    spot_vals[i] = therm_obj.get_spots_values(image, corrected_array, raw_np, therm_obj.spots)[i]

            if len(therm_obj.scale_contours) > 0:
                cv.drawContours(image, therm_obj.scale_contours, -1, (0, 0, 0), 1, 8)

            if len(therm_obj.measurement_contours) > 0:
                cv.drawContours(image, therm_obj.measurement_contours, -1, (0, 0, 255), 1, 8)

            if len(therm_obj.measurement_rects) > 0:
                for i in range(len(therm_obj.measurement_rects)):
                    cv.rectangle(image, therm_obj.measurement_rects[i], (0, 0, 255))

            if len(therm_obj.spots) > 0:
                cv.drawContours(image, therm_obj.spots, -1, (255, 255, 255), -1)

            if CFlir.xdisp != None and CFlir.ydisp != None:
                temp = Decimal(corrected_array[CFlir.ydisp][CFlir.xdisp])
                temp = round(temp, 2)
                cv.putText(image, str(temp) + "C", (CFlir.xdisp, CFlir.ydisp), cv.FONT_HERSHEY_PLAIN, 1, 0, 2, 8)

            cv.imshow("Main Window", image)

            if len(therm_obj.scale_contours) > 0 and len(therm_obj.scale_contours[0]) > 15:
                roi_vals = CFlir.get_roi(image, corrected_array, raw_np, therm_obj.scale_contours, 0)[1]
                scaled = CFlir.scale_with_roi(corrected_array, roi_vals)
                image = CFlir.get_temp_image(scaled, colormap=cmap)
                temp_min, temp_max = round(np.amin(roi_vals), 2), round(np.amax(roi_vals), 2)
            else:
                image = original_image.copy()

            k = cv.waitKey(1) & 0xFF

            if k == 13 or k == 141:
                break

        cv.destroyWindow("Main Window")
