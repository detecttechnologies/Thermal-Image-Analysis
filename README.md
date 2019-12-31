# Codes related to thermal analysis

* `CProcessThermalImage.py` is a master class that was built for the old ThermApp camera. Modify it to fix all the functions for the FLIR Camera  if needed
* `CThermal.py` is the master class for getting the temperature, and ROI scalling for FLIR and Thermapp images

## Usage

* `python CThermal.py -i <image_path> -c <camera type (Flir or Thermapp)>` 

## Updates 

* Added functionality for FLIR camera
* Center Scaling for CProcessThermalImage
* Program to extract temp values, roi scaling from thermapp and FLIR images


## To-Do

* [] Add ability to press and hold mouse in view-pixel-temperature mode (middle click?)  in thermal_analysis.py