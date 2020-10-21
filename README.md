# ThermImageAnalysis
* `CThermal.py` contains classes for sensor value-->temperature conversion from FLIR files (SEQ, images), and the analysis tools for the same, and for splitting SEQ videos
* `thermal_analysys.py` contains the main code to be run for the analysis tools on the FLIR files.
* `split_seq.py` splits an SEQ file into multiple FFF files


## Usage
* First, install the requirements for using the code in this repo using `pip install -r requirements.txt`
* For Linux users, check if exiftool is installed, by running ```exiftool``` in the command line.
* For Windows users, do the following: 
    * Download exiftool (for your OS) from [here](https://exiftool.org/)
    * Extract 'exiftool(-k).exe' to the folder where you're running the code from
    * Rename it to 'exiftool.exe'
* To run thermal analysis on a FLIR image (FFF/jpg/rjpg/...), run the following command in a terminal:
    ```bash
    python thermal_analysis.py <file name>
    ```  
  The usage instructions and other notes for this are in the [wiki page](https://github.com/detecttechnologies/Thermal_Image_Analysis/wiki/Instructions-Manual-for-the-Thermal-Image-Analysis-software)
* To extract FFF(RJPG header that can be used by `thermal_analysis.py` of this repo) and JPG files from an SEQ video, you can run the following command:
    ```bash
    python split_seq.py <SEQ file name>
    ```


## Features
1. **ROI Scaling** - Draw a (freehand) Region of Interest area to scale the rest of the image with. This is useful in cases where the region of your interest is low in contrast compared to the scale of the entire image. This drawn area can be moved around to change the region

2. **Area Measurement** - Draw a rectangle, or freehand area(s), to get the *average, minimum, and maximum* temperatures of that area. These can be moved around as well.

3. **Line Tool** - Draw a line to get a plot (temp vs pixel distance) of the temperatures along the points.

4. **Spot Measurement** - Draw spots(circular areas with a small radius). Similar to 'Area Measurement'

5. **Change Image Parameters** - Option to change the global parameters: *Object Distance, Relative Humidity, Reflected Apparent Temperature, Atmospheric Temperature, Emissivity* of the image. The default values are obtained from the metadatawiki

6. **Change Color Map** - Change the color map representation of the thermal data (Default-Jet). Options available are: *Gray* *(No false colormap)*, *Rainbow*, and *Hot*

7. **Invert Scaling** - Change the way the way the image is scaled between the default scaling, and the raw image.

## To-Do

* General Interface changes for easier use 
* Draw multiple areas in the same go
* Line visualization while drawing from the 'Draw Line' tool
* Change free hand to polygon 
