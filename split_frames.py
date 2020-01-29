#!/usr/bin/python
# refer: https://exiftool.org/forum/index.php?topic=5279.0
# @purpose:
#   Read .seq files from Flir IR camera and write each frame to temporary binary file.
#
# @usage:
#   seqToBin.py _FILE_NAME_.seq
#
# @note:
#   When first using this code for a new camera, it might need find the bits separating
#   each frame, which is possibly IR camera specific. Please run:
#     hexdump -n16 -C _FILE_NAME_.seq 
#
#   @@Example
#     >$ hexdump -n16 -C Rec-000667_test.seq 
#     00000000  46 46 46 00 52 65 73 65  61 72 63 68 49 52 00 00  |FFF.ResearchIR..|
#     00000010
#     So, for this camera, the separation patten is:
#     \x46\x46\x46\x00\x52\x65\x73\x65\x61\x72\x63\x68\x49\x52
#     which == FFFResearchIR
#
# P.S. Runs much faster when writing data to an empty folder rather than rewriting existing folder's files
# from __future__ import unicode_literals
# from builtins import bytes
# from builtins import str

import sys
import os
import subprocess as sp 
from pathlib import Path
import utils
#pat=b'\x46\x46\x46\x00\x52\x65\x73\x65\x61\x72\x63\x68\x49\x52';
# pat2='\x46\x46\x46\x00\x44\x4a\x49\x00\x00\x00\x00\x00\x00\x00\x00\x00'
#pat = 'FFF.ResearchIR'
pat = ''

def get_hex_sep_pattern(input_video):
  '''
  Function to get the hex separation pattern from the seq file automatically. 
  The split, and replace functions might have to be modified. Haven't tried with files other than from the Zenmuse XT2
  Information on '\\x':
    https://stackoverflow.com/questions/2672326/what-does-a-leading-x-mean-in-a-python-string-xaa
    https://www.experts-exchange.com/questions/26938912/Get-rid-of-escape-character.html
  Python eval() function:
    https://www.geeksforgeeks.org/eval-in-python
  '''
  pat = sp.check_output(['hexdump', '-n16', '-C', '{}'.format(input_video)])
  pat = pat.decode('ascii')
  #Following lines are to get the marker (pattern) to the appropriate hex form
  pat  = pat.split('00000000 ')[1]
  pat = pat.split('  |')[0]
  pat = pat.replace('  ',' ')
  pat = pat.replace(' ','\\x')
  pat = "'" + pat + "'"
  pat = eval(pat)  #eval is apparently risky to use. Change later
  return pat

def split_by_marker(f, marker = pat, block_size = 10240):
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

def split_thermal(input_video=sys.argv[1], output_folder=sys.argv[2],path_to_base_thermal_class_folder='.'):
  '''
  Splits the thermal SEQ file into separate 'fff' frames by its hex separator pattern (TO DO: Find out more about how exactly this is done)
  Inputs: 'input_video':thermal SEQ video, 'output_folder': Path to output folder (Creates folder if it doesn't exist)  
  The Threading makes all the cores run at 100%, but it gives ~x4 speed-up.
  '''
  sys.path.insert(0,path_to_base_thermal_class_folder)
  from CThermal import CFlir

  from tqdm import tqdm
  from threading import Thread
  idx=0
  inputname=input_video
  outdir = output_folder
  if not os.path.isdir(outdir):
    os.mkdir(outdir)
  pat = get_hex_sep_pattern(input_video)
  for line in tqdm(split_by_marker(open(inputname, 'rb'), marker=pat)):
    outname=outdir + "frame_{0}.fff".format( idx )
    output_file = open( outname ,"wb")
    line = line.encode('latin-1')
    output_file.write( line )
    output_file.close()
    Thread(target= get_thermal_image_from_file, args=(outname, thermal_class=CFlir)).start()
    idx=idx+1
    if idx % 100000 == 0:
      print ('running index : {} '.format( idx ) )
      break
  return True

def split_visual(visual_video, fps, fps_ratio, output_folder='visual_frames'):
  '''
  Splits video into frames based on the actual fps, and time between frames of the thermal sequence.
  There is a sync issue where the thermal fps, and visual fps don't have an integer LCM/if LCM is v large. Have to try motion interpolation to fix this
  '''
  import cv2
  import time
  from logzero import logger, loglevel
  
  if not os.path.isdir(output_folder):
    os.mkdir(output_folder)
  vid = cv2.VideoCapture(visual_video)
  skip_frames = round(fps_ratio)
  total_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
  current_frame = 0
  thermal_fps = fps * (1/fps_ratio)
  thermal_time = 1/thermal_fps
  logger.info('Time between frames for Thermal SEQ: {}'.format(thermal_time))
  # Uncomment below lines if you need total time of visual video
  # vid.set(cv2.CAP_PROP_POS_AVI_RATIO,1)
  # total_time = vid.get(cv2.CAP_PROP_POS_MSEC)
  last_save_time = -1*thermal_time #So that it saves the 0th frame
  idx=0
  while current_frame < total_frames:
    current_frame = vid.get(cv2.CAP_PROP_POS_FRAMES)  
    try:
      current_time = (1/fps)*current_frame
    except:
      current_time = 0
    ret,frame = vid.read()
    if ret:
      if (current_time - last_save_time)*1000 >= ((thermal_time*1000)-5):
        # logger.info(f'Current Time: {current_time}  Last save time: {last_save_time}')
        cv2.imwrite(output_folder+str(idx)+'.jpg', frame)
        idx+=1
        last_save_time=current_time        
  return True

if __name__ == '__main__':
  split_thermal()
  