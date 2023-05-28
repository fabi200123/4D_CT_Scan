import os,sys
sys.path.append("./precision-medicine-toolbox/")
from pmtool.ToolBox import ToolBox
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

'''
How to run:
$ python get_data_in_mongo.py PACIENT_FOLDER
'''
# Arguments
name_of_pacient = sys.argv[1]

parameters = {'data_path': r"C:\\Users\\fabi2\\OneDrive\\Desktop\\Betty's idea of doing shit\\" + name_of_pacient + "\\dcms\\", # path to your DICOM data
              'data_type': 'dcm', # original data format: DICOM
              'multi_rts_per_pat': True}   # when False, it will look only for 1 rtstruct in the patient folder, 
                                            # this will speed up the process, 
                                            # if you have more then 1 rtstruct per patient, set it to True

export_path =r"C:\\Users\\fabi2\\OneDrive\\Desktop\\Betty's idea of doing shit\\" + name_of_pacient + "\\" # the function will create 'converted_nrrd' folder in the specified directory
data_ct = ToolBox(**parameters)
data_ct.convert_to_nrrd(export_path, 'gtv')

# Generate the 2D PNG files of the nodule
data_path = "C:\\Users\\fabi2\\OneDrive\\Desktop\\Betty's idea of doing shit\\" + name_of_pacient + "\\converted_nrrds\\"
data_ct_nrrd = ToolBox(data_path, data_type='nrrd')
data_ct_nrrd.get_jpegs(r"C:\\Users\\fabi2\\OneDrive\\Desktop\\Betty's idea of doing shit\\" + name_of_pacient + "\\") # the function will create 'images_quick_check' folder in the specified directory 