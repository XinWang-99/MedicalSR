"""

Hello world nifti

"""
import sys
sys.path.append('SAINR')
import numpy as np
import SimpleITK as sitk
import gradio as gr
from pathlib import Path
from slice_test import test

def sepia(input_img, Position):

    # Loading a nifti in the traditional sense
    sr_path=test(input_img.name)
    return sr_path
    


demo = gr.Interface(sepia, [gr.File(file_types=['.nii.gz'], label="Input MRI"), gr.Radio(['Knee', 'Brain', 'Cardiac'])], [gr.File(label="Reconstructed MRI")], live=False, 
    title="Isotropic Super-resolution of MR images",examples=[['examples/sub085_4_T1_LR.nii.gz']])

demo.launch(share=False)
