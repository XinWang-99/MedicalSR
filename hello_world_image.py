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
import cv2

ORI_MR = []
OUTPUT_MR = []
VIS_INPUT_MR = []
VIS_TARGET_MR = []
VIS_TEXT = []

def resize_image_itk(ori_img, target_size, target_spacing, resamplemethod=sitk.sitkNearestNeighbor):
    target_origin = ori_img.GetOrigin()  # 目标的起点 [x,y,z]
    target_direction = ori_img.GetDirection()  # 目标的方向 [冠,矢,横]=[z,y,x]

    # itk的方法进行resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ori_img)  # 需要重新采样的目标图像
    # 设置目标图像的信息
    resampler.SetSize(target_size)  # 目标图像大小
    resampler.SetOutputOrigin(target_origin)
    resampler.SetOutputDirection(target_direction)
    resampler.SetOutputSpacing(target_spacing)
    # 根据需要重采样图像的情况设置不同的dype
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itk_img_resampled = resampler.Execute(ori_img)  # 得到重新采样后的图像
    return itk_img_resampled

def sepia(input_img, Position):

    # Loading a nifti in the traditional sense
    sr_path=test(input_img.name)

    # return sr_path
    ORI_MR.append(input_img.name)
    OUTPUT_MR.append(sr_path)

    return sr_path

def visualize(slice_num_axial, slice_num_sagittal, slice_num_coronal):
    
    if len(VIS_INPUT_MR) == 0:
        img = sitk.ReadImage(ORI_MR[-1])
        ori_spacing = img.GetSpacing()

        out_img = sitk.ReadImage(OUTPUT_MR[-1])
        out_data = sitk.GetArrayFromImage(out_img)
        out_size = out_img.GetSize()
        out_spacing = out_img.GetSpacing()

        img = resize_image_itk(img, out_size, out_spacing)
        data = sitk.GetArrayFromImage(img)
        out_text = f'original spacing: {ori_spacing[0]}mm×{ori_spacing[1]}mm×{ori_spacing[2]}mm, reconstructed spacing: {out_spacing[0]}mm×{out_spacing[1]}mm×{out_spacing[2]}mm'

        VIS_INPUT_MR.append(data)
        VIS_TARGET_MR.append(out_data)
        VIS_TEXT.append(out_text)
    else:
        data = VIS_INPUT_MR[-1]
        out_data = VIS_TARGET_MR[-1]
        out_text = VIS_TEXT[-1]

    data = (data - data.min()) / (data.max() - data.min()) * 255
    data = data.astype('uint8')[::-1,...]
    in_slice_axial = data[int(data.shape[0]*slice_num_axial)]
    in_slice_sagittal = data[:,int(data.shape[1]*slice_num_sagittal),:]
    in_slice_coronal = data[:,:,int(data.shape[2]*slice_num_coronal)]
    
    out_data = (out_data - out_data.min()) / (out_data.max() - out_data.min()) * 255
    out_data = out_data.astype('uint8')[::-1,...]
    out_slice_axial = out_data[int(out_data.shape[0]*slice_num_axial)]
    out_slice_sagittal = out_data[:,int(out_data.shape[1]*slice_num_sagittal),:]
    out_slice_coronal = out_data[:,:,int(out_data.shape[2]*slice_num_coronal)]
    

    return out_text, in_slice_axial, in_slice_sagittal, in_slice_coronal, out_slice_axial, out_slice_sagittal, out_slice_coronal

'''
# block 1: output the file
input_file = gr.File(file_types=['.nii.gz'])
input_option = gr.Radio(['Knee', 'Brain', 'Cardiac'])
output_file = gr.File(label="Reconstructed MRI")
block_1 = gr.Interface(sepia, [input_file, input_option], output_file, live=False, title="Isotropic Super-resolution of MR images",examples=[['examples/sub085_4_T1_LR.nii.gz']])

# block 2: visualization
input_slider_axial = gr.inputs.Slider(minimum=0, maximum=1, step=0.01, default=0.5, label="Select a relative axial position:")
input_slider_sagittal = gr.inputs.Slider(minimum=0, maximum=1, step=0.01, default=0.5, label="Select a relative sagittal position:")
input_slider_coronal = gr.inputs.Slider(minimum=0, maximum=1, step=0.01, default=0.5, label="Select a relative coronal position:")
output_img_axial = gr.Image()
output_img_sagittal = gr.Image()
output_img_coronal = gr.Image()
output_img_axial_1 = gr.Image()
output_img_sagittal_1 = gr.Image()
output_img_coronal_1 = gr.Image()
block_2 = gr.Interface(visualize, [input_slider_axial, input_slider_sagittal, input_slider_coronal], 
                       [output_img_axial, output_img_sagittal, output_img_coronal, output_img_axial_1, output_img_sagittal_1, output_img_coronal_1], live=True, title="Visualization of MR images")
'''


block = gr.Blocks()
with block:
    with gr.Tabs():
        with gr.TabItem("Inference"):
            with gr.Row():
                input_file = gr.File(file_types=['.nii.gz'])
                input_option = gr.Radio(['Knee', 'Brain', 'Cardiac'])
                b1 = gr.Button("Generate")
                output_file = gr.File(label="Reconstructed MRI")
                b1.click(sepia,
                           inputs=[input_file, input_option],
                           outputs=output_file)
            with gr.Column():
                input_slider_axial = gr.inputs.Slider(minimum=0, maximum=1, step=0.01, default=0.5, label="Select a relative axial position:")
                input_slider_sagittal = gr.inputs.Slider(minimum=0, maximum=1, step=0.01, default=0.5, label="Select a relative sagittal position:")
                input_slider_coronal = gr.inputs.Slider(minimum=0, maximum=1, step=0.01, default=0.5, label="Select a relative coronal position:")
                out_text = gr.Textbox(show_label=False)
            with gr.Row():
                output_img_axial = gr.Image(label='original')
                output_img_sagittal = gr.Image(show_label=False)
                output_img_coronal = gr.Image(show_label=False)
            with gr.Row():
                output_img_axial_1 = gr.Image(label='reconstructed')
                output_img_sagittal_1 = gr.Image(show_label=False)
                output_img_coronal_1 = gr.Image(show_label=False)

                input_slider_axial.change(visualize,
                        inputs=[input_slider_axial, input_slider_sagittal, input_slider_coronal],
                        outputs=[out_text, output_img_axial, output_img_sagittal, output_img_coronal, output_img_axial_1, output_img_sagittal_1, output_img_coronal_1])
                input_slider_sagittal.change(visualize,
                        inputs=[input_slider_axial, input_slider_sagittal, input_slider_coronal],
                        outputs=[out_text, output_img_axial, output_img_sagittal, output_img_coronal, output_img_axial_1, output_img_sagittal_1, output_img_coronal_1])
                input_slider_coronal.change(visualize,
                        inputs=[input_slider_axial, input_slider_sagittal, input_slider_coronal],
                        outputs=[out_text, output_img_axial, output_img_sagittal, output_img_coronal, output_img_axial_1, output_img_sagittal_1, output_img_coronal_1])


# demo = gr.Interface(sepia, [gr.File(file_types=['.nii.gz'], label="Input MRI"), gr.Radio(['Knee', 'Brain', 'Cardiac']), gr.inputs.Slider(minimum=0, maximum=10, step=0.1, default=5, label="Select a slice:")], [gr.File(label="Reconstructed MRI"), gr.Image(size=(128,128))], live=True, 
#     title="Isotropic Super-resolution of MR images",examples=[['examples/sub085_4_T1_LR.nii.gz']])

# layout = [[block_1], [block_2]]
# demo = gr.Interface(layout=layout)

block.launch(share=False)
