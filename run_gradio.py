import gradio
import torch
from time import  strftime
import os, sys, time
from argparse import ArgumentParser

from preprocess import CropAndExtract
from test_audio2coeff import Audio2Coeff  
from facerender.animate import AnimateFromCoeff
from generate_batch import get_data
from generate_facerender_batch import get_facerender_data


def synthesize(audio, image, video, pose_style, batch_size):
    pic_path = video if image is None else image
    audio_path = audio
    save_dir = os.path.join("results", strftime("%Y_%m_%d_%H.%M.%S"))
    os.makedirs(save_dir, exist_ok=True)
    device = "cuda"
    camera_yaw_list = [0]
    camera_pitch_list = [0]
    camera_roll_list = [0]
    batch_size = int(batch_size)

    current_code_path = sys.argv[0]
    current_root_path = os.path.split(current_code_path)[0]

    os.environ['TORCH_HOME']=os.path.join(current_root_path, 'checkpoints')

    path_of_lm_croper = os.path.join(current_root_path, 'checkpoints', 'shape_predictor_68_face_landmarks.dat')
    path_of_net_recon_model = os.path.join(current_root_path, 'checkpoints', 'epoch_20.pth')
    dir_of_BFM_fitting = os.path.join(current_root_path, 'checkpoints', 'BFM_Fitting')
    wav2lip_checkpoint = os.path.join(current_root_path, 'checkpoints', 'wav2lip.pth')

    audio2pose_checkpoint = os.path.join(current_root_path, 'checkpoints', 'auido2pose_00140-model.pth')
    audio2pose_yaml_path = os.path.join(current_root_path, 'config', 'auido2pose.yaml')
    
    audio2exp_checkpoint = os.path.join(current_root_path, 'checkpoints', 'auido2exp_00300-model.pth')
    audio2exp_yaml_path = os.path.join(current_root_path, 'config', 'auido2exp.yaml')

    free_view_checkpoint = os.path.join(current_root_path, 'checkpoints', 'facevid2vid_00189-model.pth.tar')
    mapping_checkpoint = os.path.join(current_root_path, 'checkpoints', 'mapping_00229-model.pth.tar')
    facerender_yaml_path = os.path.join(current_root_path, 'config', 'facerender.yaml')

    #init model
    print(path_of_net_recon_model)
    preprocess_model = CropAndExtract(path_of_lm_croper, path_of_net_recon_model, dir_of_BFM_fitting, device)

    print(audio2pose_checkpoint)
    print(audio2exp_checkpoint)
    audio_to_coeff = Audio2Coeff(audio2pose_checkpoint, audio2pose_yaml_path, 
                                audio2exp_checkpoint, audio2exp_yaml_path, 
                                wav2lip_checkpoint, device)
    
    print(free_view_checkpoint)
    print(mapping_checkpoint)
    animate_from_coeff = AnimateFromCoeff(free_view_checkpoint, mapping_checkpoint, 
                                            facerender_yaml_path, device)

    #crop image and extract 3dmm from image
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    first_coeff_path, crop_pic_path =  preprocess_model.generate(pic_path, first_frame_dir)
    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return

    #audio2ceoff
    batch = get_data(first_coeff_path, audio_path, device)
    coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style)
    #coeff2video
    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, 
                                batch_size, camera_yaw_list, camera_pitch_list, camera_roll_list)
    animate_from_coeff.generate(data, save_dir)
    video_name = data['video_name']
    print("LOCAL RESULT PATH : ", os.path.join(save_dir, f"{video_name}.mp4"))
    return os.path.join(save_dir, f"{video_name}.mp4")


audio = gradio.Audio(label="Driver audio", type="filepath")
image = gradio.Image(label="Pass either image", type="filepath")
video = gradio.Video(label="Or pass video", type="filepath")
output_video = gradio.PlayableVideo(label="Result")
pose_style = gradio.Slider(0, 45, type=int, value=0, label="Pose style")
batch_size = gradio.Number(value=8, type=int, label="Batch Size")

interface = gradio.Interface(
    synthesize,
    [audio, image, video, pose_style, batch_size],
    output_video
)

if __name__=="__main__":
    interface.launch(share=True)