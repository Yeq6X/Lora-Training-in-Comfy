# Original LoRA train script by @Akegarasu ; rewritten in Python by LJRE.
import subprocess
import os
import folder_paths
import random
import torch
import sys
import json
import time
import requests
import httpx
from gradio_client import Client

def GetTrainScript(script_name:str):
    # Current file directory from __file__
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    sd_script_dir = os.path.join(current_file_dir, "sd-scripts")
    train_script_path = os.path.join(sd_script_dir, f"{script_name}.py")
    return train_script_path, sd_script_dir

def wait_for_gradio(max_retries=30, retry_interval=3):
    """
    Gradio APIサーバーの起動を待つ
    """
    for i in range(max_retries):
        print(f"Waiting for Gradio API (attempt {i+1}/{max_retries})")
        try:
            client = Client("http://127.0.0.1:7860/")
            if client:
                time.sleep(5)  # 安定するまで少し待つ
                return client
        except (requests.exceptions.RequestException, httpx.ConnectError) as e:
            print(f"Connection error: {str(e)}")
            time.sleep(retry_interval)
    return None

def common_loratraining_sdxl(self, params):
    
    kohya_process = None
    try:
        # kohya_gui.pyの起動
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        kohya_ss_dir = os.path.join(current_file_dir, "kohya_ss")
        kohya_gui_path = os.path.join(kohya_ss_dir, "kohya_gui.py")
        
        # kohya_ss/venv/のPythonパスを取得
        venv_python_path = os.path.join(kohya_ss_dir, "venv", "Scripts", "python.exe")
        
        print("Starting kohya_gui.py --headless...")
        print(f"Using Python: {venv_python_path}")
        print(f"Script: {kohya_gui_path}")

        # パラメータをtmpファイルに保存
        tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        tmp_json_path = os.path.join(tmp_dir, "train.json")
        with open(tmp_json_path, "w") as f:
            json.dump(params, f)
        print(f"パラメータをtmpファイルに保存: {tmp_json_path}")
        
        if os.path.exists(kohya_gui_path) and os.path.exists(venv_python_path):
            # venv環境の環境変数を設定
            env = os.environ.copy()
            venv_scripts_path = os.path.join(kohya_ss_dir, "venv", "Scripts")
            env['PATH'] = venv_scripts_path + os.pathsep + env['PATH']
            env['VIRTUAL_ENV'] = os.path.join(kohya_ss_dir, "venv")
            
            kohya_process = subprocess.Popen([
                venv_python_path, 
                kohya_gui_path, 
                "--headless",
                "--noverify"
            ], cwd=kohya_ss_dir, env=env)
            print(f"kohya_gui.py started with PID: {kohya_process.pid}")
        else:
            if not os.path.exists(kohya_gui_path):
                print(f"kohya_gui.py not found at: {kohya_gui_path}")
                raise FileNotFoundError(f"kohya_gui.py not found at: {kohya_gui_path}")
            if not os.path.exists(venv_python_path):
                print(f"venv Python not found at: {venv_python_path}")
                raise FileNotFoundError(f"venv Python not found at: {venv_python_path}")
            raise FileNotFoundError(f"Required files not found. Script: {kohya_gui_path}, Python: {venv_python_path}")
        
        # Gradio APIの起動を待つ
        client = wait_for_gradio()
        if not client:
            raise Exception("Gradio API failed to start within the timeout period")
        
        print("Gradio API is ready")
        
        # Step 1: Configure LoRA parameters with /open_configuration_2
        print("Step 1: Configuring LoRA parameters...")
        config_params = {
            "ask_for_file": False,
            "apply_preset": False,
            "file_path": tmp_json_path,
            "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
            "v2": False,
            "v_parameterization": False,
            "sdxl": False,
            "flux1_checkbox": False,
            "dataset_config": "",
            "save_model_as": "safetensors",
            "save_precision": "fp16",
            "train_data_dir": "",
            "output_name": "last",
            "model_list": "",
            "training_comment": "",
            "logging_dir": "",
            "reg_data_dir": "",
            "output_dir": "",
            "max_resolution": "512,512",
            "learning_rate": 0.0001,
            "lr_scheduler": "cosine",
            "lr_warmup": 10,
            "lr_warmup_steps": 0,
            "train_batch_size": 1,
            "epoch": 1,
            "save_every_n_epochs": 1,
            "seed": 0,
            "cache_latents": True,
            "cache_latents_to_disk": False,
            "caption_extension": ".txt",
            "enable_bucket": True,
            "stop_text_encoder_training": 0,
            "min_bucket_reso": 256,
            "max_bucket_reso": 2048,
            "max_train_epochs": 0,
            "max_train_steps": 1600,
            "lr_scheduler_num_cycles": 1,
            "lr_scheduler_power": 1,
            "optimizer": "AdamW8bit",
            "optimizer_args": "",
            "lr_scheduler_args": "",
            "lr_scheduler_type": "",
            "max_grad_norm": 1,
            "mixed_precision": "fp16",
            "num_cpu_threads_per_process": 2,
            "num_processes": 1,
            "num_machines": 1,
            "multi_gpu": False,
            "gpu_ids": "",
            "main_process_port": 0,
            "dynamo_backend": "no",
            "dynamo_mode": "default",
            "dynamo_use_fullgraph": False,
            "dynamo_use_dynamic": False,
            "extra_accelerate_launch_args": "",
            "gradient_checkpointing": False,
            "fp8_base": False,
            "fp8_base_unet": False,
            "full_fp16": False,
            "highvram": False,
            "lowvram": False,
            "xformers": "xformers",
            "shuffle_caption": False,
            "save_state": False,
            "save_state_on_train_end": False,
            "resume": "",
            "prior_loss_weight": 1,
            "color_aug": False,
            "flip_aug": False,
            "masked_loss": False,
            "clip_skip": 1,
            "gradient_accumulation_steps": 1,
            "mem_eff_attn": False,
            "max_token_length": 75,
            "max_data_loader_n_workers": 0,
            "keep_tokens": 0,
            "persistent_data_loader_workers": False,
            "bucket_no_upscale": True,
            "random_crop": False,
            "bucket_reso_steps": 64,
            "v_pred_like_loss": 0,
            "caption_dropout_every_n_epochs": 0,
            "caption_dropout_rate": 0,
            "noise_offset_type": "Original",
            "noise_offset": 0,
            "noise_offset_random_strength": False,
            "adaptive_noise_scale": 0,
            "multires_noise_iterations": 0,
            "multires_noise_discount": 0.3,
            "ip_noise_gamma": 0,
            "ip_noise_gamma_random_strength": False,
            "additional_parameters": "",
            "loss_type": "l2",
            "huber_schedule": "snr",
            "huber_c": 0.1,
            "huber_scale": 1,
            "vae_batch_size": 0,
            "min_snr_gamma": 0,
            "save_every_n_steps": 0,
            "save_last_n_steps": 0,
            "save_last_n_steps_state": 0,
            "save_last_n_epochs": 0,
            "save_last_n_epochs_state": 0,
            "skip_cache_check": False,
            "log_with": "",
            "wandb_api_key": "",
            "wandb_run_name": "",
            "log_tracker_name": "",
            "log_tracker_config": "",
            "log_config": False,
            "scale_v_pred_loss_like_noise_pred": False,
            "full_bf16": False,
            "min_timestep": 0,
            "max_timestep": 1000,
            "vae": "",
            "weighted_captions": False,
            "debiased_estimation_loss": False,
            "sdxl_cache_text_encoder_outputs": False,
            "sdxl_no_half_vae": False,
            "text_encoder_lr": 0,
            "t5xxl_lr": 0,
            "unet_lr": 0.0001,
            "network_dim": 8,
            "network_weights": "Hello!!",
            "dim_from_weights": False,
            "network_alpha": 1,
            "LoRA_type": "Standard",
            "factor": -1,
            "bypass_mode": False,
            "dora_wd": False,
            "use_cp": False,
            "use_tucker": False,
            "use_scalar": False,
            "rank_dropout_scale": False,
            "constrain": 0,
            "rescaled": False,
            "train_norm": False,
            "decompose_both": False,
            "train_on_input": True,
            "conv_dim": 1,
            "conv_alpha": 1,
            "sample_every_n_steps": 0,
            "sample_every_n_epochs": 0,
            "sample_sampler": "euler_a",
            "sample_prompts": "",
            "down_lr_weight": "Hello!!",
            "mid_lr_weight": "Hello!!",
            "up_lr_weight": "Hello!!",
            "block_lr_zero_threshold": "Hello!!",
            "block_dims": "Hello!!",
            "block_alphas": "Hello!!",
            "conv_block_dims": "Hello!!",
            "conv_block_alphas": "Hello!!",
            "unit": 1,
            "scale_weight_norms": 0,
            "network_dropout": 0,
            "rank_dropout": 0,
            "module_dropout": 0,
            "LyCORIS_preset": "full",
            "loraplus_lr_ratio": 0,
            "loraplus_text_encoder_lr_ratio": 0,
            "loraplus_unet_lr_ratio": 0,
            "train_lora_ggpo": False,
            "ggpo_sigma": 0.03,
            "ggpo_beta": 0.01,
            "huggingface_repo_id": "",
            "huggingface_token": "",
            "huggingface_repo_type": "",
            "huggingface_repo_visibility": "",
            "huggingface_path_in_repo": "",
            "save_state_to_huggingface": False,
            "resume_from_huggingface": "",
            "async_upload": False,
            "metadata_author": "",
            "metadata_description": "",
            "metadata_license": "",
            "metadata_tags": "",
            "metadata_title": "",
            "flux1_cache_text_encoder_outputs": False,
            "flux1_cache_text_encoder_outputs_to_disk": False,
            "ae": "",
            "clip_l": "",
            "t5xxl": "",
            "discrete_flow_shift": 3,
            "model_prediction_type": "sigma_scaled",
            "timestep_sampling": "sigma",
            "split_mode": False,
            "train_blocks": "all",
            "t5xxl_max_token_length": 512,
            "enable_all_linear": False,
            "guidance_scale": 3.5,
            "mem_eff_save": False,
            "apply_t5_attn_mask": False,
            "split_qkv": False,
            "train_t5xxl": False,
            "cpu_offload_checkpointing": False,
            "blocks_to_swap": 0,
            "single_blocks_to_swap": 0,
            "double_blocks_to_swap": 0,
            "img_attn_dim": "",
            "img_mlp_dim": "",
            "img_mod_dim": "",
            "single_dim": "",
            "txt_attn_dim": "",
            "txt_mlp_dim": "",
            "txt_mod_dim": "",
            "single_mod_dim": "",
            "in_dims": "",
            "train_double_block_indices": "all",
            "train_single_block_indices": "all",
            "sd3_cache_text_encoder_outputs": False,
            "sd3_cache_text_encoder_outputs_to_disk": False,
            "sd3_fused_backward_pass": False,
            "clip_g": "",
            "clip_g_dropout_rate": 0,
            "sd3_clip_l": "",
            "sd3_clip_l_dropout_rate": 0,
            "sd3_disable_mmap_load_safetensors": False,
            "sd3_enable_scaled_pos_embed": False,
            "logit_mean": 0,
            "logit_std": 1,
            "mode_scale": 1.29,
            "pos_emb_random_crop_rate": 0,
            "save_clip": False,
            "save_t5xxl": False,
            "sd3_t5_dropout_rate": 0,
            "sd3_t5xxl": "",
            "t5xxl_device": "",
            "t5xxl_dtype": "bf16",
            "sd3_text_encoder_batch_size": 1,
            "weighting_scheme": "logit_normal",
            "sd3_checkbox": False,
            "training_preset": "none",
            "api_name": "/open_configuration_2"
        }
        
        # Call the configuration API first
        config_result = client.predict(**config_params)
        print("Configuration result length:", len(config_result))

        train_params_values = config_result[1:len(config_result)-1]
        # Step 2: Start LoRA training with /train_model_2
        print("Step 2: Starting LoRA training...")
        train_param_keys = [
            "pretrained_model_name_or_path",
            "v2",
            "v_parameterization",
            "sdxl",
            "flux1_checkbox",
            "dataset_config",
            "save_model_as",
            "save_precision",
            "train_data_dir",
            "output_name",
            "model_list",
            "training_comment",
            "logging_dir",
            "reg_data_dir",
            "output_dir",
            "max_resolution",
            "learning_rate",
            "lr_scheduler",
            "lr_warmup",
            "lr_warmup_steps",
            "train_batch_size",
            "epoch",
            "save_every_n_epochs",
            "seed",
            "cache_latents",
            "cache_latents_to_disk",
            "caption_extension",
            "enable_bucket",
            "stop_text_encoder_training",
            "min_bucket_reso",
            "max_bucket_reso",
            "max_train_epochs",
            "max_train_steps",
            "lr_scheduler_num_cycles",
            "lr_scheduler_power",
            "optimizer",
            "optimizer_args",
            "lr_scheduler_args",
            "lr_scheduler_type",
            "max_grad_norm",
            "mixed_precision",
            "num_cpu_threads_per_process",
            "num_processes",
            "num_machines",
            "multi_gpu",
            "gpu_ids",
            "main_process_port",
            "dynamo_backend",
            "dynamo_mode",
            "dynamo_use_fullgraph",
            "dynamo_use_dynamic",
            "extra_accelerate_launch_args",
            "gradient_checkpointing",
            "fp8_base",
            "fp8_base_unet",
            "full_fp16",
            "highvram",
            "lowvram",
            "xformers",
            "shuffle_caption",
            "save_state",
            "save_state_on_train_end",
            "resume",
            "prior_loss_weight",
            "color_aug",
            "flip_aug",
            "masked_loss",
            "clip_skip",
            "gradient_accumulation_steps",
            "mem_eff_attn",
            "max_token_length",
            "max_data_loader_n_workers",
            "keep_tokens",
            "persistent_data_loader_workers",
            "bucket_no_upscale",
            "random_crop",
            "bucket_reso_steps",
            "v_pred_like_loss",
            "caption_dropout_every_n_epochs",
            "caption_dropout_rate",
            "noise_offset_type",
            "noise_offset",
            "noise_offset_random_strength",
            "adaptive_noise_scale",
            "multires_noise_iterations",
            "multires_noise_discount",
            "ip_noise_gamma",
            "ip_noise_gamma_random_strength",
            "additional_parameters",
            "loss_type",
            "huber_schedule",
            "huber_c",
            "huber_scale",
            "vae_batch_size",
            "min_snr_gamma",
            "save_every_n_steps",
            "save_last_n_steps",
            "save_last_n_steps_state",
            "save_last_n_epochs",
            "save_last_n_epochs_state",
            "skip_cache_check",
            "log_with",
            "wandb_api_key",
            "wandb_run_name",
            "log_tracker_name",
            "log_tracker_config",
            "log_config",
            "scale_v_pred_loss_like_noise_pred",
            "full_bf16",
            "min_timestep",
            "max_timestep",
            "vae",
            "weighted_captions",
            "debiased_estimation_loss",
            "sdxl_cache_text_encoder_outputs",
            "sdxl_no_half_vae",
            "text_encoder_lr",
            "t5xxl_lr",
            "unet_lr",
            "network_dim",
            "network_weights",
            "dim_from_weights",
            "network_alpha",
            "LoRA_type",
            "factor",
            "bypass_mode",
            "dora_wd",
            "use_cp",
            "use_tucker",
            "use_scalar",
            "rank_dropout_scale",
            "constrain",
            "rescaled",
            "train_norm",
            "decompose_both",
            "train_on_input",
            "conv_dim",
            "conv_alpha",
            "sample_every_n_steps",
            "sample_every_n_epochs",
            "sample_sampler",
            "sample_prompts",
            "down_lr_weight",
            "mid_lr_weight",
            "up_lr_weight",
            "block_lr_zero_threshold",
            "block_dims",
            "block_alphas",
            "conv_block_dims",
            "conv_block_alphas",
            "unit",
            "scale_weight_norms",
            "network_dropout",
            "rank_dropout",
            "module_dropout",
            "LyCORIS_preset",
            "loraplus_lr_ratio",
            "loraplus_text_encoder_lr_ratio",
            "loraplus_unet_lr_ratio",
            "train_lora_ggpo",
            "ggpo_sigma",
            "ggpo_beta",
            "huggingface_repo_id",
            "huggingface_token",
            "huggingface_repo_type",
            "huggingface_repo_visibility",
            "huggingface_path_in_repo",
            "save_state_to_huggingface",
            "resume_from_huggingface",
            "async_upload",
            "metadata_author",
            "metadata_description",
            "metadata_license",
            "metadata_tags",
            "metadata_title",
            "flux1_cache_text_encoder_outputs",
            "flux1_cache_text_encoder_outputs_to_disk",
            "ae",
            "clip_l",
            "t5xxl",
            "discrete_flow_shift",
            "model_prediction_type",
            "timestep_sampling",
            "split_mode",
            "train_blocks",
            "t5xxl_max_token_length",
            "enable_all_linear",
            "guidance_scale",
            "mem_eff_save",
            "apply_t5_attn_mask",
            "split_qkv",
            "train_t5xxl",
            "cpu_offload_checkpointing",
            "blocks_to_swap",
            "single_blocks_to_swap",
            "double_blocks_to_swap",
            "img_attn_dim",
            "img_mlp_dim",
            "img_mod_dim",
            "single_dim",
            "txt_attn_dim",
            "txt_mlp_dim",
            "txt_mod_dim",
            "single_mod_dim",
            "in_dims",
            "train_double_block_indices",
            "train_single_block_indices",
            "sd3_cache_text_encoder_outputs",
            "sd3_cache_text_encoder_outputs_to_disk",
            "sd3_fused_backward_pass",
            "clip_g",
            "clip_g_dropout_rate",
            "sd3_clip_l",
            "sd3_clip_l_dropout_rate",
            "sd3_disable_mmap_load_safetensors",
            "sd3_enable_scaled_pos_embed",
            "logit_mean",
            "logit_std",
            "mode_scale",
            "pos_emb_random_crop_rate",
            "save_clip",
            "save_t5xxl",
            "sd3_t5_dropout_rate",
            "sd3_t5xxl",
            "t5xxl_device",
            "t5xxl_dtype",
            "sd3_text_encoder_batch_size",
            "weighting_scheme",
            "sd3_checkbox",
        ]

        train_params = {}
        for key in train_param_keys:
            train_params[key] = train_params_values[train_param_keys.index(key)]
        train_params.update({
            "headless": True,
            "print_only": False,
            "api_name": "/train_model_2"
        })

        print("Starting LoRA training via Gradio API...")
        result = client.predict(**train_params)
        print("Result:", result)
        
        file_path = os.path.join(kohya_ss_dir, "logs", "print_command.txt")
        # 最終行を取得
        with open(file_path, "r") as f:
            command_to_run = f.readlines()[-1]
        print(command_to_run)

        return ()
        
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        raise e
    
    finally:
        # kohya_gui.pyプロセスの終了処理
        if kohya_process and kohya_process.poll() is None:
            print("Terminating kohya_gui.py process...")
            try:
                kohya_process.terminate()
                # 5秒待ってもまだ動いていたら強制終了
                kohya_process.wait(timeout=5)
                print("kohya_gui.py process terminated successfully")
            except subprocess.TimeoutExpired:
                print("Force killing kohya_gui.py process...")
                kohya_process.kill()
                kohya_process.wait()
                print("kohya_gui.py process killed")
            except Exception as cleanup_error:
                print(f"Error during cleanup: {str(cleanup_error)}")

class LoraTraininginComfySDXLJSON:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
         return {
            "required": {
            "json_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()

    FUNCTION = "loratraining"

    OUTPUT_NODE = True

    CATEGORY = "LJRE/LORA"

    def loratraining(self, json_path):
        # jsonパスの親ディレクトリを取得
        parent_dir = os.path.dirname(json_path)
        # 親ディレクトリ以下をlsで取得
        files = os.listdir(parent_dir)
        print(files)
        
        # JSONファイル読み込み
        try:
            with open(json_path, 'r') as f:
                params = json.load(f)
        except FileNotFoundError:
            raise ValueError(f"JSON file not found: {json_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file {json_path}: {str(e)}")
        
        # トレーニング実行（ここで発生する例外はそのまま伝播させる）
        return common_loratraining_sdxl(self, params)

# Update existing LoraTraininginComfyAdvanced to use common function
class LoraTraininginComfySDXL:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
         return {
            "required": {
            "network_module": (["networks.lora", "lycoris.kohya"], ),
            "LoRA_type": ("STRING", {"default": "Standard"}),
            "LyCORIS_preset": ("STRING", {"default": "full"}),
            "adaptive_noise_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.1}),
            "additional_parameters": ("STRING", {"default": ""}),
            "block_alphas": ("STRING", {"default": ""}),
            "block_dims": ("STRING", {"default": ""}),
            "block_lr_zero_threshold": ("STRING", {"default": ""}),
            "bucket_no_upscale": ("BOOLEAN", {"default": True}),
            "bucket_reso_steps": ("INT", {"default": 64, "min": 1}),
            "bypass_mode": ("BOOLEAN", {"default": False}),
            "cache_latents": ("BOOLEAN", {"default": True}),
            "cache_latents_to_disk": ("BOOLEAN", {"default": False}),
            "caption_dropout_every_n_epochs": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.1}),
            "caption_dropout_rate": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.1}),
            "caption_extension": ("STRING", {"default": ".txt"}),
            "clip_skip": ("STRING", {"default": "1"}),
            "color_aug": ("BOOLEAN", {"default": False}),
            "constrain": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.1}),
            "conv_alpha": ("INT", {"default": 1}),
            "conv_block_alphas": ("STRING", {"default": ""}),
            "conv_block_dims": ("STRING", {"default": ""}),
            "conv_dim": ("INT", {"default": 1}),
            "dataset_config": ("STRING", {"default": ""}),
            "debiased_estimation_loss": ("BOOLEAN", {"default": False}),
            "decompose_both": ("BOOLEAN", {"default": False}),
            "dim_from_weights": ("BOOLEAN", {"default": False}),
            "dora_wd": ("BOOLEAN", {"default": False}),
            "down_lr_weight": ("STRING", {"default": ""}),
            "enable_bucket": ("BOOLEAN", {"default": True}),
            "epoch": ("INT", {"default": 15, "min": 1}),
            "extra_accelerate_launch_args": ("STRING", {"default": ""}),
            "factor": ("INT", {"default": -1}),
            "flip_aug": ("BOOLEAN", {"default": False}),
            "fp8_base": ("BOOLEAN", {"default": False}),
            "full_bf16": ("BOOLEAN", {"default": False}),
            "full_fp16": ("BOOLEAN", {"default": False}),
            "gpu_ids": ("STRING", {"default": ""}),
            "gradient_accumulation_steps": ("STRING", {"default": "1"}),
            "gradient_checkpointing": ("BOOLEAN", {"default": False}),
            "huber_c": ("FLOAT", {"default": 0.1, "min": 0.0, "step": 0.1}),
            "huber_schedule": ("STRING", {"default": "snr"}),
            "ip_noise_gamma": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.1}),
            "ip_noise_gamma_random_strength": ("BOOLEAN", {"default": False}),
            "keep_tokens": ("STRING", {"default": "1"}),
            "learning_rate": ("FLOAT", {"default": 0.0001, "min": 0.0, "step": 0.00001}),
            "log_tracker_config": ("STRING", {"default": ""}),
            "log_tracker_name": ("STRING", {"default": ""}),
            "logging_dir": ("STRING", {"default": ""}),
            "lora_network_weights": ("STRING", {"default": ""}),
            "loss_type": ("STRING", {"default": "l2"}),
            "lr_scheduler": ("STRING", {"default": "cosine"}),
            "lr_scheduler_args": ("STRING", {"default": ""}),
            "lr_scheduler_num_cycles": ("STRING", {"default": ""}),
            "lr_scheduler_power": ("STRING", {"default": ""}),
            "lr_warmup": ("INT", {"default": 10}),
            "main_process_port": ("INT", {"default": 0}),
            "masked_loss": ("BOOLEAN", {"default": False}),
            "max_bucket_reso": ("INT", {"default": 2048, "min": 64}),
            "max_data_loader_n_workers": ("STRING", {"default": "0"}),
            "max_grad_norm": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.1}),
            "max_resolution": ("STRING", {"default": "1024,1024"}),
            "max_timestep": ("INT", {"default": 1000, "min": 0}),
            "max_token_length": ("STRING", {"default": "75"}),
            "max_train_epochs": ("STRING", {"default": ""}),
            "max_train_steps": ("STRING", {"default": ""}),
            "mem_eff_attn": ("BOOLEAN", {"default": False}),
            "mid_lr_weight": ("STRING", {"default": ""}),
            "min_bucket_reso": ("INT", {"default": 256, "min": 64}),
            "min_snr_gamma": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.1}),
            "min_timestep": ("INT", {"default": 0, "min": 0}),
            "mixed_precision": ("STRING", {"default": "bf16"}),
            "model_list": ("STRING", {"default": "custom"}),
            "module_dropout": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.1}),
            "multi_gpu": ("BOOLEAN", {"default": False}),
            "multires_noise_discount": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.1}),
            "multires_noise_iterations": ("INT", {"default": 0, "min": 0}),
            "network_alpha": ("INT", {"default": 32, "min": 0}),
            "network_dim": ("INT", {"default": 64, "min": 0}),
            "network_dropout": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.1}),
            "noise_offset": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.1}),
            "noise_offset_random_strength": ("BOOLEAN", {"default": False}),
            "noise_offset_type": ("STRING", {"default": "Original"}),
            "num_cpu_threads_per_process": ("INT", {"default": 2, "min": 1}),
            "num_machines": ("INT", {"default": 1, "min": 1}),
            "num_processes": ("INT", {"default": 1, "min": 1}),
            "optimizer": ("STRING", {"default": "AdamW8bit"}),
            "optimizer_args": ("STRING", {"default": ""}),
            "output_dir": ("STRING", {"default": "models/loras"}),
            "output_name": ("STRING", {"default": "Desired name for LoRA."}),
            "persistent_data_loader_workers": ("BOOLEAN", {"default": False}),
            "pretrained_model_name_or_path": ("STRING", {"default": ""}),
            "prior_loss_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.1}),
            "random_crop": ("BOOLEAN", {"default": False}),
            "rank_dropout": ("FLOAT", {"default": 0, "min": 0, "step": 0.1}),
            "rank_dropout_scale": ("BOOLEAN", {"default": False}),
            "reg_data_dir": ("STRING", {"default": ""}),
            "rescaled": ("BOOLEAN", {"default": False}),
            "resume": ("STRING", {"default": ""}),
            "sample_every_n_epochs": ("INT", {"default": 0, "min": 0}),
            "sample_every_n_steps": ("INT", {"default": 0, "min": 0}),
            "sample_prompts": ("STRING", {"default": ""}),
            "sample_sampler": ("STRING", {"default": "euler_a"}),
            "save_every_n_epochs": ("INT", {"default": 1, "min": 0}),
            "save_every_n_steps": ("INT", {"default": 0, "min": 0}),
            "save_last_n_steps": ("INT", {"default": 0, "min": 0}),
            "save_last_n_steps_state": ("INT", {"default": 0, "min": 0}),
            "save_model_as": ("STRING", {"default": "safetensors"}),
            "save_precision": ("STRING", {"default": "bf16"}),
            "save_state": ("BOOLEAN", {"default": False}),
            "save_state_on_train_end": ("BOOLEAN", {"default": False}),
            "scale_v_pred_loss_like_noise_pred": ("BOOLEAN", {"default": False}),
            "scale_weight_norms": ("FLOAT", {"default": 0, "min": 0, "step": 0.1}),
            "sdxl": ("BOOLEAN", {"default": True}),
            "sdxl_cache_text_encoder_outputs": ("BOOLEAN", {"default": False}),
            "sdxl_no_half_vae": ("BOOLEAN", {"default": False}),
            "seed": ("STRING", {"default": "1000"}),
            "shuffle_caption": ("BOOLEAN", {"default": True}),
            "stop_text_encoder_training": ("INT", {"default": 0, "min": 0}),
            "text_encoder_lr": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.00001}),
            "train_batch_size": ("INT", {"default": 1, "min": 1}),
            "train_data_dir": ("STRING", {"default": "Insert path of image folders"}),
            "train_norm": ("BOOLEAN", {"default": False}),
            "train_on_input": ("BOOLEAN", {"default": True}),
            "training_comment": ("STRING", {"default": ""}),
            "unet_lr": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.00001}),
            "unit": ("INT", {"default": 1, "min": 1}),
            "up_lr_weight": ("STRING", {"default": ""}),
            "use_cp": ("BOOLEAN", {"default": False}),
            "use_scalar": ("BOOLEAN", {"default": False}),
            "use_tucker": ("BOOLEAN", {"default": False}),
            "use_wandb": ("BOOLEAN", {"default": False}),
            "v2": ("BOOLEAN", {"default": False}),
            "v_parameterization": ("BOOLEAN", {"default": False}),
            "v_pred_like_loss": ("FLOAT", {"default": 0, "min": 0, "step": 0.1}),
            "vae": ("STRING", {"default": ""}),
            "vae_batch_size": ("INT", {"default": 0, "min": 0}),
            "wandb_api_key": ("STRING", {"default": ""}),
            "wandb_run_name": ("STRING", {"default": ""}),
            "weighted_captions": ("BOOLEAN", {"default": False}),
            "xformers": ("STRING", {"default": "xformers"})
            },
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()

    FUNCTION = "loratraining"

    OUTPUT_NODE = True

    CATEGORY = "LJRE/LORA"

    def loratraining(self, **kwargs):
        return common_loratraining_sdxl(self, kwargs)

class TensorboardAccess:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
         return {
            "required": {
           
            },
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()

    FUNCTION = "opentensorboard"

    OUTPUT_NODE = True

    CATEGORY = "LJRE/LORA"

    def opentensorboard(self):
        command = f'tensorboard --logdir="{logging_dir}"'
        subprocess.Popen(command, shell=True)
        return()