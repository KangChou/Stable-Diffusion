参考：https://www.kaggle.com/code/kangfilm/stable-diffusion-webui-various-models/edit

```
#Original notebook: https://www.kaggle.com/code/camenduru/stable-diffusion-webui-kaggle
#Install Python 3.10
!conda create -n webui_env -c cctbx202208 -y
!source /opt/conda/bin/activate webui_env && conda install -q -c cctbx202208 python -y

#Install PyTorch
!/opt/conda/envs/webui_env/bin/python3 -m pip install -q torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html

#Download Automatic1111's Stable Diffusion Web UI
!git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui
%cd /kaggle/working/stable-diffusion-webui
#Use a version committed on 2/18/2023, because newer version might not work
!git checkout 0cc0ee1bcb4c24a8c9715f66cede06601bfc00c8


#[Model section] Please use one model at a time. Uncomment means remove # from the beginning of a line. 

#Uncomment the following line to use Analog Diffusion 1.0
#!wget -O /kaggle/working/stable-diffusion-webui/models/Stable-diffusion/analog-diffusion-1.0.ckpt -L https://huggingface.co/wavymulder/Analog-Diffusion/resolve/main/analog-diffusion-1.0.ckpt

#Uncomment the following two lines to use Anything 3.0
#!wget -O /kaggle/working/stable-diffusion-webui/models/Stable-diffusion/anything-v3-fp16-pruned.safetensors -L https://huggingface.co/Linaqruf/anything-v3.0/resolve/main/anything-v3-fp16-pruned.safetensors

#Uncomment the following line to use Dreamlike Diffusion 1.0
#!wget -O /kaggle/working/stable-diffusion-webui/models/Stable-diffusion/dreamlike-diffusion-1.0.ckpt -L https://huggingface.co/dreamlike-art/dreamlike-diffusion-1.0/resolve/main/dreamlike-diffusion-1.0.ckpt

#Uncomment the following line to use Elldreth's Lucid Mix 1.0
#!wget -O /kaggle/working/stable-diffusion-webui/models/Stable-diffusion/elldrethSLucidMix_v10.safetensors -L https://civitai.com/api/download/models/1450

#Uncomment the following line to use Openjourney
#!wget -O /kaggle/working/stable-diffusion-webui/models/Stable-diffusion/mdjrny-v4.ckpt -L https://huggingface.co/prompthero/openjourney/resolve/main/mdjrny-v4.ckpt

#Uncomment the following line to use ProtoGen X3.4
#!wget -O /kaggle/working/stable-diffusion-webui/models/Stable-diffusion/ProtoGen_X3.4.safetensors -L https://huggingface.co/darkstorm2150/Protogen_x3.4_Official_Release/resolve/main/ProtoGen_X3.4.safetensors

#Uncomment the following line to use ProtoThing_200
#!wget -O /kaggle/working/stable-diffusion-webui/models/Stable-diffusion/ProtoThing_200.ckpt -L https://huggingface.co/NiteStormz/ProtoThing_200/resolve/main/ProtoThing_200.ckpt

#Add a model of your choice. Replace model_name, model_link and remove # from the #!wget line. 
model_name="EimisAnimeDiffusion_1-0v.ckpt"
model_link="https://huggingface.co/eimiss/EimisAnimeDiffusion_1.0v/resolve/main/EimisAnimeDiffusion_1-0v.ckpt"
#!wget -O /kaggle/working/stable-diffusion-webui/models/Stable-diffusion/{model_name} -L {model_link}

#[VAE section](Uncomment the one you want to use. Go to settings to select the vae and apply settings to use it)
#!wget -O /kaggle/working/stable-diffusion-webui/models/VAE/vae-ft-mse-840000-ema-pruned.vae.ckpt -L https://huggingface.co/AmethystVera/SimpMaker-3K1/resolve/main/vae-ft-mse-840000-ema-pruned.ckpt
#Add a VAE of your choice. Replace vae_name, vae_link, and remove # from the #!wget line
vae_name="Anything-V3.0.vae.pt"
vae_link="https://huggingface.co/Linaqruf/anything-v3.0/resolve/main/Anything-V3.0.vae.pt"
#!wget -O /kaggle/working/stable-diffusion-webui/models/VAE/{vae_name} -L {vae_link}
    
#[Embedding section](Uncomment the one you want to use)
#!wget -O /kaggle/working/stable-diffusion-webui/embeddings/bad-artist.pt -L https://huggingface.co/nick-x-hacker/bad-artist/resolve/main/bad-artist.pt
#Add an embedding of your choice. Replace embedding_name, embedding_link
embedding_name="bad_prompt_version2.pt"
embedding_link="https://huggingface.co/datasets/Nerfgun3/bad_prompt/resolve/main/bad_prompt_version2.pt"
#!wget -O /kaggle/working/stable-diffusion-webui/embeddings/{embedding_name} -L {embedding_link}

#[Hypernetwork section](Uncomment the one you want to use)
#Create hypernetworks directory
!mkdir /kaggle/working/stable-diffusion-webui/models/hypernetworks
#!wget -O /kaggle/working/stable-diffusion-webui/models/hypernetworks/eimi.pt -L https://huggingface.co/WarriorMama777/HyperNetworkCollection_v2/resolve/main/_Korea_arca.live_HypernetworkCollection/eimi.pt
#Add a hypernetwork of your choice. Replace hypernetwork_name, hypernetwork_link
hypernetwork_name="azusa.pt"
hypernetwork_link="https://huggingface.co/WarriorMama777/HyperNetworkCollection_v2/resolve/main/_Korea_arca.live_HypernetworkCollection/azusa.pt"
#!wget -O /kaggle/working/stable-diffusion-webui/models/hypernetworks/{hypernetwork_name} -L {hypernetwork_link}

#Enable additional networks (LoRA) extension
%cd /kaggle/working/stable-diffusion-webui/extensions
!git clone  https://github.com/kohya-ss/sd-webui-additional-networks.git 
!mkdir /kaggle/working/stable-diffusion-webui/models/Lora
lora_model_name="makimaChainsawMan_offset.safetensors"
lora_model_link="https://civitai.com/api/download/models/6244"
#!wget -O /kaggle/working/stable-diffusion-webui/models/Lora/{lora_model_name} -L {lora_model_link}

    
%cd /kaggle/working/stable-diffusion-webui    
!/opt/conda/envs/webui_env/bin/python3 launch.py --share --enable-insecure-extension-access

```


![image](https://user-images.githubusercontent.com/36963108/228440861-a5a81402-f037-4dcd-8c31-c9a9a998224b.png)
