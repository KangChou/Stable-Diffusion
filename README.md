# Stable-Diffusion
Stable Diffusion  博客介绍：https://huggingface.co/blog/stable_diffusion

stable-diffusion-webui-colab： https://github.com/camenduru/stable-diffusion-webui-colab

stable_diffusion_chilloutmix：https://github.com/wibus-wee/stable_diffusion_chilloutmix_ipynb/blob/main/prompts.md


运行地址：
https://colab.research.google.com/drive/1Gg_dUYWet87CSKZ3vpZqYN97nX7DQ1hH#scrollTo=DCancU56vEwm

lora模型：
```
from diffusers import StableDiffusionPipeline, DEISMultistepScheduler
 
model_id = "./model/cilloutmixNi"
 
pipe = StableDiffusionPipeline.from_pretrained(model_id, custom_pipeline="lpw_stable_diffusion")
pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
 
prompt = "<lora:koreanDollLikeness_v10:0.8>, best quality, ultra high res, (photorealistic:1.4), 1woman, sleeveless white button shirt, black skirt, black choker, cute, (Kpop idol), (aegyo sal:1), (silver hair:1), ((puffy eyes)), looking at viewer, peace sign"
negative_prompt = "paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, nsfw, nipples"
 
image = pipe(
    prompt,
    num_inference_steps=40,
    guidance_scale=10,
    width=512,
    height=512,
    max_embeddings_multiples=2,
    negative_prompt=negative_prompt
).images[0]
 
image.save("test.png")
```

中国古典：https://www.kocpc.com.tw/archives/482719



参考：[https://self-development5/](https://self-development.info/%e3%80%90stable-diffusion%e3%80%91chilloutmix%e3%81%ae%e5%88%a9%e7%94%a8%e6%96%b9%e6%b3%95/)

你所要的提示词库可以在这里找到：https://playgroundai.com/

建议您使用以下资源：

*   我们的[科拉布笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb)。
*   [扩散器入门](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb)笔记本，其中提供了有关扩散系统的更广泛概述。
*   [带注释的扩散模型](https://huggingface.co/blog/annotated-diffusion)博客文章。
*   我们在 [GitHub 中的代码](https://github.com/huggingface/diffusers)，如果您留下 ⭐ if 对您有用，我们将非常高兴！`diffusers`
