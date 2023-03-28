# Stable-Diffusion
Stable Diffusion  博客介绍：https://huggingface.co/blog/stable_diffusion

stable-diffusion-webui-colab： https://github.com/camenduru/stable-diffusion-webui-colab


运行地址：
https://colab.research.google.com/drive/1Gg_dUYWet87CSKZ3vpZqYN97nX7DQ1hH#scrollTo=DCancU56vEwm


稳定扩散是由[CompVis](https://github.com/CompVis)，[Stability AI](https://stability.ai/)和[LAION](https://laion.ai/)的研究人员和工程师创建的文本到图像潜在扩散模型。 它是在[来自LAION-512B](https://laion.ai/blog/laion-5b/)数据库子集的512x5图像上进行训练的。*LAION-5B*是目前存在的最大，可自由访问的多模态数据集。

在这篇文章中，我们想展示如何将稳定扩散与[扩散器库一起使用🧨](https://github.com/huggingface/diffusers)，解释模型是如何工作的，最后更深入地探讨如何允许 一个用于自定义映像生成管道。`diffusers`

**注意**：强烈建议对扩散模型的工作原理有一个基本的了解。如果扩散 模型对您来说是全新的，我们建议您阅读以下博客文章之一：

*   [带注释的扩散模型](https://huggingface.co/blog/annotated-diffusion)
*   [扩散器入门 🧨](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb)

现在，让我们开始 生成一些图像 🎨 .

## [](https://huggingface.co/blog/stable_diffusion#running-stable-diffusion)运行稳定扩散

### [](https://huggingface.co/blog/stable_diffusion#license)许可证

在使用模型之前，您需要接受模型[许可证](https://huggingface.co/spaces/CompVis/stable-diffusion-license)才能下载和使用权重。**注意：不再需要通过 UI 显式接受许可证**。

该许可证旨在减轻如此强大的机器学习系统的潜在有害影响。 我们要求用户**完整而仔细地阅读许可证**。在这里，我们提供一个摘要：

1.  您不能使用该模型故意生成或共享非法或有害的输出或内容，
2.  我们对您生成的输出不主张任何权利，您可以自由使用它们，并对它们的使用负责，这不应违反许可证中的规定，并且
3.  您可以重新分配权重，并将模型用于商业和/或服务。如果您这样做，请注意您必须包含与许可证中相同的使用限制，并将CreativeML OpenRAIL-M的副本共享给您的所有用户。

### [](https://huggingface.co/blog/stable_diffusion#usage)用法

首先，您应该安装以运行以下代码片段：`diffusers==0.10.2`

```
pip install diffusers==0.10.2 transformers scipy ftfy accelerate

```

在这篇文章中，我们将使用模型版本 [`v1-4`](https://huggingface.co/CompVis/stable-diffusion-v1-4)，但您也可以使用模型的其他版本，例如 1.5、2 和 2.1，只需最少的代码更改。

稳定扩散模型可以使用[`稳定扩散管道`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py)仅用几行进行推理。管道设置了从文本生成图像所需的一切 一个简单的函数调用。`from_pretrained`

```
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

```

如果有可用的 GPU，让我们将其移动到一个！

```
pipe.to("cuda")

```

**注意**：如果您受到 GPU 内存的限制并且可用的 GPU 内存少于 10GB，请 确保加载 In Float16 精度而不是默认值 浮点数32精度如上所述。`StableDiffusionPipeline`

您可以通过从分支加载权重并告诉期望 要保持浮点16精度的重量：`fp16``diffusers`

```
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)

```

要运行管道，只需定义提示并调用 。`pipe`

```
prompt = "a photograph of an astronaut riding a horse"

image = pipe(prompt).images[0]

# you can save the image with
# image.save(f"astronaut_rides_horse.png")

```

结果如下所示

[[图片上传中...(image-3c8874-1678157858352-5)]](https://huggingface.co/blog/assets/98_stable_diffusion/stable_diffusion_12_1.png) 

前面的代码每次运行时都会给你一个不同的图像。

如果在某个时候得到黑色图像，可能是因为模型内部构建的内容过滤器可能检测到 NSFW 结果。 如果您认为情况并非如此，请尝试调整提示或使用其他种子。事实上，模型预测包括有关是否针对特定结果检测到NSFW的信息。让我们看看它们是什么样子的：

```
result = pipe(prompt)
print(result)

```

```
{
    'images': [<PIL.Image.Image image mode=RGB size=512x512>],
    'nsfw_content_detected': [False]
}

```

如果你想要确定性输出，你可以设定一个随机种子，并将一个生成器传递给管道。 每次使用具有相同种子的生成器时，您都会获得相同的图像输出。

```
import torch

generator = torch.Generator("cuda").manual_seed(1024)
image = pipe(prompt, guidance_scale=7.5, generator=generator).images[0]

# you can save the image with
# image.save(f"astronaut_rides_horse.png")

```

结果如下所示

[[图片上传中...(image-f6c76b-1678157858352-4)]](https://huggingface.co/blog/assets/98_stable_diffusion/stable_diffusion_14_1.png) 

您可以使用参数更改推理步骤数。`num_inference_steps`

通常，使用的步骤越多，结果越好，但是步骤越多，生成所需的时间就越长。 稳定扩散在相对较少的步骤下效果很好，因此我们建议使用默认的推理步骤数。 如果您想要更快的结果，可以使用较小的数字。如果您想要更高质量的结果， 您可以使用更大的数字。`50`

让我们尝试使用较少的降噪步骤运行管道。

```
import torch

generator = torch.Generator("cuda").manual_seed(1024)
image = pipe(prompt, guidance_scale=7.5, num_inference_steps=15, generator=generator).images[0]

# you can save the image with
# image.save(f"astronaut_rides_horse.png")

```

[[图片上传中...(image-9cbbcd-1678157858352-3)]](https://huggingface.co/blog/assets/98_stable_diffusion/stable_diffusion_16_1.png) 

注意结构是相同的，但宇航员的服和马的一般形式存在问题。 这表明仅使用15个去噪步骤会显著降低生成结果的质量。如前所述，去噪步骤通常足以生成高质量的图像。`50`

除此之外，我们一直在使用另一个函数参数，在所有 前面的例子。 是一种提高对指导生成的条件信号（在本例中为文本）以及整体样本质量的依从性的方法。 它也被称为[无分类器引导](https://arxiv.org/abs/2207.12598)，简单来说，它迫使生成更好地匹配提示，这可能会以牺牲图像质量或多样性为代价。 介于 和 之间的值通常是稳定扩散的不错选择。默认情况下，管道 使用 7.5 的 a。`num_inference_steps``guidance_scale``guidance_scale``7``8.5``guidance_scale`

如果使用非常大的值，图像可能看起来不错，但会不那么多样化。 您可以在帖子的[这一部分中](https://huggingface.co/blog/stable_diffusion#how-to-write-your-own-inference-pipeline-with-diffusers)了解此参数的技术细节。

接下来，让我们看看如何一次生成同一提示的多个图像。 首先，我们将创建一个函数来帮助我们在网格中很好地可视化它们。`image_grid`

```
from PIL import Image

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

```

我们可以通过简单地使用重复多次相同提示的列表来为同一提示生成多个图像。我们将列表发送到管道，而不是我们之前使用的字符串。

```
num_images = 3
prompt = ["a photograph of an astronaut riding a horse"] * num_images

images = pipe(prompt).images

grid = image_grid(images, rows=1, cols=3)

# you can save the grid with
# grid.save(f"astronaut_rides_horse.png")

```



默认情况下，稳定扩散会产生像素图像。使用 and 参数覆盖默认值非常容易，以纵向或横向比例创建矩形图像。`512 × 512``height``width`

选择图像尺寸时，我们建议如下：

*   确保 和 都是 的倍数。`height``width``8`
*   低于 512 可能会导致图像质量较低。
*   在两个方向上超过 512 将重复图像区域（全局一致性丢失）。
*   创建非方形图像的最佳方法是在一个维度中使用，并在另一个维度中使用大于该值的值。`512`

让我们运行一个示例：

```
prompt = "a photograph of an astronaut riding a horse"
image = pipe(prompt, height=512, width=768).images[0]

# you can save the image with
# image.save(f"astronaut_rides_horse.png")

```

## [](https://huggingface.co/blog/stable_diffusion#how-does-stable-diffusion-work)稳定扩散如何工作？

看过稳定扩散可以产生的高质量图像后，让我们试着了解一下 模型的功能更好一些。

稳定扩散基于一种称为潜在扩散的特定类型的扩散模型，该模型在具有[潜在扩散模型的高分辨率图像合成](https://arxiv.org/abs/2112.10752)中提出。

一般来说，扩散模型是机器学习系统，经过训练可以逐步*去噪*随机高斯噪声，以获得感兴趣的样本，例如*图像*。有关它们如何工作的更详细概述，请查看[此 colab](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb)。

扩散模型已经证明可以实现生成图像数据的最新结果。但扩散模型的一个缺点是，反向去噪过程很慢，因为它具有重复的顺序性质。此外，这些模型消耗大量内存，因为它们在像素空间中运行，在生成高分辨率图像时会变得很大。因此，训练这些模型并将其用于推理具有挑战性。

潜在扩散可以通过在较低维度*的潜在*空间上应用扩散过程来降低内存和计算复杂性，而不是使用实际像素空间。这是标准扩散和潜在扩散模型之间的主要区别：**在潜在扩散中，模型被训练以生成图像的潜在（压缩）表示。**

潜伏扩散有三个主要组成部分。

1.  自动编码器（VAE）。
2.  [一个U-Net](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb#scrollTo=wW8o1Wp0zRkq)。
3.  文本编码器，*例如* [CLIP 的文本编码器](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel)。

**1\. 自动编码器 （VAE）**

VAE型号由编码器和解码器两部分组成。编码器用于将图像转换为低维潜在表示，该表示将用作*U-Net*模型的输入。 相反，解码器将潜在表示转换回图像。

在潜在扩散*训练*期间，编码器用于获取前向扩散过程图像的潜在表示（*潜在*表示），在每一步应用越来越多的噪声。在*推理*过程中，反向扩散过程中产生的去噪潜伏通过VAE解码器转换回图像。正如我们将在推理过程中看到的，我们**只需要VAE解码器**。

**2\. U-Net**

U-Net有一个编码器部分和一个解码器部分，都由ResNet块组成。 编码器将图像表示压缩为较低分辨率的图像表示，解码器将较低分辨率的图像表示解码回原始的高分辨率图像表示，该表示的噪声较小。 更具体地说，U-Net输出预测噪声残差，可用于计算预测的去噪图像表示。

为了防止U-Net在下采样时丢失重要信息，通常在编码器的下采样ResNet和解码器的上采样ResNet之间添加快捷方式连接。 此外，稳定的扩散U-Net能够通过交叉注意力层将其输出条件化为文本嵌入。交叉注意层通常添加到U-Net的编码器和解码器部分，通常在ResNet块之间。

**3\. 文本编码器**

文本编码器负责将输入提示（*例如*“宇航员骑马”）转换为U-Net可以理解的嵌入空间。它通常是*一个简单的基于转换器的*编码器，它将一系列输入标记映射到一系列潜在的文本嵌入。

受 [Imagen](https://imagen.research.google/) 的启发，稳定**扩散不会在**训练期间训练文本编码器，而只是使用 CLIP 已经训练的文本编码器 [CLIPTextModel](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel)。

**为什么潜在扩散快速高效？**

由于潜在扩散在低维空间上运行，因此与像素空间扩散模型相比，它大大降低了内存和计算要求。例如，稳定扩散中使用的自动编码器的折减系数为 8。这意味着形状的图像变为潜在空间，这需要的内存要少几倍。`(3, 512, 512)``(3, 64, 64)``8 × 8 = 64`

这就是为什么即使在 16GB Colab GPU 上也能如此快速地生成图像的原因！`512 × 512`

**推理过程中的稳定扩散**

综上所述，现在让我们通过说明逻辑流来仔细看看模型在推理中的工作原理。

[图片上传中...(image-8a4c8a-1678157858353-6)]

稳定扩散模型将潜在种子和文本提示作为输入。然后使用潜在种子生成大小的随机潜在图像表示<math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn _msttexthash="10322" _msthash="96">64</mn><mo _msttexthash="19565" _msthash="97">×</mo><mn _msttexthash="10322" _msthash="98">64</mn></mrow></semantics></math>64×64其中，文本提示转换为大小的文本嵌入<math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn _msttexthash="10725" _msthash="103">77</mn><mo _msttexthash="19565" _msthash="104">×</mo><mn _msttexthash="17173" _msthash="105">768</mn></mrow></semantics></math>77×768通过 CLIP 的文本编码器。

接下来，U-Net迭代地对随机潜在图像表示进行*降噪，*同时以文本嵌入为条件。U-Net的输出作为噪声残差，用于通过调度器算法计算去噪的潜在图像表示。许多不同的调度程序算法可用于此计算，每种算法都有其优点和缺点。对于稳定扩散，我们建议使用以下之一：

*   [PNDM 调度程序](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_pndm.py)（默认使用）
*   [DDIM 调度程序](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py)
*   [K-LMS 调度程序](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_lms_discrete.py)

关于调度程序算法函数如何超出本笔记本范围的理论，但简而言之，应该记住，他们根据先前的噪声表示和预测的噪声残差计算预测的去噪图像表示。 有关更多信息，我们建议研究阐明[基于扩散的生成模型的设计空间](https://arxiv.org/abs/2206.00364)

*去噪*过程重复*约*50次，以逐步检索更好的潜在图像表示。 完成后，潜在图像表示由变分自动编码器的解码器部分解码。

在对潜在和稳定扩散的简要介绍之后，让我们看看如何高级使用🤗拥抱脸库！`diffusers`

## [](https://huggingface.co/blog/stable_diffusion#writing-your-own-inference-pipeline)编写自己的推理管道

最后，我们将展示如何使用 创建自定义扩散管线。 编写自定义推理管道是该库的高级用法，可用于切换某些组件，例如上面解释的VAE或调度程序。`diffusers``diffusers`

例如，我们将展示如何将稳定扩散与不同的调度器一起使用，即[在此 PR](https://github.com/huggingface/diffusers/pull/185) 中添加[的 Katherine Crowson 的](https://github.com/crowsonkb) K-LMS 调度器。

[预训练模型](https://huggingface.co/CompVis/stable-diffusion-v1-4/tree/main)包括设置完整扩散管道所需的所有组件。它们存储在以下文件夹中：

*   `text_encoder`：稳定扩散使用 CLIP，但其他扩散模型可能使用其他编码器，例如 。`BERT`
*   `tokenizer`.它必须与模型使用的匹配。`text_encoder`
*   `scheduler`：用于在训练过程中逐步向图像添加噪声的调度算法。
*   `unet`：用于生成输入的潜在表示的模型。
*   `vae`：自动编码器模块，我们将使用它将潜在表示解码为真实图像。

我们可以通过引用组件保存的文件夹来加载组件，使用参数 .`subfolder``from_pretrained`

```
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler

# 1\. Load the autoencoder model which will be used to decode the latents into image space. 
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

# 2\. Load the tokenizer and text encoder to tokenize and encode the text. 
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# 3\. The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

```

现在，我们不再加载预定义的调度程序，而是使用一些拟合参数加载 [K-LMS 调度器](https://github.com/huggingface/diffusers/blob/71ba8aec55b52a7ba5a1ff1db1265ffdd3c65ea2/src/diffusers/schedulers/scheduling_lms_discrete.py#L26)。

```
from diffusers import LMSDiscreteScheduler

scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

```

接下来，让我们将模型移动到 GPU。

```
torch_device = "cuda"
vae.to(torch_device)
text_encoder.to(torch_device)
unet.to(torch_device) 

```

现在，我们定义将用于生成图像的参数。

请注意，定义类似于[Imagen论文](https://arxiv.org/pdf/2205.11487.pdf)中方程（2）的指导权重。 对应于不执行无分类器指导。在这里，我们像之前一样将其设置为 7.5。`guidance_scale``w``guidance_scale == 1`

与前面的示例相比，我们设置为 100 以获得更明确的图像。`num_inference_steps`

```
prompt = ["a photograph of an astronaut riding a horse"]

height = 512                        # default height of Stable Diffusion
width = 512                         # default width of Stable Diffusion

num_inference_steps = 100           # Number of denoising steps

guidance_scale = 7.5                # Scale for classifier-free guidance

generator = torch.manual_seed(0)    # Seed generator to create the inital latent noise

batch_size = len(prompt)

```

首先，我们得到传递的提示。 这些嵌入将用于调节 UNet 模型，并引导图像生成类似于输入提示的内容。`text_embeddings`

```
text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

```

我们还将获得无分类器指南的无条件文本嵌入，这些嵌入只是填充标记（空文本）的嵌入。它们需要具有与条件（和`text_embeddings``batch_size``seq_length`)

```
max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer(
    [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
)
uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]   

```

对于无分类器指导，我们需要执行两次正向传递：一次使用条件输入 （），另一次使用无条件嵌入 （）。在实践中，我们可以将两者连接成一个批处理，以避免执行两次正向传递。`text_embeddings``uncond_embeddings`

```
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

```

接下来，我们生成初始随机噪声。

```
latents = torch.randn(
    (batch_size, unet.in_channels, height // 8, width // 8),
    generator=generator,
)
latents = latents.to(torch_device)

```

如果我们在这个阶段检查，我们会看到它们的形状是 ，比我们想要生成的图像小得多。该模型稍后会将这种潜在表示（纯噪声）转换为图像。`latents``torch.Size([1, 4, 64, 64])``512 × 512`

接下来，我们使用我们选择的 . 这将计算在去噪过程中要使用的确切时间步长值。`num_inference_steps``sigmas`

```
scheduler.set_timesteps(num_inference_steps)

```

K-LMS 调度程序需要将 乘以其值。让我们在这里这样做：`latents``sigma`

```
latents = latents * scheduler.init_noise_sigma

```

我们已准备好编写去噪循环。

```
from tqdm.auto import tqdm

scheduler.set_timesteps(num_inference_steps)

for t in tqdm(scheduler.timesteps):
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)

    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

    # predict the noise residual
    with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # compute the previous noisy sample x_t -> x_t-1
    latents = scheduler.step(noise_pred, t, latents).prev_sample

```

我们现在使用 将生成的解码回图像。`vae``latents`

```
# scale and decode the image latents with vae
latents = 1 / 0.18215 * latents
with torch.no_grad():
    image = vae.decode(latents).sample

```

最后，让我们将图像转换为 PIL，以便我们可以显示或保存它。

```
image = (image / 2 + 0.5).clamp(0, 1)
image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
images = (image * 255).round().astype("uint8")
pil_images = [Image.fromarray(image) for image in images]
pil_images[0]

```

[[图片上传中...(image-4cf375-1678157858351-0)]](https://huggingface.co/blog/assets/98_stable_diffusion/stable_diffusion_k_lms.png) 

我们已经从使用拥抱面扩散器稳定扩散的基本使用到库的更高级使用🤗，我们试图在现代扩散系统中引入所有部分。如果您喜欢本主题并想要了解更多信息，我们建议您使用以下资源：

*   我们的[科拉布笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb)。
*   [扩散器入门](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb)笔记本，其中提供了有关扩散系统的更广泛概述。
*   [带注释的扩散模型](https://huggingface.co/blog/annotated-diffusion)博客文章。
*   我们在 [GitHub 中的代码](https://github.com/huggingface/diffusers)，如果您留下 ⭐ if 对您有用，我们将非常高兴！`diffusers`
