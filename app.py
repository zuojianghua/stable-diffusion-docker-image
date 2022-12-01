import gradio as gr
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline,StableDiffusionImg2ImgPipeline,EulerDiscreteScheduler
import logging

#### 日志设置
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#### 参数设置
# model_path = "/workspace/models/stable-diffusion-2"
# model_path = "/workspace/models/anything-v3.0"
# model_path = "/workspace/models/openjourney"
text2img:any
img2img:any
model_path:str

#### 载入模型
def change_model(model_name):
    global model_path, text2img, img2img
    model_path = "/workspace/models/" + model_name
    
    #### 开始加载
    logger.info("开始加载")
    scheduler = EulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
    text2img  = StableDiffusionPipeline.from_pretrained(model_path, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16)
    text2img  = text2img.to("cuda")
    # text2img.enable_attention_slicing()
    text2img.enable_xformers_memory_efficient_attention()

    img2img = StableDiffusionImg2ImgPipeline(**text2img.components)
    logger.info("加载结束")
    return gr.update(visible=True), gr.update(visible=True)

#### 初始载入默认模型
change_model('stable-diffusion-2')

#### 主函数
def main_func(prompt,negative_prompt,batchSize,num,width,height,seed,steps,scale,init_img,strength):
    global model_path
    imageData = []
    if model_path=="/workspace/models/openjourney":
        prompt = "mdjrny-v4 style,"+prompt
    for _ in range(num):
        # with torch.cuda.amp.autocast():
        with autocast("cuda"):
            data = text2img(prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=scale, negative_prompt=negative_prompt, num_images_per_prompt=batchSize).images
            imageData.extend(data)
    return imageData

#### 界面
css = """
        .container {
            max-width: 100%;
        }
        #generated_id {
            min-height: 768px;
        }

"""

with gr.Blocks(css=css) as demo:
    with gr.Row():
        with gr.Column(scale=2):
            output = gr.Gallery(show_label=False,elem_id="generated_id",).style(grid=[3],height="768",container=True)
            
        with gr.Column(scale=1):
            prompt = gr.Textbox(lines=4, label='关键字:(正向)',max_lines=4,show_label=True)
            negative_prompt = gr.Textbox(lines=2, label='关键字:(反向)',max_lines=2,show_label=True,value="lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, signature, watermark, username, ugly, duplicate, blurry, morbid, tranny, out of frame, mutation")
            btn = gr.Button("执行",variant="primary")

            with gr.Row():
                with gr.Column(scale=1):
                    width = gr.Slider(512, 1024, step=128, value=768, label='图像宽度(width)')
                    height = gr.Slider(512, 1024, step=128, value=768, label='图像高度(height)')
                    batchSize = gr.Slider(1, 4, step=1, value=1, label='一轮几张(batchSize)')
                    num = gr.Slider(1, 10, step=1, value=1, label='运行几轮(num)')
                    steps = gr.Slider(15, 100, step=5, value=25, label='运行步数(steps)')
                    scale = gr.Slider(5, 15, step=0.5, value=7.5, label='语义接近(scale)')
                    seed = gr.Textbox(lines=1, label='种子(seed)')
                with gr.Column(scale=1):
                    init_img = gr.Image(tool="select",type="filepath",label='参考图:(可不传)')
                    strength = gr.Slider(0.0, 1.0, step=0.05, value=0.75, label='参考程度:(越大越自由)')
                    model_selector = gr.Dropdown([
                        "stable-diffusion-2",
                        "anything-v3.0",
                        "openjourney",
                        "waifu-diffusion"
                    ], value="stable-diffusion-2")
                    btn2 = gr.Button("切换模型")

    btn.click(fn=main_func, inputs=[prompt,negative_prompt,batchSize,num,width,height,seed,steps,scale,init_img,strength], outputs=[output])
    btn2.click(fn=change_model, inputs=[model_selector], outputs=[btn, btn2])

#### 启动程序
demo.queue()
demo.launch(
    show_api=False,
    server_name="0.0.0.0",
    server_port=8888,
    # share=True
)
logger.info("启动界面")
