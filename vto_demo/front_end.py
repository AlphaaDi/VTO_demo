import os
import gradio as gr
from misc.common_utils import get_files_from_dir

# Helper Function: Prepare Human Example List
def prepare_human_examples(human_list_path):
    human_ex_list = []
    for ex_human in human_list_path:
        ex_dict = {
            'background': ex_human,
            'layers': None,
            'composite': None
        }
        human_ex_list.append(ex_dict)
    return human_ex_list

# UI Setup Function: Gradio Components
def setup_gradio_interface(processing_fn, human_ex_list, garm_list_path):
    gr.Markdown("## IDM-VTON Adaptation for HeyGen from Ivan Shpuntov 👕👔👚")
    gr.Markdown("Virtual Try-on with your image and garment image. Check out the [source codes](https://github.com/yisol/IDM-VTON) and the [model](https://huggingface.co/yisol/IDM-VTON)")

    with gr.Row():
        with gr.Column():
            imgs = gr.ImageEditor(sources='upload', type="pil", label='Human. Mask with pen or use auto-masking', interactive=True)

            gr.Examples(
                inputs=imgs,
                examples_per_page=10,
                examples=human_ex_list
            )

        with gr.Column():
            garm_img = gr.Image(label="Garment", sources='upload', type="pil")
            with gr.Row(elem_id="prompt-container"):
                with gr.Row():
                    prompt = gr.Textbox(
                        placeholder="Description of garment ex) Short Sleeve Round Neck T-shirts",
                        value="Short Sleeve Open Front Kimono in Yellow with Tropical Floral Print",
                        show_label=False,
                        elem_id="prompt",
                    )
            gr.Examples(
                inputs=garm_img,
                examples_per_page=8,
                examples=garm_list_path
            )

        with gr.Column():
            masked_img = gr.Image(label="Masked image output", elem_id="masked-img", show_share_button=False)

        with gr.Column():
            image_out = gr.Image(label="Output", elem_id="output-img", show_share_button=False)

    with gr.Column():
        try_button = gr.Button(value="Try-on")
        with gr.Accordion(label="Advanced Settings", open=False):
            with gr.Row():
                denoise_steps = gr.Number(label="Denoising Steps in range [20,100]", minimum=20, maximum=100, value=30, step=1)
                seed = gr.Number(label="Seed in range [-1, 2147483647]", minimum=-1, maximum=2147483647, step=1, value=42)

    try_button.click(
        fn=processing_fn,
        inputs=[
            imgs,
            garm_img,
            prompt,
            denoise_steps,
            seed
        ],
        outputs=[image_out, masked_img],
        api_name='tryon'
    )

# Main Function: Appending Gradio Frontend
def append_gradio_frontend(example_path, processing_fn):
    garm_list_path = get_files_from_dir(os.path.join(example_path, "cloth"))
    human_list_path = get_files_from_dir(os.path.join(example_path, "human"))

    human_ex_list = prepare_human_examples(human_list_path)

    setup_gradio_interface(processing_fn, human_ex_list, garm_list_path)