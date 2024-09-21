import gradio as gr
from pipeline_loader import PipelineLoader
from front_end import append_gradio_frontend
from request_processor import TryOnProcessor


device = 'cuda'
base_path = 'yisol/IDM-VTON'
example_path = './example'

pipeline_loader = PipelineLoader(base_path=base_path, device=device)
pipe = pipeline_loader.get_pipeline()
openpose_model = pipeline_loader.get_openpose_model()
parsing_model = pipeline_loader.get_parsing_model()
tensor_transform = pipeline_loader.get_tensor_transform()

try_on_processor = TryOnProcessor(device, pipe, openpose_model, parsing_model, tensor_transform)

image_blocks = gr.Blocks().queue()
with image_blocks as demo:
    append_gradio_frontend(example_path = example_path, processing_fn=try_on_processor.start_tryon)

image_blocks.launch(server_port=7834, debug=True)
