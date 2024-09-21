import gradio as gr
import yaml
from pipeline_loader import PipelineLoader
from front_end import append_gradio_frontend
from request_processor import TryOnProcessor


with open('pipeline_config.yaml', 'r') as file:
    pipeline_config = yaml.safe_load(file)


pipeline_loader = PipelineLoader(
    base_path=pipeline_config['base_path'],
    device=pipeline_config['device']
)
pipe = pipeline_loader.get_pipeline()
openpose_model = pipeline_loader.get_openpose_model()
parsing_model = pipeline_loader.get_parsing_model()
tensor_transform = pipeline_loader.get_tensor_transform()

try_on_processor = TryOnProcessor(
    pipeline_config, pipeline_loader
)

image_blocks = gr.Blocks().queue()
with image_blocks as demo:
    append_gradio_frontend(
        example_path = pipeline_config['example_path'],
        processing_fn=try_on_processor.start_tryon
    )

image_blocks.launch(server_port=7834, debug=True)
