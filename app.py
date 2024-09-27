import gradio as gr
import yaml
from pipeline_loader import PipelineLoader
from front_end import append_gradio_frontend
from request_processor import TryOnProcessor
from dotenv import load_dotenv

load_dotenv()


with open('pipeline_config.yaml', 'r') as file:
    pipeline_config = yaml.safe_load(file)


pipeline_loader = PipelineLoader(
    base_path=pipeline_config['base_path'],
    device=pipeline_config['device']
)

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
