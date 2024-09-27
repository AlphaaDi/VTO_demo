from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from attributes_predictor_module.human_traints_predictor import HumanTraitsPredictor
from attributes_predictor_module.openai.garment_attributes import GarmentDescriptionGenerator
from vto_core_module.vto_core import VtoCore

class PipelineLoader:
    def __init__(self, config: dict):
        self.config = config
        self.device = config['device']
        self._load_components()
        self.to(self.device)

    def _load_components(self):
        self.garment_description_generator = GarmentDescriptionGenerator()
        self.parsing_model = Parsing(0)
        self.openpose_model = OpenPose(0)
        self.attributes_classifier = HumanTraitsPredictor(
            self.config['persone_attributes']
        )
        self.vto_core = VtoCore(self.config['core_config'])

    def to(self, device):
        self.openpose_model.preprocessor.body_estimation.model.to(device)
        self.vto_core.to(device)
        self.device = device

    def get_pipeline(self):
        if not hasattr(self, 'pipe') or self.pipe is None:
            raise AttributeError('Loader could not load pipeline, try again')
        return self.pipe

    def get_openpose_model(self):
        if not hasattr(self, 'openpose_model') or self.openpose_model is None:
            raise AttributeError('Loader could not load openpose model, try again')
        return self.openpose_model

    def get_parsing_model(self):
        if not hasattr(self, 'parsing_model') or self.parsing_model is None:
            raise AttributeError('Loader could not load parsing model, try again')
        return self.parsing_model

    def get_attributes_classifier(self):
        if not hasattr(self, 'attributes_classifier') or self.attributes_classifier is None:
            raise AttributeError('Loader could not load attributes classifier, try again')
        return self.attributes_classifier

    def get_vto_core(self):
        if not hasattr(self, 'vto_core') or self.vto_core is None:
            raise AttributeError('Loader could not load vto core, try again')
        return self.vto_core 
    
    def get_garment_description_generator(self):
        if not hasattr(self, 'garment_description_generator') or self.garment_description_generator is None:
            raise AttributeError('Loader could not load garment description generator, try again')
        return self.garment_description_generator 