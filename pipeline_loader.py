from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from attributes_predictor_module.attributes_predictor import PersonAttributesClassifier
from vto_core_module.vto_core import VtoCore

class PipelineLoader:
    def __init__(self, base_path: str, config: dict, device: str = "cuda"):
        self.base_path = base_path
        self.config = config
        self.device = device
        self._load_components()
        self.to(self.device)

    def _load_components(self):
        self.parsing_model = Parsing(0)
        self.openpose_model = OpenPose(0)
        self.attributes_classifier = PersonAttributesClassifier(
            self.config['persone_attributes']
        )
        self.vto_core = VtoCore(self.config['core_config'])

    def to(self):
        self.openpose_model.preprocessor.body_estimation.model.to(self.device)
        self.vto_core.to(self.device)

    def get_pipeline(self):
        if not hasattr(self, 'pipe') or not self.pipe:
            raise AttributeError('Loader could not load pipeline, try again')
        return self.pipe

    def get_openpose_model(self):
        if not hasattr(self, 'openpose_model') or not self.pipe:
            raise AttributeError('Loader could not load openpose model, try again')
        return self.openpose_model

    def get_parsing_model(self):
        if not hasattr(self, 'parsing_model') or not self.pipe:
            raise AttributeError('Loader could not load parsing model, try again')
        return self.parsing_model

    def get_attributes_classifier(self):
        if not hasattr(self, 'attributes_classifier') or not self.pipe:
            raise AttributeError('Loader could not load attributes classifier, try again')
        return self.attributes_classifier

    def get_vto_core(self):
        if not hasattr(self, 'vto_core') or not self.pipe:
            raise AttributeError('Loader could not load vto core, try again')
        return self.vto_core 
