import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel

class HumanTraitsPredictor:
    def __init__(self, config):
        super(HumanTraitsPredictor, self).__init__()

        # Set device, model, and processor based on the config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        self.model = CLIPModel.from_pretrained(config['model_name']).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(config['model_name'])

        # Set texts and labels from the config
        self.attributes = config['attributes']

    def process_text_and_predict(self, texts, labels, image_features):
        # Process the input texts and compute text features
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        
        # Compute similarity and get the predicted label
        similarity = torch.matmul(image_features, text_features.T)
        predicted_index = similarity.argmax(dim=-1).item()
        predicted_label = labels[predicted_index]
        
        return predicted_label

    def forward(self, image):
        """
        Predicts the gender and age attributes from an input image.

        Args:
            image (PIL Image or List[PIL Image]): The input image of a person.

        Returns:
            Tuple[str, str]: Predicted gender and age labels.
        """
        # Process the image
        image_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        # Get image features
        with torch.no_grad():
            image_features = self.model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        predicted_attributes = {}
        for attribute_name, attribute_info in self.attributes.items():
            predicted_attribute = self.process_text_and_predict(
                texts=attribute_info["texts"], 
                labels=attribute_info["labels"], 
                image_features=image_features,
            )
            predicted_attributes[attribute_name] = predicted_attribute
        
        return predicted_attributes
