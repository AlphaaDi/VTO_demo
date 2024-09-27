from pydantic import BaseModel
from typing import Optional

class GarmentAttributes(BaseModel):
    type_of_garment: str
    sleeve_length: str
    neckline_or_collar_type: str
    button_or_zipper: Optional[str]
    pattern_or_design: Optional[str]
    dominant_colors: str
    picture_or_graphic: Optional[str]
    material_or_fabric_type: Optional[str]
    fit: Optional[str]
    hemline_length: Optional[str]
    special_features: Optional[str]
    cuffs: Optional[str]
    occasion_or_style: Optional[str]
    brand: Optional[str]


class GarmentDescription(GarmentAttributes):
    attributes: GarmentAttributes
    text_description: str