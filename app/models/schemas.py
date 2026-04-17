from pydantic import BaseModel
from typing import Dict

class AnalyzeResponse(BaseModel):
    deficiency_percentage: float
    status: str
    indices: Dict[str, float]
    heatmap_path: str
