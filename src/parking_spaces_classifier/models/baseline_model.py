import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_baseline_detector(num_classes: int = 3, freeze_all_but_head: bool = False) -> FasterRCNN:
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    
    if freeze_all_but_head:
        for param in model.parameters():
            param.requires_grad = False
            
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model
