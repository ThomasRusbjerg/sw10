from platform import node
import collections

import numpy as np
from detectron2.evaluation import DatasetEvaluator

from mung.io import read_nodes_from_file
# from mung.graph import NotationGraph
# from mung.node import Node
from muscima.cropobject import CropObject

from data_handling.detectron2_muscima import get_muscima_classid_mapping

from models.detr.util.box_ops import box_xyxy_to_cxcywh
from data_handling.evaluate_notation_assembly_from_mung import evaluate_result


class MuNGEvaluator(DatasetEvaluator):
    def reset(self):
        self.precisions = []
        self.recalls = []
        self.f1_scores = []

    def process(self, inputs, outputs):
        # Class to id mappings
        class_mappings = get_muscima_classid_mapping()
        # Iterate images in batch
        for img in range(0, len(outputs)):
            # Get data for img
            instances = outputs[img]["instances"]
            boxes = instances.pred_boxes.tensor
            classes = instances.pred_classes
            relations = instances.pred_relations

            # Create CropObjects for predictions
            pred_crop_obj = []
            for pred in range(0, len(classes)):
                # BBox (left, top, width, height)
                [left, top, _, _] = boxes[pred]
                [_, _, width, height] = box_xyxy_to_cxcywh(boxes[pred])

                # Reverse class->id dictionary (+1 since we did -1 in detectron)
                class_name = list(class_mappings.keys())[list(class_mappings.values()).index(classes[pred]+1)]

                # Get relations (select row; if col > threshold: rel from row to col=yes)
                outlinks = []
                for id, rel in enumerate(relations[pred]):
                    if rel > 0.5:
                        outlinks.append(id)
                # Create CropObject
                pred_crop_obj.append(CropObject(pred, clsname=class_name, top=top, left=left, height=height, width=width, outlinks=outlinks))

            # Load mung from file based on img name
            img_file = inputs[img]["file_name"]
            annotaion_file = img_file.replace("images", "annotations")[:-3] + "xml"
            gt_mung = (read_nodes_from_file(annotaion_file))
            # Convert mung Node to CropObject
            gt_crop_obj = []
            for i, mung in enumerate(gt_mung):
                gt_crop_obj.append(CropObject(i, clsname=mung.class_name, top=mung.top, left=mung.left, height=mung.height, width=mung.width, outlinks=mung.outlinks))

            # Calculate scores for all objects in one image using munglinker code
            precision, recall, f1_score, true_positives, false_positives, false_negatives = \
                evaluate_result(gt_crop_obj, pred_crop_obj)
            self.precisions.append(precision)
            self.recalls.append(recall)
            self.f1_scores.append(f1_score)

    def evaluate(self):
        results = {
            "precision": np.average(self.precisions),
            "recall": np.average(self.recalls),
            "f1_score": np.average(self.f1_scores)
        }
        return collections.OrderedDict({"relations": results})
