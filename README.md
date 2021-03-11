# Software 10: Optical Music Recognition

## Training a model with the TF Object Detection API
Consider reading the official [Object Detection API tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html)
for a general guide to the API.
The guide below is specific to this project.
### Installation
Install the required packages from the Pipfile.

Git clone the [TensorFow Model Garden](https://github.com/tensorflow/models)
repo into `sw10/`. This repo contains the Object Detection API (in 
`models/research/object_detection`).

Install protobuf by [downloading](https://github.com/protocolbuffers/protobuf/releases)
the appropriate release for you OS. Extract the contents of the zip file, and 
add that directory to $PATH, by addding it to home/.bashrc.
Example: `export PATH=/home/username/protoc/bin:$PATH`.

Compile protobuf libraries (from within models/research):

```bash
protoc object_detection/protos/*.proto --python_out=.
```

More details about the last two steps can be found in the Object Detection API
[installation guide](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html).

Furthermore, cd into models/research and run:
```bash
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

This last step has to be repeated each time you wish to use code from the
Object Detection API in a new session.

### Data retrieval and preprocessing
To get the MUSCIMA++ dataset, run 

```bash
python src/data_handling/download_muscima.py
```

The Object Detection API requires the training data in TF Record files, and
these can be generated directly from xml files in Pascal VOC format.
Run the following command to generate Pascal VOC xml files from the annotations
(currently only supports full-page training):

```bash
python src/data_handling/prepare_muscima_full_page_annotations.py --destination_dir=obj_det_api_training/data/MUSCIMA++/full_page/annotations
```

Split the data into train, validation, and test:

```bash
python src/data_handling/dataset_splitter.py --source_directory=data/MUSCIMA++/v2.0/data/images --destination_directory=obj_det_api_training/data/MUSCIMA++/full_page
```

The Object Detection API requires a mapping of class label names to id's, which can
be generated by running:

```bash
python src/data_handling/generate_mapping_for_muscima_pp.py
```

Finally, the TensorFlow record files for training, validation, and testing can
be generated from the Pascal VOC xml files and the label mapping:

```bash
python src/data_handling/create_muscima_tf_record.py --data_dir=obj_det_api_training/data/MUSCIMA++/full_page/ --set=training --annotations_dir=annotations --output_path=obj_det_api_training/data/MUSCIMA++/full_page/train.record --label_map_path=obj_det_api_training/data/MUSCIMA++/mapping_all_classes.pbtxt
python src/data_handling/create_muscima_tf_record.py --data_dir=obj_det_api_training/data/MUSCIMA++/full_page/ --set=validation --annotations_dir=annotations --output_path=obj_det_api_training/data/MUSCIMA++/full_page/validation.record --label_map_path=obj_det_api_training/data/MUSCIMA++/mapping_all_classes.pbtxt
python src/data_handling/create_muscima_tf_record.py --data_dir=obj_det_api_training/data/MUSCIMA++/full_page/ --set=test --annotations_dir=annotations --output_path=obj_det_api_training/data/MUSCIMA++/full_page/test.record --label_map_path=obj_det_api_training/data/MUSCIMA++/mapping_all_classes.pbtxt
```

### Training

Download a pre-trained model from the [TF2 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
and extract the model folder to `sw10/obj_det_api_training/pre_trained_models/`

The models we currently have customised pipeline.config files for:
- SSD ResNet152 V1 FPN 1024x1024 (RetinaNet152)
- SSD MobileNet v2 320x320
- Faster R-CNN Inception ResNet V2 1024x1024 (**WARNING**: running this model
  currently produces an error)
  
If you download a pre-trained model which is not listed above, follow
[this guide](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html)
under the "Configure The Training Pipeline" section to set up the
`pipeline.config` file for the model, and place it in
`obj_det_api_training/models/[MODEL_NAME]/`.

To begin training, insert the correct folder names into the following command
and run it:

```bash
python src/run_model_w_obj_det_api.py --model_dir=obj_det_api_training/models/[MODEL FOLDER] --pipeline_config_path=obj_det_api_training/models/[MODEL FOLDER]/pipeline.config
```

If it fails because a module could not be found, run 
```export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim``` in models/research again.
As mentioned, this needs to be done each time a new session is started.

### Evaluation

To evaluate a model (during or after training), insert the correct folder names
into the following command and run it:

```bash
python src/run_model_w_obj_det_api.py --model_dir=obj_det_api_training/models/[MODEL FOLDER] --pipeline_config_path=obj_det_api_training/models/[MODEL FOLDER]/pipeline.config --checkpoint_dir=obj_det_api_training/models/[MODEL FOLDER]
```

The model is evaluated using the specified metric in the pipeline.config file. 
The latest checkpoint of the model is used for evaluation, and the evaluation
process will check every 300 seconds for a new checkpoint to use.

The results are stored in TF event files at
`obj_det_api_training/models[MODEL FOLDER]/eval_0`.
These results can be monitored using TensorBoard by setting logdir to
`obj_det_api_training/models/[MODEL_NAME]`.

For example: 
```bash
tensorboard --logdir=obj_det_api_training/models/custom_ssd_mobilenet/
```