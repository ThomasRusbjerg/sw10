# Software 10: Optical Music Recognition

## Training a model with the TF Object Detection API
### Installation
Install the required packages from the Pipfile.

Set up PATH environment by adding the following lines to ~/.bashrc:
```bash
export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

Compile protobuf libraries (from within models/research):

```bash
protoc object_detection/protos/*.proto --python_out=.
```

More details about the last two steps can be found in the Object Detection API
[installation guide](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html).




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
python src/data_handling/prepare_muscima_full_page_annotations.py
```

Split the data into train, validation, and test:

```bash
python src/data_handling/dataset_splitter.py --source_directory=data/MUSCIMA++/full_page_annotations/annotations --destination_directory=data/MUSCIMA++/full_page_annotations/
```

The Object Detection API requires a mapping of class label names to id's, which can
be generated by running:

```bash
python src/data_handling/generate_mapping_for_muscima_pp.py
```

Finally, the TensorFlow record files for training, validation, and testing can
be generated from the Pascal VOC xml files and the label mapping:

```bash
python src/data_handling/generate_tfrecord.py -x data/full_page_annotations/training/ -l data/MUSCIMA++/mapping_all_classes.pbtxt -o data/MUSCIMA++/full_page_annotations/train.record
python src/data_handling/generate_tfrecord.py -x data/full_page_annotations/validation/ -l data/MUSCIMA++/mapping_all_classes.pbtxt -o data/MUSCIMA++/full_page_annotations/validation.record
python src/data_handling/generate_tfrecord.py -x data/full_page_annotations/test/ -l data/MUSCIMA++/mapping_all_classes.pbtxt -o data/MUSCIMA++/full_page_annotations/test.record
```

### Training


