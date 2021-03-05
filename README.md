# Palaeoanalytics
Repository for the Paleoanalytics project.

# Installation

The `pylithics` package requires Python 3.6 or greater. To install, start by creating a fresh conda environment.
```
conda create -n paleo python=3.7
conda activate paleo
```

Get the source.
```
git clone https://github.com/alan-turing-institute/Palaeoanalytics.git
```

Enter the repository and check out a relevant branch if necessary (the develop branch contains the most up to date stable version of the code).
```
cd Palaeoanalytics
git checkout develop
```
Install the package using `pip`.
```
pip install .
```

# Running pylithics


*pylithics* can be run via command line. The following command displays all available options:

```bash
pylithics_run --help
```

Output:

```bash
usage: pylithics_run [-h] -c config-file [--input_dir INPUT_DIR]
                     [--output_dir OUTPUT_DIR]

Run lithics characterisation pipeline

optional arguments:
  -h, --help            show this help message and exit
  -c config-file, --config config-file
                        the model config file (YAML)
  --input_dir INPUT_DIR
                        directory where the input images are
  --output_dir OUTPUT_DIR
                        directory where the output data is saved
  --metadata_filename FILENAME
                        Name of the metadata CSV file that pairs scales and lithics

```

For example, given that you have a set of lithics images (and it respective scales), you can run the pylithics processing script with the
following:

```
pylithics_run -c configs/test_config.yml --input_dir <path_to_input_images_dir> --output_dir <path_to_output_directory> --metadata_filename metatada_file.csv
```

This ```test_config.yml``` config file contains the following options:


```yaml

threshold: 0.01
contour_parameter: 0.1
contour_fully_connected: 'low'
minimum_pixels_contour: 0.01
denoise_weight: 0.06
contrast_stretch: [4, 96]


```

You can modify or create your on config file and provide it to the CLI. 

The images found in ```<path_to_input_images_dir>``` should follow the this directory structure:

```bash
input_directory
   ├── metatada_file.csv
   ├── images 
        ├── id1_lithics.png
        ├── id2_lithics.png
        ├── id2_scale.png
        ├── id3_lithics.png
        └── id3_scale.png
            .
            .
            .
        ├── idn_lithics.png
   └──  scales
        ├── id1_scale.png
        ├── id2_scale.png
        ├── id3_scale.png
            .
            .
            .
        └── idn_scale.png



```

where the mapping between the lithics and scale images should be available in the metadata CSV file. 

This CSV file should have as a minimum the following 3 variables:
 
- *PA_ID*:  corresponding the the lithics image id
(the name of the image file), 
- *scale_ID*: The scale id (name of the scale image file)
- *PA_scale*: The scale measurement (how many centemeters this scale represents).

An example of this table is the following:

|PA_ID | scale_ID  | PA_scale  | 
|------|-----------|-----------|
| 1    | sc1       | 5         | 
|------|-----------|-----------|
| 2    | sc1       | 5         |
|------|-----------|-----------|
| 3    | sc1       | 5         |   
|------|-----------|-----------|

## Note:

In the scenario that the scale and csv file are not avalaible, it is possible to run the analysis only using the images
with the command:

```
pylithics_run -c configs/test_config.yml --input_dir <path_to_input_images_dir> --output_dir <path_to_output_directory> 
```
lithics image files must still be inside  the '<path_to_input_images_dir>/images/' directory). However all the measurements will only be
provided as number of pixels. 




