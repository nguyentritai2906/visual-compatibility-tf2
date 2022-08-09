
# DEMO Context-Aware Visual Compatibility Prediction

## Requirements
    pip install -r requirements.txt

## Download
### Download the demo dataset
* [images](https://drive.google.com/file/d/1XQ9QuZDPM-Z-jDQpcU9uuUJ9xPBlk01H/view?usp=sharing) 
* [dataset](https://drive.google.com/file/d/1l7XycWYVhOObq4LY4LVV9cDxC_gxxe_g/view?usp=sharing)

**NOTE: After the download process is completed, you need to extract all of them and put them like the directory tree in below:**

```commandline
data
└── polyvore
    ├── utils
    ├── dataset    
    └── images
```
### Download the checkpoint 
* [checkpoint](https://drive.google.com/file/d/1kU6lCRzo4ugGH4wyL-I89lW9e2JAbKPq/view?usp=sharing)

**NOTE: After the download process is completed, you need to extract them.**

## Run demo 
```commandline
python vis_out.py -q "119314458_1" -q "119314458_2" -q "119314458_3" -q "119314458_4" -n 5 -lf /content/visual-compatibility-tf2/ckpt -k 15
```
### Parameters:
* `-q`: setid_index of the question (setid is the folder name contain the image, index is the name of the image on **the images folder**)
* `-n`: the number of answers you want to see 
* `-lf`: directory of the **checkpoint folder**
* `-k`: the number of neighbourhoods

