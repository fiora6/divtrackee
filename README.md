- **Build environment**
```shell
# use anaconda to build environment 
conda create -n div2trackee python=3.8
conda activate div2trackee
# install packages
pip install -r requirements.txt
```

1. Place the pre-trained StyleGAN2 [weights](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view?usp=sharing) in the 'pretrained_models' folder.

2. Download face recognition models at [here](https://drive.google.com/file/d/1Vuek5-YTZlYGoeoqyM5DlvnaXMeii4O8/view) and place them at 'models' folder.

3. Using [e4e](https://github.com/omertov/encoder4editing) method to get the latent codes of facial images

4. Run `main.py`

 ```shell
     python main.py --data_dir input_images --noise_path noises.pt --latent_path latents.pt --checkpoint_dir checkpoint_dir --output_dir output
    


