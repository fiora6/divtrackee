import argparse
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from utils.ImagesDataset import ImagesDataset
from pivot_tuning import GeneratorTuning
from adversarial_optimization import DivTrackee
import yaml

def parse_args_from_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description="DivTrackee")

    parser.add_argument('--data_dir', type=str, default='input_images', help='The directory of input images')
    parser.add_argument('--noise_path', type=str, default='noises.pt', help='Path to save the generator noise file')
    parser.add_argument('--inverted_image_path', type=str, default='inverted_imgs', help='Path to save the inverted images in the first stage')
    parser.add_argument('--latent_path', type=str, default='latents.pt', help='Path to the latent file calculated by e4e method')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint_dir', help='Path to the finetuned generator weights in the first stage')
    parser.add_argument('--num_steps', type=int, default=450, help='The number of steps for generator tuning')
    parser.add_argument('--gt_lr', type=float, default=0.0005, help='Learning rate for generator tuning')

    parser.add_argument('--num_aug', type=int, default=1)
    parser.add_argument('--source_text', type=str, default='face')
    parser.add_argument('--makeup_prompt', type=str, default='natural makeup')
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--noise_optimize', type=bool, default=True, help = 'Use noise vectors in StyleGAN during optimization')
    parser.add_argument('--margin', type=int, default=0, help = 'MTCNN margin')
    parser.add_argument('--lambda_lat', type=float, default=0.02)
    parser.add_argument('--lambda_clip', type=float, default=0.3)
    parser.add_argument('--lambda_adv', type=float, default=1)
    parser.add_argument('--lambda_queue', type=float, default=1.2)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the YAML configuration file')
    args = parser.parse_args()
    return args




if __name__ == '__main__':
    # Parse the arguments
    args = parse_args()
    config = parse_args_from_yaml(args.config)

    # 将配置转换为 argparse.Namespace 对象
    args = argparse.Namespace(**config)
    import os
 
    # Define your dataset
    dataset = ImagesDataset(args.data_dir, transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))

    # Create the DataLoader
    args.dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Create an instance of the the stage-1 (generator tuning)
    generator_tuning = GeneratorTuning(args)
    generator_tuning.run()

    # Create an instance of the the stage-2 (adversarial optimization)
    divtrackee = DivTrackee(args)
    divtrackee.run()
