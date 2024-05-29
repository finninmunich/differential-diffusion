import argparse
import torch
from PIL import Image
from torchvision import transforms

from diff_pipe import StableDiffusionDiffImg2ImgPipeline

device = "cuda"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SD1/2 -- image2image')
    parser.add_argument('--input', type=str, default="assets/input.jpg",
                        help='input image path')
    parser.add_argument('--map', type=str, default="assets/map.jpg", help='input map path')
    parser.add_argument('--output', type=str, default="output.png", help='output image path')
    parser.add_argument('--model-path', type=str,
                        default="/home/turing/cfs_cz/finn/codes/DrivingEdition/examples/text_to_image/stable-diffusion-v1-5")
    args = parser.parse_args()
    # This is the default model, you can use other fine tuned models as well
    pipe = StableDiffusionDiffImg2ImgPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16).to(device)


    def preprocess_image(image):
        image = image.convert("RGB")
        image = transforms.CenterCrop((image.size[1] // 64 * 64, image.size[0] // 64 * 64))(image)
        image = transforms.ToTensor()(image)
        image = image * 2 - 1
        image = image.unsqueeze(0).to(device)
        return image


    def preprocess_map(map):
        map = map.convert("L")
        map = transforms.CenterCrop((map.size[1] // 64 * 64, map.size[0] // 64 * 64))(map)
        # convert to tensor
        map = transforms.ToTensor()(map)
        map = map.to(device)
        return map


    with Image.open(args.input) as imageFile:
        image = preprocess_image(imageFile)

    with Image.open(args.map) as mapFile:
        map = preprocess_map(mapFile)

    edited_image = \
        pipe(prompt=[
            "snowy-day,building,streetlight,sky,tree,road,sidewalk,person,signboard,traffic,grass,plant,fence, "
            "The image depicts an empty city street"],
             image=image,
             guidance_scale=7,
             num_images_per_prompt=1,
             negative_prompt=["sunny-day, blurry, shadow polaroid photo, scary angry pose"], map=map,
             num_inference_steps=999).images[0]
    edited_image.save(args.output)

    print("Done!")
