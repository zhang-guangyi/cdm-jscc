import os
from PIL import Image

input_folder = './SharedData/Kodak'    # Kodak folder
output_folder = './SharedData/cropped_Kodak'    # output cropped folder

os.makedirs(output_folder, exist_ok=True)


for img_name in os.listdir(input_folder):
    img_path = os.path.join(input_folder, img_name)
    if img_name.lower().endswith(('png', 'jpg', 'jpeg')):
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                if width < 256 or height < 256:
                    print(f"Skipping {img_name}, image size {width}x{height} is smaller than 256x256.")
                    continue
                top=0
                img_name, img_ext = os.path.splitext(img_name)
                while (top+256) <= height:
                    bottom = top + 256
                    left=0
                    while (left+256) <= width:
                        right = left + 256
                        cropped_img = img.crop((left, top, right, bottom))
                        output_path = os.path.join(output_folder, img_name+'_left'+str(left//256)+'_top'+str(top//256)+img_ext)
                        cropped_img.save(output_path)
                        print(f"Saved cropped image: {output_path}")
                        left = right
                    top = bottom
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
