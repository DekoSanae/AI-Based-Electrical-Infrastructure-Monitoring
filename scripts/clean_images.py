import os
from PIL import Image

def clean_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(root, file)

                try:
                    with Image.open(path) as img:
                        if img.mode in ("P", "RGBA", "LA"):
                            rgb_img = img.convert("RGB")
                            rgb_img.save(path, format="JPEG")
                            print(f"Converted and saved: {path}")

                except Exception as e:
                    print(f"Error: {path} -> {e}")

clean_directory("data/train")
clean_directory("data/val")
clean_directory("data/test")