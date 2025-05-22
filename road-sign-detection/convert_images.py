import os
from PIL import Image


def convert_ppm_to_png(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in os.listdir(input_folder):
        if file.endswith(".ppm"):
            ppm_path = os.path.join(input_folder, file)
            png_path = os.path.join(output_folder, file.replace(".ppm", ".png"))

            # Convertir en PNG
            img = Image.open(ppm_path)
            img.save(png_path, "PNG")
            print(f"Converted: {ppm_path} -> {png_path}")


# Chemins des dossiers
test_input = "dataset/test"  # Dossier contenant les images .ppm de test
test_output = "dataset/test_converted"  # Dossier pour les images converties en .png

# Convertir les images
convert_ppm_to_png(test_input, test_output)