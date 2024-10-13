import os
from src.base import *

if __name__ == "__main__":
    # Preprocessing transformations (uncomment one)
    # T = preprocess_1()
    T = preprocess_2()
    # T = preprocess_3()

    # Vein extraction algorithms
    E = [extract_rtl(), extract_mc(), extract_wl(), extract_pc()]

    # Examples finger images to process
    IMGS = [f"vein_imgs/{img}" for img in os.listdir("vein_imgs")]
    procimg = ImagePreprocessor()
    extractor = VeinExtractor()

    # Apply preprocessing transformations to each image
    processed = []
    for img in IMGS:
        procimg.load_image(img)
        processed_data = procimg.preprocess(*T)
        # procimg.show_all_transformations() # Uncomment to display all transformations
        processed.append((img, processed_data)) # Store the processed data for vein extraction

    # Extract and display extracted veins for each preprocessed image
    for (img_name, img_and_mask) in processed:
        extractor.extract(E, img_and_mask)
        extractor.show_veins(img_name)
