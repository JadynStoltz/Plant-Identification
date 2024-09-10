import os
import requests
from PIL import Image
from io import BytesIO
import random
import time

# Pixabay API key
PIXABAY_API_KEY = '45268573-b8f1b0e4aaef89e7a84e8d16e'
BASE_URL = 'https://pixabay.com/api/'

# Set up paths
base_dir = "C:/Users/jadyn/Documents/Plant Identification/archive/split_ttv_dataset_type_of_plants"
train_dir = os.path.join(base_dir, 'Train_Set_Folder')
validation_dir = os.path.join(base_dir, 'Validation_Set_Folder')
test_dir = os.path.join(base_dir, 'Test_Set_Folder')

# Ensure directories exist
for directory in [train_dir, validation_dir, test_dir]:
    os.makedirs(directory, exist_ok=True)

def fetch_image_urls(plant_name, max_images=200):
    search_url = f"{BASE_URL}?key={PIXABAY_API_KEY}&q={requests.utils.quote(plant_name)}&image_type=photo&per_page={max_images}"
    response = requests.get(search_url)
    data = response.json()
    if 'hits' not in data:
        print(f"No images found for: {plant_name}")
        return []

    image_urls = [hit['webformatURL'] for hit in data['hits']]
    return image_urls

def download_image(download_path, url, file_name):
    try:
        image_content = requests.get(url).content
        image_file = BytesIO(image_content)
        try:
            img = Image.open(image_file)
            img.verify()  # Verify that it's a valid image
            img_file = Image.open(BytesIO(image_content))
            width, height = img_file.size
            if width > 100 and height > 100:  # Check for minimum dimensions
                file_path = os.path.join(download_path, file_name)
                img_file.save(file_path)
                print(f"Downloaded {file_name}")
                return True
            else:
                print(f"Skipping small image {file_name}")
        except (IOError, SyntaxError) as e:
            print(f"Skipping invalid image {file_name}: {e}")
    except Exception as e:
        print(f"FAILED - {e}")
    return False

def fetch_and_download_images(plant_name, train_dir, validation_dir, test_dir, images_per_split=60):
    image_urls = fetch_image_urls(plant_name, max_images=200)

    if not image_urls:
        return

    # Shuffle the URLs
    random.shuffle(image_urls)

    # Create folder for plant
    folder_name = plant_name.replace(" Plant", "")
    train_folder = os.path.join(train_dir, folder_name)
    validation_folder = os.path.join(validation_dir, folder_name)
    test_folder = os.path.join(test_dir, folder_name)

    for folder in [train_folder, validation_folder, test_folder]:
        os.makedirs(folder, exist_ok=True)

    def download_images(urls, download_path, prefix, count):
        downloaded = 0
        for i, url in enumerate(urls):
            if downloaded >= count:
                break
            file_name = f"{prefix}_{i + 1}.jpg"
            if download_image(download_path, url, file_name):
                downloaded += 1
        return downloaded

    # Download images into respective folders
    train_downloaded = download_images(image_urls, train_folder, folder_name, images_per_split)
    validation_downloaded = download_images(image_urls[train_downloaded:], validation_folder, folder_name, images_per_split)
    test_downloaded = download_images(image_urls[train_downloaded + validation_downloaded:], test_folder, folder_name, images_per_split)

    print(f"Downloaded {train_downloaded} training images, {validation_downloaded} validation images, and {test_downloaded} test images for {plant_name}")

# List of 50 common household plants, herbs, and fruits
plant_names = [
    "Aloe Vera Plant", "Basil Plant", "Mint Plant", "Rosemary Plant", "Thyme Plant",
    "Lavender Plant", "Oregano Plant", "Cilantro Plant", "Parsley Plant", "Chives Plant",
    "Sage Plant", "Dill Plant", "Lemon Balm Plant", "Chamomile Plant", "Peppermint Plant",
    "Tomato Plant", "Bell Pepper Plant", "Cucumber Plant", "Zucchini Plant", "Eggplant Plant",
    "Strawberry Plant", "Blueberry Plant", "Raspberry Plant", "Lemon Tree Plant", "Lime Tree Plant",
    "Apple Tree Plant", "Cherry Tree Plant", "Peach Tree Plant", "Pear Tree Plant", "Plum Tree Plant",
    "Lettuce Plant", "Spinach Plant", "Kale Plant", "Arugula Plant", "Swiss Chard Plant",
    "Carrot Plant", "Radish Plant", "Beet Plant", "Onion Plant", "Garlic Plant",
    "Potato Plant", "Sweet Potato Plant", "Pumpkin Plant", "Squash Plant", "Sunflower Plant",
    "Marigold Plant", "Geranium Plant", "Begonia Plant", "Philodendron Plant", "Spider Plant"
]

# Calculate estimated time
estimated_time_per_plant = 2  # minutes
total_estimated_time = len(plant_names) * estimated_time_per_plant
print(f"Estimated time to download all images: {total_estimated_time} minutes")

# Fetch and download images for each plant
start_time = time.time()
for plant in plant_names:
    print(f"Fetching images for: {plant}")
    fetch_and_download_images(plant, train_dir, validation_dir, test_dir)

end_time = time.time()
total_time = (end_time - start_time) / 60  # Convert to minutes
print(f"Finished downloading images. Total time taken: {total_time:.2f} minutes")