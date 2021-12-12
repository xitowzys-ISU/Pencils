import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import filters, measure


def draw_images(images: list[np.ndarray], pencils_count: list) -> None:
    fig = plt.figure(figsize=(10, 6), dpi=144)

    box = {'facecolor': 'black',
           'edgecolor': 'red',
           'boxstyle': 'round'}

    for i, image in enumerate(images):
        h, w = image.shape

        left, top = .25, .25
        right, bottom = left + w, top + h

        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.title(f"Image #{i}")
        plt.text(right - 200, bottom - 550, pencils_count[i],
                 fontsize=10,
                 bbox=box,
                 color='white',
                 horizontalalignment='center',
                 verticalalignment='top')
        plt.imshow(image)

    fig.suptitle(f'Всего количество карандашей на всех изображениях: {sum(pencils_count)}', fontsize=16)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    pencils_count = list()

    print("Read images")
    images = [plt.imread(f'./data/img ({i}).jpg') for i in tqdm(range(1, 13))]

    print("Binary images")
    binary_images = [np.mean(image, 2) for image in tqdm(images)]

    print("Threshold images")
    threshold_images = [image < filters.threshold_otsu(image) for image in tqdm(binary_images)]

    print("Search for pencils")
    for image in tqdm(threshold_images):
        pencil_count_image = 0

        # Маркировка
        label = measure.label(image)

        # Измерение свойств помеченных областей изображения
        for region in measure.regionprops(label):
            if region.perimeter > 2500:
                if 30 > (region.major_axis_length / region.minor_axis_length) > 15:
                    pencil_count_image += 1

        pencils_count.append(pencil_count_image)

    draw_images(threshold_images, pencils_count)
