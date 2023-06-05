import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image, ImageColor
import matplotlib.cm as cm
import numpy as np
import os
from skimage import measure
from constants_v1 import *


def outline_black(image):
    image.convert("RGBA")
    curr_pixel = image.getpixel((0, 0))
    for x in range(image.width):
        for y in range(image.height):
            if image.getpixel((x, y)) == curr_pixel:
                continue
            else:
                if image.getpixel((x, y)) in RGB_VALUES:
                    curr_pixel = image.getpixel((x, y))
                    image.putpixel((x, y), (0, 0, 0, 0))

    curr_pixel = image.getpixel((0, 0))
    for y in range(image.height):
        for x in range(image.width):
            if image.getpixel((x, y)) == curr_pixel:
                continue
            else:
                if image.getpixel((x, y)) in RGB_VALUES:
                    curr_pixel = image.getpixel((x, y))
                    image.putpixel((x, y), (0, 0, 0, 0))
    
    return image

def outline2(image, color_list, T1, width, height, mode):
    image = image.convert("RGBA")
    plt.imshow(T1, cmap='gray', interpolation='none')
    plt.axis('off')
    plt.savefig(f'temp.png', format = 'png', dpi=300, bbox_inches='tight')
    T1 = Image.open('temp.png')
    T1 = T1.convert("RGBA")
    if mode != 'axial':
        T1 = T1.rotate(90, expand=True)
    T1 = T1.resize((width, height), resample=Image.NEAREST)
    outlined_image = Image.new("RGBA", image.size, (0, 0, 0, 0))
    for _num, color_to_outline in enumerate(color_list):
        inside = False
        for x in range(image.width):
            for y in range(image.height):
                if inside:
                    if image.getpixel((x, y)) == color_to_outline:
                        continue
                    else:
                        outlined_image.putpixel((x, y), color_to_outline)
                        #outlined_image.putpixel((x, y - 1), color_to_outline)
                        inside = False
                else:
                    if image.getpixel((x, y)) == color_to_outline:
                        outlined_image.putpixel((x, y), color_to_outline)
                        #outlined_image.putpixel((x, y - 1), color_to_outline)
                        inside = True

        inside = False
        for y in range(image.height):
            for x in range(image.width):
                if inside:
                    if image.getpixel((x, y)) == color_to_outline:
                        pass
                    else:
                        outlined_image.putpixel((x, y), color_to_outline)
                        #outlined_image.putpixel((x - 1, y), color_to_outline)
                        inside = False
                else:
                    if image.getpixel((x, y)) == color_to_outline:
                        outlined_image.putpixel((x, y), color_to_outline)
                        #outlined_image.putpixel((x - 1, y), color_to_outline)
                        inside = True
    os.remove('temp.png')
    return Image.alpha_composite(T1, outlined_image)

def outline2_no_T1(image, color_list):
    image = image.convert("RGBA")
    outlined_image = Image.new("RGBA", image.size, (0, 0, 0, 0))
    for _num, color_to_outline in enumerate(color_list):
        inside = False
        for x in range(image.width):
            for y in range(image.height):
                if inside:
                    if image.getpixel((x, y)) == color_to_outline:
                        continue
                    else:
                        outlined_image.putpixel((x, y), color_to_outline)
                        #outlined_image.putpixel((x, y - 1), color_to_outline)
                        inside = False
                else:
                    if image.getpixel((x, y)) == color_to_outline:
                        outlined_image.putpixel((x, y), color_to_outline)
                        #outlined_image.putpixel((x, y - 1), color_to_outline)
                        inside = True

        inside = False
        for y in range(image.height):
            for x in range(image.width):
                if inside:
                    if image.getpixel((x, y)) == color_to_outline:
                        pass
                    else:
                        outlined_image.putpixel((x, y), color_to_outline)
                        #outlined_image.putpixel((x - 1, y), color_to_outline)
                        inside = False
                else:
                    if image.getpixel((x, y)) == color_to_outline:
                        outlined_image.putpixel((x, y), color_to_outline)
                        #outlined_image.putpixel((x - 1, y), color_to_outline)
                        inside = True
    return outlined_image

def max_height(o_width, o_height, new_height):
    return int(o_width / o_height * new_height)

def max_width(o_width, o_height, new_width):
    return int(o_width / o_height * new_width)

def generate_image(color_list, outline_list, T1_load, image_load, axial, coronal, saggital, sub_name, mode, length, width, height):
    T1_axial = T1_load[:, :, axial]
    image = image_load[:, :, axial]
    plt.imshow(T1_axial, cmap='gray', interpolation='none')
    plt.axis('off')
    plt.savefig(f'temp1.png', format = 'png', dpi=300, bbox_inches='tight')
    for num, color in enumerate(color_list):
        if color == BLACK_TRANSPARENT:
            continue
        cmap = ListedColormap([color])
        cmap.set_over((0, 0, 0, 0))
        cmap.set_under((0, 0, 0, 0))
        cmap.set_bad((0, 0, 0, 0))
        plt.imshow(image, cmap=cmap, interpolation='none', vmin=RANGES[num][0], vmax=RANGES[num][1])
    plt.axis('off')
    plt.savefig(f'{sub_name}_a_{mode}.png', format = 'png', dpi=300, bbox_inches='tight',transparent=True)
    mask = Image.open(f'{sub_name}_a_{mode}.png')
    T1 = Image.open('temp1.png')
    mask = Image.alpha_composite(T1, mask)
    mask.save(f'{sub_name}_a_{mode}.png', format='PNG')
    
    T1_coronal = T1_load[:, coronal, :]
    image = image_load[:, coronal, :]
    plt.imshow(T1_coronal, cmap='gray', interpolation='none')
    plt.axis('off')
    plt.savefig(f'temp1.png', format = 'png', dpi=300, bbox_inches='tight')
    for num, color in enumerate(color_list):
        if color == BLACK_TRANSPARENT:
            continue
        cmap = ListedColormap([color])
        cmap.set_over((0, 0, 0, 0))
        cmap.set_under((0, 0, 0, 0))
        cmap.set_bad((0, 0, 0, 0))
        plt.imshow(image, cmap=cmap, interpolation='none', vmin=RANGES[num][0], vmax=RANGES[num][1])
    plt.axis('off')
    plt.savefig(f'{sub_name}_c_{mode}.png', format = 'png', dpi=300, bbox_inches='tight', transparent=True)
    mask = Image.open(f'{sub_name}_c_{mode}.png')
    T1 = Image.open('temp1.png')
    mask = Image.alpha_composite(T1, mask)
    mask.save(f'{sub_name}_c_{mode}.png', format='PNG')

    T1_saggital = T1_load[saggital, :, :]
    image = image_load[saggital, :, :]
    plt.imshow(T1_saggital, cmap='gray', interpolation='none')
    plt.axis('off')
    plt.savefig(f'temp1.png', format = 'png', dpi=300, bbox_inches='tight')
    for num, color in enumerate(color_list):
        if color == BLACK_TRANSPARENT:
            continue
        cmap = ListedColormap([color])
        cmap.set_over((0, 0, 0, 0))
        cmap.set_under((0, 0, 0, 0))
        cmap.set_bad((0, 0, 0, 0))
        plt.imshow(image, cmap=cmap, interpolation='none', vmin=RANGES[num][0], vmax=RANGES[num][1])
    plt.axis('off')
    plt.savefig(f'{sub_name}_s_{mode}.png', format = 'png', dpi=300, bbox_inches='tight', transparent=True)
    mask = Image.open(f'{sub_name}_s_{mode}.png')
    T1 = Image.open('temp1.png')
    mask = Image.alpha_composite(T1, mask)
    mask.save(f'{sub_name}_s_{mode}.png', format='PNG')

    images = [f'{sub_name}_a_{mode}.png', f'{sub_name}_c_{mode}.png', f'{sub_name}_s_{mode}.png']
    list_im = [Image.open(i) for i in images]
    list_im[1] = list_im[1].rotate(90, expand=True)
    list_im[1] = list_im[1].transpose(method=Image.FLIP_LEFT_RIGHT)
    list_im[2] = list_im[2].rotate(90, expand=True)
    #list_im = [i.resize((max_width_, max_height_)) for i in list_im]
    list_im[0] = list_im[0].resize((length, width), resample=Image.NEAREST)
    list_im[1] = list_im[1].resize((width, height), resample=Image.NEAREST)
    list_im[2] = list_im[2].resize((length, height), resample=Image.NEAREST)

    outline_im = []
    outline_im.append(outline2(list_im[0], outline_list, T1_axial, length, width, 'axial'))
    outline_im.append(outline2(list_im[1], outline_list, T1_coronal, width, height, 'coronal'))
    outline_im.append(outline2(list_im[2], outline_list, T1_saggital, length, height, 'saggital'))
    list_im = [outline_black(i) for i in list_im]

    outline_im = np.vstack(outline_im)
    list_im = np.vstack(list_im)

    imgs_comb = Image.fromarray(list_im)
    imgs_comb_out = Image.fromarray(outline_im)

    # imgs_comb = imgs_comb.convert("RGBA")
    imgs_comb.save(f'{sub_name}_{mode}.png', format='PNG')
    imgs_comb_out.save(f'{sub_name}_{mode}_outline.png', format='PNG')
    for image in images:
        os.remove(image)
    os.remove('temp1.png')

def generate_image_no_T1(color_list, outline_list, image_load, axial, coronal, saggital, sub_name, mode, length, width, height):
    image = image_load[:, :, axial]
    for num, color in enumerate(color_list):
        if color == BLACK_TRANSPARENT:
            continue
        cmap = ListedColormap([color])
        cmap.set_over((0, 0, 0, 0))
        cmap.set_under((0, 0, 0, 0))
        cmap.set_bad((0, 0, 0, 0))
        plt.imshow(image, cmap=cmap, interpolation='none', vmin=RANGES[num][0], vmax=RANGES[num][1])
    plt.axis('off')
    plt.savefig(f'{sub_name}_a_{mode}.png', format = 'png', dpi=300, bbox_inches='tight',transparent=True)
    plt.clf()
    
    image = image_load[:, coronal, :]
    for num, color in enumerate(color_list):
        if color == BLACK_TRANSPARENT:
            continue
        cmap = ListedColormap([color])
        cmap.set_over((0, 0, 0, 0))
        cmap.set_under((0, 0, 0, 0))
        cmap.set_bad((0, 0, 0, 0))
        plt.imshow(image, cmap=cmap, interpolation='none', vmin=RANGES[num][0], vmax=RANGES[num][1])
    plt.axis('off')
    plt.savefig(f'{sub_name}_c_{mode}.png', format = 'png', dpi=300, bbox_inches='tight', transparent=True)
    plt.clf()

    image = image_load[saggital, :, :]
    for num, color in enumerate(color_list):
        if color == BLACK_TRANSPARENT:
            continue
        cmap = ListedColormap([color])
        cmap.set_over((0, 0, 0, 0))
        cmap.set_under((0, 0, 0, 0))
        cmap.set_bad((0, 0, 0, 0))
        plt.imshow(image, cmap=cmap, interpolation='none', vmin=RANGES[num][0], vmax=RANGES[num][1])
    plt.axis('off')
    plt.savefig(f'{sub_name}_s_{mode}.png', format = 'png', dpi=300, bbox_inches='tight', transparent=True)
    plt.clf()

    images = [f'{sub_name}_a_{mode}.png', f'{sub_name}_c_{mode}.png', f'{sub_name}_s_{mode}.png']
    list_im = [Image.open(i) for i in images]
    list_im[1] = list_im[1].rotate(90, expand=True)
    list_im[1] = list_im[1].transpose(method=Image.FLIP_LEFT_RIGHT)
    list_im[2] = list_im[2].rotate(90, expand=True)
    #list_im = [i.resize((max_width_, max_height_)) for i in list_im]
    list_im[0] = list_im[0].resize((length, width), resample=Image.NEAREST)
    list_im[1] = list_im[1].resize((width, height), resample=Image.NEAREST)
    list_im[2] = list_im[2].resize((length, height), resample=Image.NEAREST)

    outline_im = []
    outline_im.append(outline2_no_T1(list_im[0], outline_list))
    outline_im.append(outline2_no_T1(list_im[1], outline_list))
    outline_im.append(outline2_no_T1(list_im[2], outline_list))
    list_im = [outline_black(i) for i in list_im]

    outline_im = np.vstack(outline_im)
    list_im = np.vstack(list_im)

    imgs_comb = Image.fromarray(list_im)
    imgs_comb_out = Image.fromarray(outline_im)

    # imgs_comb = imgs_comb.convert("RGBA")
    imgs_comb.save(f'{sub_name}_{mode}.png', format='PNG')
    imgs_comb_out.save(f'{sub_name}_{mode}_outline.png', format='PNG')
    for image in images:
        os.remove(image)