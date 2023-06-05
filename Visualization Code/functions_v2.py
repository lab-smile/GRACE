import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
import numpy as np
import os
from constants_v2 import *
from multiprocessing import Process, Queue


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
                    image.putpixel((x, y), (0, 0, 0, 256))

    curr_pixel = image.getpixel((0, 0))
    for y in range(image.height):
        for x in range(image.width):
            if image.getpixel((x, y)) == curr_pixel:
                continue
            else:
                if image.getpixel((x, y)) in RGB_VALUES:
                    curr_pixel = image.getpixel((x, y))
                    image.putpixel((x, y), (0, 0, 0, 256))
    
    return image

def outline2(image, color_list, sub_name, mode, queue): # outlines with color
    image = image.convert("RGBA")
    if mode == 'axial':
        T1_new = Image.open(f't1a_{sub_name}.png')
        T1_new = T1_new.convert("RGBA")
    elif mode == 'coronal':
        T1_new = Image.open(f't1c_{sub_name}.png')
        T1_new = T1_new.convert("RGBA")
    elif mode == 'saggital':
        T1_new = Image.open(f't1s_{sub_name}.png')
        T1_new = T1_new.convert("RGBA")
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
    queue.put(Image.alpha_composite(T1_new, outlined_image))

def outline2_no_T1(image, color_list, queue):
    image = image.convert("RGBA")
    outlined_image = Image.new("RGBA", image.size, (0, 0, 0, 256))
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
    queue.put(outlined_image)

def max_height(o_width, o_height, new_height):
    return int(o_width / o_height * new_height)

def max_width(o_width, o_height, new_width):
    return int(o_width / o_height * new_width)

def make_view(T1, view_image, color_list, sub_name, mode, x, y, view_name):
    plt.imshow(T1, cmap='gray', interpolation='none')
    plt.axis('off')
    plt.savefig(f't1{view_name}_{sub_name}.png', format = 'png', dpi=300, bbox_inches='tight', transparent=True )
    for num, color in enumerate(color_list):
        if color == BLACK_TRANSPARENT:
            continue
        cmap = ListedColormap([color])
        cmap.set_over((0, 0, 0, 0))
        cmap.set_under((0, 0, 0, 0))
        cmap.set_bad((0, 0, 0, 0))
        plt.imshow(view_image, cmap=cmap, interpolation='none', vmin=RANGES[num][0], vmax=RANGES[num][1])
    plt.axis('off')
    plt.savefig(f'{sub_name}_{view_name}_{mode}.png', format = 'png', dpi=300, bbox_inches='tight', transparent=True)
    mask = Image.open(f'{sub_name}_{view_name}_{mode}.png')
    T1_new = Image.open(f't1{view_name}_{sub_name}.png')
    if view_name == 'c':
        mask = mask.rotate(90, expand=True)
        mask = mask.transpose(method=Image.FLIP_LEFT_RIGHT)
        mask = mask.resize((x, y), resample=Image.NEAREST)
        T1_new = T1_new.rotate(90, expand=True)
        T1_new = T1_new.transpose(method=Image.FLIP_LEFT_RIGHT)
        T1_new = T1_new.resize((x, y), resample=Image.NEAREST)
        T1_new.save(f't1c_{sub_name}.png')
    elif view_name == 's':
        mask = mask.rotate(90, expand=True)
        T1_new = T1_new.rotate(90, expand=True)
        mask = mask.resize((x, y), resample=Image.NEAREST)
        T1_new = T1_new.resize((x, y), resample=Image.NEAREST)
        T1_new.save(f't1s_{sub_name}.png')
    else:
        mask = mask.resize((x, y), resample=Image.NEAREST)
        T1_new = T1_new.resize((x, y), resample=Image.NEAREST)
        T1_new.save(f't1a_{sub_name}.png')
    mask = Image.alpha_composite(T1_new.convert('RGBA'), mask.convert('RGBA'))
    mask.save(f'{sub_name}_{view_name}_{mode}.png', format='PNG')

def create_final_img(list_im, sub_name, mode):
    imgs_comb = Image.fromarray(list_im)

    imgs_comb = imgs_comb.convert("RGBA")
    imgs_comb.save(f'{sub_name}_{mode}.png', format='PNG')

def create_final_img_outline(outline_im, sub_name, mode):
    imgs_comb_out = Image.fromarray(outline_im)
    imgs_comb_out.save(f'{sub_name}_{mode}_outline.png', format='PNG')

def generate_image(color_list, outline_list, T1_load, image_load, axial, coronal, saggital, sub_name, mode, length, width, height):
    # creating individual images
    T1_axial = T1_load[:, :, axial]
    image_axial = image_load[:, :, axial]
    T1_coronal = T1_load[:, coronal, :]
    image_coronal = image_load[:, coronal, :]
    T1_saggital = T1_load[saggital, :, :]
    image_saggital = image_load[saggital, :, :]

    procs = []
    proc = Process(target=make_view, args=(T1_axial, image_axial, color_list, sub_name, mode, length, width, 'a',))
    procs.append(proc)
    proc.start()
    proc = Process(target=make_view, args=(T1_coronal, image_coronal, color_list, sub_name, mode, width, height, 'c',))
    procs.append(proc)
    proc.start()
    proc = Process(target=make_view, args=(T1_saggital, image_saggital, color_list, sub_name, mode, length, height, 's',))
    procs.append(proc)
    proc.start()

    for proc in procs:
        proc.join()

    procs.clear()

    # creating outline-only images
    images = [f'{sub_name}_a_{mode}.png', f'{sub_name}_c_{mode}.png', f'{sub_name}_s_{mode}.png']
    list_im = [Image.open(i) for i in images]

    outline_im = []
    queue = Queue()
    proc = Process(target=outline2, args=(list_im[0], outline_list, sub_name, 'axial', queue, ))
    procs.append(proc)
    proc.start()
    proc = Process(target=outline2, args=(list_im[1], outline_list, sub_name, 'coronal', queue, ))
    procs.append(proc)
    proc.start()
    proc = Process(target=outline2, args=(list_im[2], outline_list, sub_name, 'saggital', queue, ))
    procs.append(proc)
    proc.start()
    for proc in procs:
        ret = queue.get()
        outline_im.append(ret)
    for proc in procs:
        proc.join()

    procs.clear()

    # combining all images
    list_im = [outline_black(i) for i in list_im]

    outline_im = np.vstack(outline_im)
    list_im = np.vstack(list_im)
  
    proc = Process(target=create_final_img, args=(list_im, sub_name, mode, ))
    procs.append(proc)
    proc.start()
    proc = Process(target=create_final_img_outline, args=(outline_im, sub_name, mode, ))
    procs.append(proc)
    proc.start()

    for proc in procs:
        proc.join()
    
    procs.clear()

    # removing temporary images
    for image in images:
        os.remove(image)
    os.remove(f't1a_{sub_name}.png')
    os.remove(f't1c_{sub_name}.png')
    os.remove(f't1s_{sub_name}.png')

    print(f'{sub_name} finished, with T1.')



def make_view_no_T1(color_list, image, sub_name, mode, x, y, view_name):
    for num, color in enumerate(color_list):
        if color == BLACK_TRANSPARENT:
            cmap = ListedColormap([BLACK])
        else:
            cmap = ListedColormap([color])
            cmap.set_over((0, 0, 0, 0))
            cmap.set_under((0, 0, 0, 0))
            cmap.set_bad((0, 0, 0, 0))
        plt.imshow(image, cmap=cmap, interpolation='none', vmin=RANGES[num][0], vmax=RANGES[num][1])
    plt.axis('off')
    plt.savefig(f'{sub_name}_{view_name}_{mode}.png', format = 'png', dpi=300, bbox_inches='tight',transparent=True)
    plt.clf()
    image_new = Image.open(f'{sub_name}_{view_name}_{mode}.png')
    if view_name == 'a':
        image_new = image_new.resize((x, y), resample=Image.NEAREST)
    elif view_name == 'c':
        image_new = image_new.rotate(90, expand=True).transpose(method=Image.FLIP_LEFT_RIGHT).resize((x, y), resample=Image.NEAREST)
    elif view_name == 's':
        image_new = image_new.rotate(90, expand=True).resize((x, y), resample=Image.NEAREST)
    image_new.save(f'{sub_name}_{view_name}_{mode}.png')

def create_final_img_no_T1(list_im, sub_name, mode):
    imgs_comb = Image.fromarray(list_im)

    imgs_comb = imgs_comb.convert("RGBA")
    imgs_comb.save(f'{sub_name}_{mode}_no_T1.png', format='PNG')

def create_final_img_outline_no_T1(outline_im, sub_name, mode):
    imgs_comb_out = Image.fromarray(outline_im)

    imgs_comb_out.save(f'{sub_name}_{mode}_outline_no_T1.png', format='PNG')

def generate_image_no_T1(color_list, outline_list, image_load, axial, coronal, saggital, sub_name, mode, length, width, height):
    #creating individual images
    image_axial = image_load[:, :, axial]
    image_coronal = image_load[:, coronal, :]
    image_saggital = image_load[saggital, :, :]

    procs = []
    proc = Process(target=make_view_no_T1, args=(color_list, image_axial, sub_name, mode, length, width, 'a'))
    procs.append(proc)
    proc.start()
    proc = Process(target=make_view_no_T1, args=(color_list, image_coronal, sub_name, mode, width, height, 'c'))
    procs.append(proc)
    proc.start()
    proc = Process(target=make_view_no_T1, args=(color_list, image_saggital, sub_name, mode, length, height, 's'))
    procs.append(proc)
    proc.start()

    for proc in procs:
        proc.join()

    procs.clear()

    #creating outlines
    images = [f'{sub_name}_a_{mode}.png', f'{sub_name}_c_{mode}.png', f'{sub_name}_s_{mode}.png']
    list_im = [Image.open(i) for i in images]

    outline_im = []
    queue = Queue()
    proc = Process(target=outline2_no_T1, args=(list_im[0], outline_list, queue, ))
    procs.append(proc)
    proc.start()
    proc = Process(target=outline2_no_T1, args=(list_im[1], outline_list, queue, ))
    procs.append(proc)
    proc.start()
    proc = Process(target=outline2_no_T1, args=(list_im[2], outline_list, queue, ))
    procs.append(proc)
    proc.start()
    for proc in procs:
        ret = queue.get()
        outline_im.append(ret)
    for proc in procs:
        proc.join()
    
    procs.clear()

    #combining images
    list_im = [outline_black(i) for i in list_im]

    outline_im = np.vstack(outline_im)
    list_im = np.vstack(list_im)

    proc = Process(target=create_final_img_no_T1, args=(list_im, sub_name, mode, ))
    procs.append(proc)
    proc.start()
    proc = Process(target=create_final_img_outline_no_T1, args=(outline_im, sub_name, mode, ))
    procs.append(proc)
    proc.start()

    for proc in procs:
        proc.join()

    procs.clear()

    #deleting temporary images
    for image in images:
        os.remove(image)

    print(f'{sub_name} finished, no T1.')