import nibabel as nib
from constants_v2 import *
from functions_v2 import *
from multiprocessing import Process

AXIAL_SLICENUM = 88 # overhead view
SAGGITAL_SLICENUM = 128 # side view
CORONAL_SLICENUM = 128 # back view
LENGTH = 256
WIDTH = 256
HEIGHT = 176
SUBLIST = dict()
# each subject should follow this template, if a filepath is not needed for a subject enter '' instead
# SUBLIST['subject_name'] = ['T1_filepath', 'charm_filepath', 'headreco_filepath', 'grace_filepath', 'domino_filepath']
# SUBLIST['missing_Grace'] = ['T1_filepath_1', 'charm_filepath_1', 'headreco_filepath_1', '', 'domino_filepath_1']


if __name__ == "__main__" :
    if len(SUBLIST) == 0:
        print('ERROR:')
        print("Make sure to add your subjects")
        quit()
    procs_outer = []
    for name, files in SUBLIST.items():
        if len(files) != 5:
            print(f'Please enter the right number of methods for subject {name}.')
            print('If a method is not needed, enter an empty string instead.')
            quit()
        try:
            if files[0] != '':
                T1_load = nib.load(files[0]).get_fdata()
            if files[1] != '':
                charm_load = nib.load(files[1]).get_fdata()
            if files[2] != '':
                headreco_load = nib.load(files[2]).get_fdata()
            if files[3] != '':
                grace_load = nib.load(files[3]).get_fdata()
            if files[4] != '':
                domino_load = nib.load(files[4]).get_fdata()
        except:
            print('ERROR:')
            print("One of these files was unable to be opened:")
            for j in range(len(files)):
                print(files[j], end=' ')
            quit()
        
        print('Running...')
        for j in range(4):
            if j == 0 and files[1] != '': # charm
                rgba_colors = [BLACK_TRANSPARENT, LIGHT_PINK, ORANGE, GRAY, PURPLE, BLACK_TRANSPARENT, DARK_BLUE, NEON_PINK, LIGHT_GREEN, LIGHT_BLUE, BLACK_TRANSPARENT, LIGHT_RED]
                outline_list = [(250, 128, 114, 255), (0, 255, 255, 255), (0, 255, 0, 255), (255, 0, 255, 255), (0, 0, 255, 255), (128, 0, 255, 255), (144, 144, 145, 255), (255, 128, 64, 255), (255, 128, 192, 255)]
                outline_list = outline_list[::-1]
                if files[0] == '':
                    proc = Process(target=generate_image_no_T1, args=(rgba_colors, outline_list, charm_load, AXIAL_SLICENUM, CORONAL_SLICENUM, SAGGITAL_SLICENUM, name, 'charm', int(LENGTH * 1.75), int(WIDTH * 1.75), int(HEIGHT * 1.75), ))
                    procs_outer.append(proc)
                    proc.start()
                    # generate_image_no_T1(rgba_colors, outline_list, charm_load, AXIAL_SLICENUM, CORONAL_SLICENUM, SAGGITAL_SLICENUM, name, 'charm', int(LENGTH * 1.75), int(WIDTH * 1.75), int(HEIGHT * 1.75))
                else:
                    proc = Process(target = generate_image, args=(rgba_colors, outline_list, T1_load, charm_load, AXIAL_SLICENUM, CORONAL_SLICENUM, SAGGITAL_SLICENUM, name, 'charm', int(LENGTH * 1.75), int(WIDTH * 1.75), int(HEIGHT * 1.75), ))
                    procs_outer.append(proc)
                    proc.start()
                    # generate_image(rgba_colors, outline_list, T1_load, charm_load, AXIAL_SLICENUM, CORONAL_SLICENUM, SAGGITAL_SLICENUM, name, 'charm', int(LENGTH * 1.75), int(WIDTH * 1.75), int(HEIGHT * 1.75))
            elif j == 1 and files[2] != '': # headreco
                rgba_colors = [BLACK_TRANSPARENT, LIGHT_PINK, ORANGE, GRAY, PURPLE, LIGHT_GREEN, LIGHT_BLUE]
                outline_list = [(0, 255, 255, 255), (22, 167, 11, 255), (128, 0, 255, 255), (144, 144, 145, 255), (255, 128, 64, 255), (255, 128, 192, 255)]
                outline_list = outline_list[::-1]
                if files[0] == '':
                    proc = Process(target=generate_image_no_T1, args=(rgba_colors, outline_list, headreco_load, AXIAL_SLICENUM, CORONAL_SLICENUM, SAGGITAL_SLICENUM, name, 'headreco', int(LENGTH * 1.75), int(WIDTH * 1.75), int(HEIGHT * 1.75), ))
                    procs_outer.append(proc)
                    proc.start()
                    # generate_image_no_T1(rgba_colors, outline_list, headreco_load, AXIAL_SLICENUM, CORONAL_SLICENUM, SAGGITAL_SLICENUM, name, 'headreco', int(LENGTH * 1.75), int(WIDTH * 1.75), int(HEIGHT * 1.75))
                else:
                    proc = Process(target=generate_image, args=(rgba_colors, outline_list, T1_load, headreco_load, AXIAL_SLICENUM, CORONAL_SLICENUM, SAGGITAL_SLICENUM, name, 'headreco', int(LENGTH * 1.75), int(WIDTH * 1.75), int(HEIGHT * 1.75), ))
                    procs_outer.append(proc)
                    proc.start()
                    # generate_image(rgba_colors, outline_list, T1_load, headreco_load, AXIAL_SLICENUM, CORONAL_SLICENUM, SAGGITAL_SLICENUM, name, 'headreco', int(LENGTH * 1.75), int(WIDTH * 1.75), int(HEIGHT * 1.75))
            elif j == 2 and files[3] != '': # grace
                rgba_colors = [BLACK_TRANSPARENT, LIGHT_PINK, ORANGE, GRAY, PURPLE, DARK_GREEN, DARK_BLUE, NEON_PINK, LIGHT_GREEN, LIGHT_BLUE, TAN, LIGHT_RED]
                outline_list = [(250, 128, 114, 255), (235, 218, 150, 255), (0, 255, 255, 255), (0, 255, 0, 255), (255, 0, 255, 255), (0, 0, 255, 255), (0, 129, 0, 255), (128, 0, 255, 255), (144, 144, 145, 255), (255, 128, 64, 255), (255, 128, 192, 255)]
                outline_list = outline_list[::-1]
                if files[0] == '':
                    proc = Process(target= generate_image_no_T1, args=(rgba_colors, outline_list, grace_load, AXIAL_SLICENUM, CORONAL_SLICENUM, SAGGITAL_SLICENUM, name, 'grace', int(LENGTH * 1.75), int(WIDTH * 1.75), int(HEIGHT * 1.75), ))
                    procs_outer.append(proc)
                    proc.start()
                    # generate_image_no_T1(rgba_colors, outline_list, grace_load, AXIAL_SLICENUM, CORONAL_SLICENUM, SAGGITAL_SLICENUM, name, 'grace', int(LENGTH * 1.75), int(WIDTH * 1.75), int(HEIGHT * 1.75))
                else:
                    proc = Process(target=generate_image, args=(rgba_colors, outline_list, T1_load, grace_load, AXIAL_SLICENUM, CORONAL_SLICENUM, SAGGITAL_SLICENUM, name, 'grace', int(LENGTH * 1.75), int(WIDTH * 1.75), int(HEIGHT * 1.75), ))
                    procs_outer.append(proc)
                    proc.start()
                    # generate_image(rgba_colors, outline_list, T1_load, grace_load, AXIAL_SLICENUM, CORONAL_SLICENUM, SAGGITAL_SLICENUM, name, 'grace', int(LENGTH * 1.75), int(WIDTH * 1.75), int(HEIGHT * 1.75))
            elif j == 3 and files[4] != '': # domino
                rgba_colors = [BLACK_TRANSPARENT, LIGHT_PINK, ORANGE, GRAY, PURPLE, DARK_GREEN, DARK_BLUE, NEON_PINK, LIGHT_GREEN, LIGHT_BLUE, TAN, LIGHT_RED]
                outline_list = [(250, 128, 114, 255), (235, 218, 150, 255), (0, 255, 255, 255), (0, 255, 0, 255), (255, 0, 255, 255), (0, 0, 255, 255), (0, 129, 0, 255), (128, 0, 255, 255), (144, 144, 145, 255), (255, 128, 64, 255), (255, 128, 192, 255)]
                outline_list = outline_list[::-1]
                if files[0] == '':
                    proc = Process(target=generate_image_no_T1, args=(rgba_colors, outline_list, domino_load, AXIAL_SLICENUM, CORONAL_SLICENUM, SAGGITAL_SLICENUM, name, 'domino', int(LENGTH * 1.75), int(WIDTH * 1.75), int(HEIGHT * 1.75), ))
                    procs_outer.append(proc)
                    proc.start()
                    # generate_image_no_T1(rgba_colors, outline_list, domino_load, AXIAL_SLICENUM, CORONAL_SLICENUM, SAGGITAL_SLICENUM, name, 'domino', int(LENGTH * 1.75), int(WIDTH * 1.75), int(HEIGHT * 1.75))
                else:
                    proc = Process(target=generate_image, args=(rgba_colors, outline_list, T1_load, domino_load, AXIAL_SLICENUM, CORONAL_SLICENUM, SAGGITAL_SLICENUM, name, 'domino', int(LENGTH * 1.75), int(WIDTH * 1.75), int(HEIGHT * 1.75), ))
                    procs_outer.append(proc)
                    proc.start()
                    # generate_image(rgba_colors, outline_list, T1_load, domino_load, AXIAL_SLICENUM, CORONAL_SLICENUM, SAGGITAL_SLICENUM, name, 'domino', int(LENGTH * 1.75), int(WIDTH * 1.75), int(HEIGHT * 1.75))

    for proc in procs_outer:
        proc.join()
    
    procs_outer.clear()
    print('Tasks finished.')