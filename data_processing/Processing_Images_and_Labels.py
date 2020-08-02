from skimage import io
import numpy as np
import cv2 as cv
import os
import json

name_cities = [
    'berlin',
    'bielefeld',
    'bonn',
    'leverkusen',
    'mainz',
    'munich'
]


for NAME_CITY in name_cities:
    print('\n\nCidade: ',NAME_CITY)
    IMG_PATH = '/home/matheus/Documentos/Road_Detection_Dataset/leftImg8bit_trainvaltest/leftImg8bit/test/' + NAME_CITY

    img_collection = io.ImageCollection(IMG_PATH+'/*.png:'+IMG_PATH+'/*.jpg')

    #Porcentagem escalonada
    scale_percent = 0.125
    #Altura e largura que será escalonado
    initial_dim = (img_collection[0].shape[1], img_collection[0].shape[0])
    width = int(initial_dim[0] * scale_percent)
    height = int(initial_dim[1] * scale_percent)
    #Tuple com as dimensoes
    dim = (width, height)

    DATA_SAVE = '/home/matheus/Documentos/E-Net_Road-Detection/CityScapes_RoadDetection/' #Pasta que salvarei

    os.chdir(DATA_SAVE)
    if not os.path.isdir(NAME_CITY):
        os.mkdir(NAME_CITY)
        os.chdir('./'+NAME_CITY)
        os.mkdir('real_image')
        os.chdir('./real_image')
        os.mkdir('img')
        os.chdir('./img')
    else:
        os.chdir('./'+NAME_CITY+'/real_image/img')

    for i, image in enumerate(img_collection):
        resized = cv.resize(image, dim, interpolation = cv.INTER_AREA)
        name_files = img_collection.files[i].split('/')
        cv.imwrite(name_files[len(name_files)-1],resized)
        print('Imagem: '+name_files[len(name_files)-1]+' {}/{} - {:.0f}%'.format(i+1,len(img_collection),(i+1)/len(img_collection)*100) )

    JSON_PATH = '/home/matheus/Documentos/Road_Detection_Dataset/gtFine_trainvaltest/gtFine/test/' + NAME_CITY
    json_list = []
    i = 0
    for file in os.listdir(JSON_PATH):
        if file.endswith(".json"):
            json_list.append(os.path.join(JSON_PATH, file))

    json_list = sorted(json_list)

    os.chdir(DATA_SAVE+'/'+NAME_CITY)
    if not os.path.isdir('mask_image'):
        os.mkdir('mask_image')
        os.chdir('./mask_image')
        os.mkdir('img')
        os.chdir('./img')
    else:
        os.chdir('./mask_image/img')

    for i in range(len(json_list)):
        mask_gen = np.zeros((initial_dim[1],initial_dim[0]), np.uint8)
        with open(json_list[i]) as json_file:
            json_file = json.load(json_file)
            for dict_elem in json_file['objects']:
                #Extraindo polígonos
                if dict_elem['label'] == 'road':
                    pts = np.array(dict_elem['polygon'],np.int32)
                    cv.fillPoly(mask_gen,[pts],255)
                else:
                    pts = np.array(dict_elem['polygon'],np.int32)
                    cv.fillPoly(mask_gen,[pts],0)
        resized_mask = cv.resize(mask_gen, dim, interpolation = cv.INTER_AREA)
        name_files = json_list[i].split('/')
        cv.imwrite(name_files[len(name_files)-1][:-5]+'.png',resized_mask)
        print('Imagem: '+name_files[len(name_files)-1][:-5]+'.png {}/{} - {:.0f}%'.format(i+1,len(json_list),(i+1)/len(json_list)*100) )