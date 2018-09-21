import os.path

import shutil
from PIL import Image
from lxml import etree as ET


def gen_template():
    root = ET.Element('annotation', verified='yes')

    folder = ET.SubElement(root, 'folder')
    folder.text = 'UECFOOD100_JS/1'

    filename = ET.SubElement(root, 'filename')
    filename.text = '1.jpg'

    path = ET.SubElement(root, 'path')
    path.text = '/Volumes/JS/UECFOOD100_JS/1/1.jpg'

    size = ET.SubElement(root, 'size')
    width = ET.SubElement(size, 'width')
    width.text = '800'
    height = ET.SubElement(size, 'width')
    height.text = '600'
    depth = ET.SubElement(size, 'width')
    depth.text = '3'

    object = ET.SubElement(root, 'object')
    name1 = ET.SubElement(object, 'name')
    name1.text = 'rice'

    bndbox1 = ET.SubElement(object, 'bndbox')
    xmin1 = ET.SubElement(bndbox1, 'xmin')
    xmin1.text = '0'
    ymin1 = ET.SubElement(bndbox1, 'ymin')
    ymin1.text = '143'
    xmax1 = ET.SubElement(bndbox1, 'xmax')
    xmax1.text = '370'
    ymax1 = ET.SubElement(bndbox1, 'ymax')
    ymax1.text = '486'

    name = ET.SubElement(object, 'name')
    name.text = 'rice'

    bndbox = ET.SubElement(object, 'bndbox')
    xmin = ET.SubElement(bndbox, 'xmin')
    xmin.text = '0'
    ymin = ET.SubElement(bndbox, 'ymin')
    ymin.text = '143'
    xmax = ET.SubElement(bndbox, 'xmax')
    xmax.text = '370'
    ymax = ET.SubElement(bndbox, 'ymax')
    ymax.text = '486'

    tree = ET.ElementTree(root)
    tree.write('template.xml', pretty_print=True)


def write_xml(line, i_folder, category):
    if not os.path.exists('/Volumes/JS/UECFOOD100_448/' + str(i_folder) + '/annotations_new/'):
        os.makedirs('/Volumes/JS/UECFOOD100_448/' + str(i_folder) + '/annotations_new')

    anno_path = '/Volumes/JS/UECFOOD100_448/' + str(i_folder) + '/annotations_new/' + line[0] + '.xml'

    if not os.path.isfile(anno_path):
        root = ET.Element('annotation', verified='yes')

        folder = ET.SubElement(root, 'folder')
        folder.text = '/Volumes/JS/UECFOOD100_448/' + str(i_folder)

        filename = ET.SubElement(root, 'filename')
        filename.text = line[0] + '.jpg'

        path = ET.SubElement(root, 'path')
        path.text = '/Volumes/JS/UECFOOD100_448/' + str(i_folder) + '/' + line[0] + '.jpg'
        assert Image.open(path.text).size == (800, 600)

        size = ET.SubElement(root, 'size')
        width = ET.SubElement(size, 'width')
        width.text = str(Image.open(path.text).size[0])
        height = ET.SubElement(size, 'height')
        height.text = str(Image.open(path.text).size[1])
        depth = ET.SubElement(size, 'depth')
        depth.text = '3'

        object = ET.SubElement(root, 'object')
        name = ET.SubElement(object, 'name')
        name.text = category[i_folder - 1]

        bndbox = ET.SubElement(object, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = line[1]
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = line[2]

        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = line[3]
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = line[4]

        tree = ET.ElementTree(root)
        tree.write(anno_path, pretty_print=True, encoding='utf-8')
    else:
        pass
        # print('folder ' + str(i_folder) + ', img ' + line[0] + ' has another object.')
        # tree = ET.parse(anno_path)
        # root = tree.getroot()
        # object_tag = [root.getchildren()[x].tag == 'object' for x in range(0, len(root.getchildren()))]
        # index = [i for i, x in enumerate(object_tag) if x][0]
        # object = root.getchildren()[index]
        #
        # name = ET.SubElement(object, 'name')
        # name.text = category[i_folder - 1]
        #
        # bndbox = ET.SubElement(object, 'bndbox')
        # xmin = ET.SubElement(bndbox, 'xmin')
        # xmin.text = line[1]
        # ymin = ET.SubElement(bndbox, 'ymin')
        # ymin.text = line[2]
        # xmax = ET.SubElement(bndbox, 'xmax')
        # xmax.text = line[3]
        # ymax = ET.SubElement(bndbox, 'ymax')
        # ymax.text = line[4]
        # pass

        # tree.write(anno_path, pretty_print=True, encoding='utf-8')  # TODO: organize the rewrite .xml file format


def read_category():
    category = []
    with open('/Volumes/JS/UECFOOD100_448/category.txt', 'r') as file:
        for i, line in enumerate(file):
            if i > 0:
                line = line.rstrip('\n')
                line = line.split('\t')
                category.append(line[1])
    return category


def rm_anno_dir():
    print('removing wrongly created annotation folder under each category...')
    for i in range(1, 101):
        if os.path.exists('/Volumes/JS/UECFOOD100_448/' + str(i) + '/annotations_new'):
            shutil.rmtree('/Volumes/JS/UECFOOD100_448/' + str(i) + '/annotations_new')
    print('Done!')


def gen_xmls():
    num_classes = 100
    category = read_category()
    new_bb_info = 'new_bb_info.txt'
    for i_folder in range(1, num_classes + 1):
        imagepath = '/Volumes/JS/UECFOOD100_448/' + str(i_folder)
        with open(imagepath + '/' + new_bb_info, 'r') as bbox_file:
            num_items = sum(1 for line in open(imagepath + '/' + new_bb_info)) - 1
            print(num_items)
            for i, line in enumerate(bbox_file):
                if i > 0:
                    line = line.rstrip('\n')
                    line = line.split(' ')
                    write_xml(line, i_folder, category)
    print('Done!')


# rm_anno_dir()
gen_xmls()
