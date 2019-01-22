from lxml import etree as ET


def read_category():
    category = []
    with open('/Volumes/JS/UECFOOD100_JS/category.txt', 'r') as file:
        for i, line in enumerate(file):
            if i > 0:
                line = line.rstrip('\n')
                line = line.split('\t')
                category.append(line[1])
    return category


def write_multifood_xml(img_name, bb_info2xml, category):
    print('rewrite ' + img_name + ' in folder ' + ' and '.join([i[0] for i in bb_info2xml]))
    for i in range(0, len(bb_info2xml)):
        root = ET.Element('annotation', verified='yes')

        folder = ET.SubElement(root, 'folder')
        folder.text = 'UECFOOD256/UECFOOD256/' + bb_info2xml[i][0]

        filename = ET.SubElement(root, 'filename')
        filename.text = img_name

        path = ET.SubElement(root, 'path')
        path.text = '/Volumes/JS/UECFOOD256/UECFOOD256/' + bb_info2xml[i][0] + '/' + img_name

        size = ET.SubElement(root, 'size')
        width = ET.SubElement(size, 'width')
        width.text = '800'
        height = ET.SubElement(size, 'height')
        height.text = '600'
        depth = ET.SubElement(size, 'depth')
        depth.text = '3'

        # object = ET.SubElement(root, 'object')

        for j in range(0, len(bb_info2xml)):
            object = ET.SubElement(root, 'object')
            name = ET.SubElement(object, 'name')
            name.text = category[int(bb_info2xml[j][0]) - 1]

            bndbox = ET.SubElement(object, 'bndbox')
            xmin = ET.SubElement(bndbox, 'xmin')
            xmin.text = bb_info2xml[j][1]
            ymin = ET.SubElement(bndbox, 'ymin')
            ymin.text = bb_info2xml[j][2]
            xmax = ET.SubElement(bndbox, 'xmax')
            xmax.text = bb_info2xml[j][3]
            ymax = ET.SubElement(bndbox, 'ymax')
            ymax.text = bb_info2xml[j][4]

        tree = ET.ElementTree(root)
        tree.write('/Volumes/JS/UECFOOD256/UECFOOD256/' + bb_info2xml[i][0] + '/' + 'annotations_new/' + img_name[:-4] + '.xml', pretty_print=True)


if __name__ == '__main__':
    print('Start here')
    category = read_category()
    with open('/Volumes/JS/UECFOOD100_JS/multiple_food.txt', 'r') as file:
        for i, line in enumerate(file):
            if i > 0:
                line = line.rstrip('\n')
                line = line.split(' ')[:-1]
                print(line)
                bb_info2xml = []
                for j in range(1, len(line)):
                    with open('/Volumes/JS/UECFOOD256/UECFOOD256/' + line[j] + '/new_bb_info.txt', 'r') as bb_file:
                        for l in bb_file.readlines():
                            if l.startswith(line[0]):
                                l = l.rstrip('\n').split(' ')
                                l[0] = line[j]
                                bb_info2xml.append(l)
                                break
                for j in range(1, len(line)):
                    img_name = line[0] + '.jpg'
                    write_multifood_xml(img_name, bb_info2xml, category)

    print('Done!')
