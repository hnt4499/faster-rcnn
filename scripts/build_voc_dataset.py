import os
import xml.etree.ElementTree as ET
import sys
import argparse
from PIL import Image

import pandas as pd
from tqdm import tqdm


DESCRIPTION = """
Build a csv file containing necessary information of a PASCAL VOC dataset that
is compatible with this package.
"""


def read_xml(img_id, root_dir):
    # Get paths
    xml_path = os.path.join(root_dir, "Annotations", f"{img_id}.xml")
    xml_path = os.path.abspath(xml_path)
    img_path = os.path.join(root_dir, "JPEGImages", f"{img_id}.jpg")
    img_path = os.path.abspath(img_path)

    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')

    # Get image info
    if size is not None:
        width = int(size.find('width').text)
        height = int(size.find('height').text)
    else:
        img = Image.open(img_path)
        width, height = img.size

    # Get objects' info
    bboxes = []
    labels = []
    difficults = []

    for obj in root.findall('object'):
        label = obj.find('name').text
        assert "," not in label

        difficult = obj.find('difficult')
        if difficult is not None:
            difficult = difficult.text

        bnd_box = obj.find('bndbox')
        bbox = [
            int(float(bnd_box.find('xmin').text)),
            int(float(bnd_box.find('ymin').text)),
            int(float(bnd_box.find('xmax').text)),
            int(float(bnd_box.find('ymax').text))
        ]

        # Add infos
        bboxes.extend(bbox)
        labels.append(label)
        difficults.append(difficult)

    bboxes = ",".join(map(str, bboxes))
    labels = ",".join(labels)

    info = {
        "ann_path": xml_path,
        "img_path": img_path,
        "width": width,
        "height": height,
        "bboxes": bboxes,
        "labels": labels,
    }

    difficults_u = list(set(difficults))
    if not (len(difficults_u) == 1 and difficults_u[0] is None):
        difficults = ",".join(difficults)
        info["difficults"] = difficults

    return info


def main(args):
    assert len(args.src_dir) == len(args.save_path)

    for src_dir, save_path in zip(args.src_dir, args.save_path):
        root_dir = os.path.abspath(src_dir)
        ann_dir = os.path.join(root_dir, "Annotations")

        img_infos = []
        for filename in tqdm(list(os.listdir(ann_dir))):
            img_id, _ = os.path.splitext(filename)
            img_info = read_xml(img_id=img_id, root_dir=root_dir)
            img_infos.append(img_info)

        df = pd.DataFrame.from_records(img_infos)
        df.to_csv(save_path, index=False)

    print("Done.")


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=DESCRIPTION)
    parser.add_argument(
        '-d', '--src-dir', type=str, required=True, nargs="+",
        help='Path(s) to the root directory of the dataset (which usually '
             'contain(s) folders like "Annotations", "JPEGImages"). Multiple '
             'paths are to be separated by space.')
    parser.add_argument(
        '-s', '--save-path', type=str, required=True, nargs="+",
        help='Path(s) to save the dataset information. Multiple paths are to '
             'be separated by space.')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
