import os
import json
import sys
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm


DESCRIPTION = """
Build a csv file containing necessary information of a COCO dataset that is
compatible with this package.
"""


def process_df(df_images, df_objects):
    if df_objects is None:
        df_merge = df_images[["id", "file_name", "width", "height"]]
        df_merge = df_merge.set_index("id")
    else:
        # Merge
        df = pd.merge(df_objects, df_images, left_on="image_id", right_on="id")
        df = df[["image_id", "bbox", "category_id",
                 "file_name", "height", "width"]]

        # Convert bboxes to integers
        df["bbox"] = df["bbox"].apply(lambda x: list(map(round, x)))

        # Merge all objects within each image
        def transform(sub_df):
            image_id, file_name, height, width = sub_df.iloc[0][
                ["image_id", "file_name", "height", "width"]]

            category_ids = sub_df["category_id"].tolist()
            category_ids = ",".join(map(str, category_ids))

            bboxes = sub_df["bbox"].tolist()
            bboxes = sum(bboxes, [])
            bboxes = ",".join(map(str, bboxes))

            return pd.Series({
                "image_id": image_id, "img_name": file_name, "width": width,
                "height": height, "bboxes": bboxes, "labels": category_ids
            })

        df_merge = df.groupby("image_id").apply(transform)
        assert len(df_merge) == df_objects["image_id"].nunique()

    return df_merge


def main(args):
    # Read annotation file
    print("Reading annotation file...")
    with open(args.ann_path) as fin:
        ann = json.load(fin)
    print(f"Number of images: {len(ann['images'])}, number of annotations: "
          f"{len(ann['annotations']) if 'annotations' in ann else -1}")

    # Convert to dataframes
    df_images = pd.DataFrame.from_records(ann["images"])
    if "annotations" in ann:
        df_objects = pd.DataFrame.from_records(ann["annotations"])
        assert df_objects["image_id"].isin(df_images["id"]).all()
    else:
        df_objects = None

    # Process dataframes
    print("Processing dataframes...")
    df = process_df(df_images, df_objects)

    # Parse images
    print("Parsing images...")
    ids = []
    file_paths = []
    no_info_ids = []
    paths = list(Path(args.image_dir).glob("*.jpg"))

    for file_path in tqdm(paths):
        _, file_name = os.path.split(file_path)
        if not file_name.startswith("COCO"):
            continue

        name, _ = os.path.splitext(file_name)
        id = int(name.split("_")[-1])
        if id not in df.index:
            no_info_ids.append(id)
        else:
            ids.append(id)
            file_paths.append(file_path)

    assert len(ids) == len(df)  # make sure all images in `df` are found
    df = df.loc[ids]
    df["img_path"] = file_paths

    if df_objects is None:
        df = df[["img_path", "width", "height"]]
    else:
        df = df[["img_path", "width", "height", "bboxes", "labels"]]
    df.to_csv(args.save_path, index=False)

    print(f"There are {len(no_info_ids)} images that have no "
          f"information: {no_info_ids}")
    print("Done.")


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=DESCRIPTION)
    parser.add_argument(
        '-d', '--image-dir', type=str, required=True,
        help='Path(s) to the image directory.')
    parser.add_argument(
        '-a', '--ann-path', type=str, required=True,
        help='Path to the annotation file (e.g., instances_train2014.json).')
    parser.add_argument(
        '-s', '--save-path', type=str, required=True,
        help='Path(s) to save the dataset information.')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
