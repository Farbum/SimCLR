import glob
import pandas as pd
import os
import shutil
import logging

logger = logging.getLogger(__name__)
logger.setLevel('INFO')

# TODO: add resizing together  with  folderizing

def folderize(data_path, labels_path, destination_path):
    """
    Args:
        data_path (str): folder where images are stored
        labels_path(str): where is the csv for labels
        destination_path(str): where folders for each class should be created

    Returns:
        classes (list str): list of classes in dataset

    """
    df = pd.read_csv(labels_path)
    df['nice_label'] = df.drop('image', axis=1).idxmax(axis=1)

    def mycopy(row):
        dest_fpath = os.path.join(destination_path, row['nice_label'])
        os.makedirs(dest_fpath, exist_ok=True)
        try:
            shutil.copy2(os.path.join(data_path, row['image'])+'.jpg', dest_fpath)
        except FileNotFoundError as e:
            logging.error(f"Did not find {row['image']} in source files.")
            logging.error(e)

    df.apply(mycopy, axis=1)

    classes = [x.split('/')[-1] for x in glob.glob(os.path.join(destination_path, '*'))]
    return classes


if __name__ == '__main__':
    print(folderize('data/images', 'data/labels.csv', destination_path='temp'))
