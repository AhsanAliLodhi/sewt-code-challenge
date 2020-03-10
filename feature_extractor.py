
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from dataset import TowelsDataset
from custom_transforms import Rescale, ToTensor
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import argparse


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(model_class, freeze=True, is_cuda=True):
    model = model_class(pretrained=True)
    modules = list(model.children())[:-1]
    model = nn.Sequential(*modules)
    for p in model.parameters():
        p.requires_grad = not freeze
    if is_cuda:
        model = model.cuda()
    return model


def extract_features(model_class, data_loader, image_col,
                     batch_vars, device='cuda', top_n=None):
    data = pd.DataFrame(columns=batch_vars + ["features"])
    for i, batch in tqdm(enumerate(data_loader),
                         total=len(data_loader)):
        samples = {var: batch[var] for idx, var in enumerate(batch_vars)}
        model = load_model(model_class)
        samples['image'] = batch["image"].to(device)
        samples['image'] = samples['image'].float()
        features = model(samples['image'])
        features = torch.flatten(features, start_dim=1)
        samples["features"] = features
        samples["features"] = (samples["features"].data).cpu().numpy()
        for idx in range(data_loader.batch_size):
            try:
                sample = {var: samples[var][idx]
                          for var in batch_vars + ["features"]}
                data = data.append(sample, ignore_index=True)
            except IndexError:
                break
        if top_n is not None:
            if top_n == 0:
                break
            else:
                top_n -= 1
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
    Extract features from the towel dataset by forward passing
    (transfer learning) all the images form dataset through Resnet152
    up until it's 2nd last layer. This converts each image of 254 * 254 * 3 to
    vector of size 1 * 2048''')
    parser.add_argument('-i', '--img_path', type=str,
                        help='directory where the images reside.',
                        default="data/train")
    parser.add_argument('-b', '--bbox_path', type=str,
                        help='path for bbox csv.',
                        default="data/train/BB_labels.txt")
    parser.add_argument('-o', '--out_pkl_file', type=str,
                        help='''path to store the extracted features as pickle
                                file. Can be read with pandas.read_pickle
                                method.''',
                        default="data/train/features.pkl")
    args = parser.parse_args()
    towel_dataset = \
                    TowelsDataset(csv_file=args.bbox_path,
                                  root_dir=args.img_path,
                                  transform=transforms.Compose([
                                        # These numbers come from
                                        # https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
                                        # as the default normalizing constants
                                        # for images if they were to be input
                                        # in to resnet or inception
                                        transforms.Normalize([0.485, 0.456,
                                                              0.406],
                                                             [0.229, 0.224,
                                                              0.225])
                                    ]),
                                  custom_transform=transforms.Compose([
                                        # 299 because that is the default input
                                        # image size for resnet152
                                        Rescale(299),
                                        ToTensor()
                                    ])
                                  )
    data_loader = torch.utils.data.DataLoader(towel_dataset, batch_size=32)
    data = extract_features(models.resnet152, data_loader, "image",
                            ["image_name", "bbox"],
                            device=device)
    data.to_pickle(args.out_pkl_file)
