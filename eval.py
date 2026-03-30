import argparse
import glob
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")
import random
import sys

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset
from torchvision import datasets, transforms
from PIL import Image
import pickle
import torch.nn.functional as F


def filter_image_paths(image_paths):
    new_image_paths = []
    new_images = []
    for p in image_paths:
        img = np.array(Image.open(p).convert('RGB'))
        if img is None:
            continue
        new_image_paths.append(p)
        new_images.append(img)
    return new_image_paths, new_images

def count_subdirectories(directory):
    return len([name for name in os.scandir(directory) if name.is_dir()])

def get_features(model, dataset, sort=False):
    if sort:
        loader_users = DataLoader(dataset, num_workers=4, batch_size=32, shuffle=False)
    else:
        loader_users = DataLoader(dataset, num_workers=4, batch_size=32, shuffle=True)
    
    with torch.no_grad():
        all_classes = [] 
        embeddings_users = []
        for data_user, class_user in iter(loader_users):
            embeddings = model(data_user.to(device)).detach().cpu().numpy()
            embeddings_users.append(embeddings)
            all_classes.extend(class_user.cpu().numpy())

    return np.concatenate(embeddings_users), np.asarray(all_classes)

def get_dataset_features(model, dataset_name, model_name, preprocess, dataset_path, train_ratio=0.7):
    features_file = f'{dataset_name}_{model_name}.pkl'
    if os.path.exists(features_file):
        with open(features_file, 'rb') as f:
            features_train, features_test, classes_train, classes_test, class2idx = pickle.load(f)
    else:
        classes_train = []
        features_train = []

        classes_test = []
        features_test = []

        dataset = datasets.ImageFolder(dataset_path, transform=preprocess, is_valid_file=None)
        class2idx = dataset.class_to_idx
        train_data_len = int(train_ratio*len(dataset))

        train_set, test_set = random_split(dataset, [train_data_len, len(dataset) - train_data_len])
        train_loader = DataLoader(train_set, num_workers=1, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_set, num_workers=1, batch_size=32, shuffle=False)

        for data, label in iter(train_loader):
            embeddings = model(data.to(device)).detach().cpu().numpy()
            features_train.append(embeddings)
            classes_train.extend(label.cpu().numpy())

        for data, label in iter(test_loader):
            embeddings = model(data.to(device)).detach().cpu().numpy()
            features_test.append(embeddings)
            classes_test.extend(label.cpu().numpy())

        
        features_train = np.concatenate(features_train, axis=0)
        features_test = np.concatenate(features_test, axis=0)
        classes_train = np.asarray(classes_train)
        classes_test = np.asarray(classes_test)

        with open(features_file, 'wb') as f:
            pickle.dump((features_train, features_test, classes_train, classes_test, class2idx), f)
    return features_train, features_test, classes_train, classes_test, class2idx

def split_per_class(dataset, train_ratio=0, seed=0):
    torch.manual_seed(seed)
    class_indices = {}  
    for idx, (data, label) in enumerate(dataset):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)

    train_indices = []
    test_indices = []

    for label, indices in class_indices.items():
        num_samples = len(indices)
        num_train_samples = max(1, int(train_ratio * num_samples))
        indices = torch.tensor(indices)
        indices = indices[torch.randperm(num_samples)]

        train_indices.extend(indices[:num_train_samples])
        test_indices.extend(indices[num_train_samples:])

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    return train_dataset, test_dataset

def dyntracker():
    global device
    device = torch.device('cuda:{}'.format(args.gpu))
    if args.model == "magface":
        import sys
        sys.path.append("..")
        from MagFace.inference.network_inf import builder_inf

        model = builder_inf(args)
        # model = model.cuda()
        model = model.to(device)
        model.eval()

        preprocess = transforms.Compose([
            transforms.Resize((112, 112), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ])
 
    print("Extracting features...", flush=True)
    global class2idx, val_people
    X_train_all, X_test_all, Y_train_all, Y_test_all, class2idx = get_dataset_features(model, args.dataset_name, args.model, preprocess, dataset_path=args.dataset_dir)

    print('Original training data:{}, test data:{}'.format(len(X_train_all), len(X_test_all)))

    val_people = args.names_list
    base_dir = args.attack_dir

    filter_cloak = lambda x: args.protected_file_match in x
    filter_uncloak = lambda x: args.unprotected_file_match in x
    # create protected user database
    cloak_dataset = datasets.ImageFolder(base_dir, transform=preprocess, is_valid_file=filter_cloak)
    uncloak_dataset = datasets.ImageFolder(base_dir, transform=preprocess, is_valid_file=filter_uncloak)
    cloak_dataset.class_to_idx = class2idx
    uncloak_dataset.class_to_idx = class2idx
    
    train_cloak, test_cloak = split_per_class(cloak_dataset, seed=25)
    train_uncloak, test_uncloak = split_per_class(uncloak_dataset, seed=25)
    
    user_feature, user_label = get_features(model, train_uncloak, sort=True)
    user_test_cloak_feature, user_test_cloak_label = get_features(model, test_cloak, sort=True)
    user_test_uncloak_feature, user_test_uncloak_label= get_features(model, test_uncloak, sort=True)

    if args.classifier == "linear":
        clf1 = LogisticRegression(random_state=0, n_jobs=-1, warm_start=False)
        clf1 = make_pipeline(StandardScaler(), clf1)
    else:
        clf1 = KNeighborsClassifier(n_neighbors=1, n_jobs=-1, metric='cosine')

    idx_train = np.asarray([y not in user_test_cloak_label for y in Y_train_all])
    idx_test = np.asarray([y not in user_test_cloak_label for y in Y_test_all])
    print(np.sum(idx_train), np.sum(idx_test))
    print((len(X_train_all)+len(X_test_all))-(np.sum(idx_train)+np.sum(idx_test)),len(cloak_dataset))
  
    X_train = np.concatenate((X_train_all[idx_train], user_feature))
    Y_train = np.concatenate((Y_train_all[idx_train], user_label))

    clf1 = clf1.fit(X_train, Y_train)
    round = 0
    print("Round {:d}, Test acc (protected user cloaked images): {:.4f}".format(round, clf1.score(user_test_cloak_feature, user_test_cloak_label)))
    
    added_to_train = np.zeros(len(user_test_cloak_feature), dtype=bool)


    while True:
        add_samples = 0
        new_data_add = False
        
        y_pred = clf1.predict(user_test_cloak_feature)
        correct_indices = [i for i in range(len(user_test_cloak_label)) if y_pred[i] == user_test_cloak_label[i] and not added_to_train[i]]
        for idx in correct_indices:
            X_train = np.concatenate((X_train, user_test_cloak_feature[idx][np.newaxis,:]))
            Y_train = np.append(Y_train, user_test_cloak_label[idx])
            add_samples += 1
            new_data_add = True
            added_to_train[idx] = True

        if new_data_add:
            clf1 = clf1.fit(X_train, Y_train)
            round += 1
            print("Round {:d}, add samples {:.3f}, Test acc (protected user cloaked images): {:.4f}".format(round, add_samples/count_subdirectories(base_dir), clf1.score(user_test_cloak_feature, user_test_cloak_label)))
        else:
            print('------------------------Round over------------------------')
            break

    print("Test acc (user cloaked): {:.4f}".format(clf1.score(user_test_cloak_feature, user_test_cloak_label)))
    print()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str,
                        help='the feature extractor', default='magface')
    parser.add_argument('--classifier', type=str,
                        help='the classifier', default='NN')
    parser.add_argument('--dataset-name', type=str, default='facescrub')
    parser.add_argument('--names-list', nargs='+', default=[], help="names of attacking users")
    parser.add_argument('--dataset-dir', help='path to unprotected facescrub directory', default="../dataset/facescrub")
    parser.add_argument('--attack-dir', help='path to protected facescrub directory', default="../dataset/facescrub_p")
    parser.add_argument('--unprotected-file-match', type=str,
                        help='pattern to match unprotected pictures', default='.jpeg')
    parser.add_argument('--protected-file-match', type=str,
                        help='pattern to match protected pictures', default='.jpg')
    parser.add_argument('--user-cloak', type=float,
                        help='rate of cloaked user images', default=0)
    # for MagFace
    parser.add_argument('--arch', default='iresnet100', type=str,
                        help='backbone architechture')
    parser.add_argument('--embedding_size', default=512, type=int,
                        help='The embedding feature size')
    parser.add_argument('--resume', default="magface_epoch_00025.pth", 
                        type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--cpu-mode', action='store_true', help='Use the CPU.')
    parser.add_argument('--gpu', type=int, default=0, help='Use the CPU.')

    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    dyntracker()
