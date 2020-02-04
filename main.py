import argparse
import utils.utils as utils
import torch
import config
from dataset import VideoDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import ToTensor, Normalization
from kiwiNet import Kiwi

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KiwiNet inference')
    parser.add_argument('-s', '--structure', nargs='+', type=int, help='Structure of the model', required=True)
    parser.add_argument('-l', '--load_state', type=str, help="state directory file to load before training")
    parser.add_argument('-d', "--dir", type=str, help="location", required=True)
    parser.add_argument('-v', "--video", type=str, help="video name", required=True)
    parser.add_argument('-o', "--out", type=str, help="output", required=True)
    parser.add_argument('--batch_size', type=int, default=16, help='number of data per batch')
    args = parser.parse_args()

    test_set = VideoDataset(args.dir, 6,
                            transform=transforms.Compose([
                                Normalization(),
                                ToTensor()
                            ]))

    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8)

    model = Kiwi(config.nb_kids, args.structure)
    model.cuda()

    if args.load_state is not None:
        model.load_state_dict(torch.load(args.load_state))

    model.eval()
    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            inputs = test_data['inputs']
            inputs = inputs.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            names_outputs = utils.output_processing(outputs, test_data['names_list'])
            utils.save_results(args.out,
                               test_data['video'],
                               test_data['frame'],
                               test_data['name'],
                               names_outputs)
