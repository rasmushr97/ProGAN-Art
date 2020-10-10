from data import create_h5_dataloader
from train import Trainer
import argparse

scale_config = [
    {'scale': 4, 'epochs': 4, 'batch_size': 64},
    {'scale': 8, 'epochs': 8, 'batch_size': 64},
    {'scale': 16, 'epochs': 12, 'batch_size': 32},
    {'scale': 32, 'epochs': 12, 'batch_size': 32},
    {'scale': 64, 'epochs': 16, 'batch_size': 16},
    {'scale': 128, 'epochs': 16, 'batch_size': 4},
    {'scale': 256, 'epochs': 20, 'batch_size': 1},
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='D:/wikiart256/fulldata.hdf5')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--features', type=int, default=8)
    parser.add_argument('--channels', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--zdim', type=int, default=256)
    args = parser.parse_args()

    trainer = Trainer(args.zdim, args.features, args.channels, args.device, lr=args.lr, log=True)

    for conf in scale_config:
        dataloader = create_h5_dataloader(args.data_path, conf['scale'], conf['batch_size'])
        print(f'Scale: {conf["scale"]}, Batch Size: {conf["batch_size"]}')

        for index, epoch in enumerate(range(conf['epochs'])):
            alpha = (index + 1) / conf['epochs']
            print(f'Alpha: {alpha:.2f}')
            trainer.set_alpha(alpha)
            trainer.train_epoch(dataloader)

        trainer.scale()
    
    trainer.save('')
