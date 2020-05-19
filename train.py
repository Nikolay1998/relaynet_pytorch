from multiprocessing.dummy import freeze_support

if __name__ == '__main__':
        freeze_support()
        import torch

        from networks.data_utils import get_imdb_data
        from networks.relay_net import ReLayNet
        from networks.solver import Solver

        train_data, test_data = get_imdb_data()

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True, num_workers=4)
        val_loader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False, num_workers=4)

        param = {
            'num_channels': 1,
            'num_filters': 64,
            'kernel_h': 3,
            'kernel_w': 7,
            'kernel_c': 1,
            'stride_conv': 1,
            'pool': 2,
            'stride_pool': 2,
            'num_class': 9
        }

        exp_dir_name = 'Exp06'

        relaynet_model = ReLayNet(param)
        solver = Solver(optim_args={"lr": 1e-2})
        solver.train(relaynet_model, train_loader, val_loader, log_nth=1, num_epochs=20, exp_dir_name=exp_dir_name)
