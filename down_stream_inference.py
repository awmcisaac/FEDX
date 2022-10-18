"""
Main code for training and evaluating FedX.

"""

import argparse
import copy
import datetime
import json
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import wandb
import pandas as pd
import copy
from losses import js_loss, nt_xent
from model import init_nets
from utils import get_dataloader, mkdirs, partition_data, test_linear_fedX, set_logger, save_feature_bank, test_featrue_bank
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# record_data = None
# matrix_data = None

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet18", help="neural network used in training")
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset used for training")
    parser.add_argument("--net_config", type=lambda x: list(map(int, x.split(", "))))
    parser.add_argument("--partition", type=str, default="noniid", help="the data partitioning strategy")
    parser.add_argument("--batch-size", type=int, default=1000, help="total sum of input batch size for training (default: 128)")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate (default: 0.1)")
    parser.add_argument("--epochs", type=int, default=1, help="number of local epochs")
    parser.add_argument("--n_parties", type=int, default=2, help="number of workers in a distributed cluster")
    parser.add_argument("--comm_round", type=int, default=50, help="number of maximum communication roun")
    parser.add_argument("--init_seed", type=int, default=0, help="Random seed")
    parser.add_argument("--datadir", type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument("--reg", type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument("--logdir", type=str, required=False, default="./logs/", help="Log directory path")
    parser.add_argument("--modeldir", type=str, required=False, default="./models/", help="Model directory path")
    parser.add_argument(
        "--beta", type=float, default=100000, help="The parameter for the dirichlet distribution for data partitioning"
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="The device to run the program")
    parser.add_argument("--optimizer", type=str, default="sgd", help="the optimizer")
    parser.add_argument("--out_dim", type=int, default=256, help="the output dimension for the projection layer")
    parser.add_argument("--temperature", type=float, default=0.1, help="the temperature parameter for contrastive loss")
    parser.add_argument("--tt", type=float, default=0.1, help="the temperature parameter for js loss in teacher model")
    parser.add_argument("--ts", type=float, default=0.1, help="the temperature parameter for js loss in student model")
    parser.add_argument("--sample_fraction", type=float, default=1.0, help="how many clients are sampled in each round")
    args = parser.parse_args()

    # global record_data, matrix_data
    # record_data = pd.DataFrame([], columns=[i for i in range(args.out_dim)])
    # matrix_data = {}
    return args


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

def train_net_fedx(
    net_id,
    net,
    global_net,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    epochs,
    lr,
    args_optimizer,
    temperature,
    args,
    round,
    device="cpu",
    op_net = None
):
    net.cuda()
    # global_net.cuda()
    logger.info("Training network %s" % str(net_id))
    logger.info("n_training: %d" % len(train_dataloader))
    logger.info("n_test: %d" % len(test_dataloader))
    if op_net:
        op_net.eval()
    # Set optimizer
    ce_loss = torch.nn.CrossEntropyLoss()
    kl_loss = torch.nn.KLDivLoss()

    if args_optimizer == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == "amsgrad":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg, amsgrad=True
        )
    elif args_optimizer == "sgd":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9, weight_decay=args.reg
        )
    net.train()
    # global_net.eval()

    # Random dataloader for relational loss
    random_loader = copy.deepcopy(train_dataloader)
    random_dataloader = iter(random_loader)

    for epoch in range(epochs):
        feature_all = None
        epoch_loss_collector = []
        for batch_idx, (x1, x2, target, _) in enumerate(train_dataloader):
            x1, x2, target = x1.cuda(), x2.cuda(), target.cuda()
            optimizer.zero_grad()
            target = target.long()

            try:
                random_x, _, _, _ = random_dataloader.next()
            except:
                random_dataloader = iter(random_loader)
                random_x, _, _, _ = random_dataloader.next()
            random_x = random_x.cuda()

            all_x = torch.cat((x1, x2, random_x), dim=0).cuda()
            _, proj1, pred1 = net(all_x)
            # with torch.no_grad():
            #     _, proj2, pred2 = global_net(all_x)

            pred1_original, pred1_pos, pred1_random = pred1.split([x1.size(0), x2.size(0), random_x.size(0)], dim=0)
            proj1_original, proj1_pos, proj1_random = proj1.split([x1.size(0), x2.size(0), random_x.size(0)], dim=0)
            # proj2_original, proj2_pos, proj2_random = proj2.split([x1.size(0), x2.size(0), random_x.size(0)], dim=0)

            if feature_all == None:
                feature_all = proj1_original
            else:
                feature_all = torch.cat((feature_all, proj1_original), dim=0)

            # Contrastive losses (local, global)

            # f_tep = proj1_original.detach()
            # u, s, v = torch.svd(f_tep)
            # global record_data, matrix_data
            # record_data.loc[str(net_id)+'_'+str(round)+'_'+str(epoch)+'_'+str(batch_idx)] = s.cpu().tolist()
            # matrix_data[str(net_id)+'_'+str(round)+'_'+str(epoch)+'_'+str(batch_idx)] = v.cpu().numpy()
            nt_local = nt_xent(proj1_original, proj1_pos, args.temperature)
            # nt_global = nt_xent(pred1_original, proj2_pos, args.temperature)
            # loss_nt = nt_local + nt_global
            loss_nt = nt_local

            # Relational losses (local, global)
            # js_global = js_loss(pred1_original, pred1_pos, proj2_random, args.temperature, args.tt)
            js_local = js_loss(proj1_original, proj1_pos, proj1_random, args.temperature, args.ts)
            # loss_js = js_global + js_local
            loss_js = js_local

            # if hasattr(net, 'v'):
            #     v_global = net.v
            #     u, s, v = torch.svd(proj1_original.detach())
            #     A = torch.diag_embed(s)
            #     proj1_global = torch.matmul(u, torch.matmul(A, v_global.transpose(-2, -1)))
            #     loss_ours = kl_loss(proj1_original, proj1_global)
            # else:
            #     loss_ours = 0

            # loss_supervised = ce_loss(pred1_original, target)
            # loss = loss_supervised

            _, op_proj1, op_pred1 = op_net(x1)
            op_loss = kl_loss(proj1_original, op_proj1)
            loss = loss_nt + loss_js
            loss.backward()
            optimizer.step()
            epoch_loss_collector.append(loss.item())

        feature_all = feature_all.detach()
        torch.save(feature_all,'./ckpt_2_individual_guided/'+ str(net_id)+'_'+str(round)+'_'+str(epoch)+'.pth')
        u, s, v = torch.svd(feature_all)
        net.v = v
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info("Epoch: %d Loss: %f" % (epoch, epoch_loss))

    net.eval()
    logger.info(" ** Training complete **")


def local_train_net(
    nets,
    args,
    net_dataidx_map,
    train_dl_local_dict,
    val_dl_local_dict,
    train_dl=None,
    test_dl=None,
    global_model=None,
    prev_model_pool=None,
    round=None,
    device="cpu",
):

    if global_model:
        global_model.cuda()

    n_epoch = args.epochs
    for net_id, net in nets.items():
        dataidxs = net_dataidx_map[net_id]
        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        train_dl_local = train_dl_local_dict[net_id]
        val_dl_local = val_dl_local_dict[net_id]
        op_net = nets[1-net_id]
        train_net_fedx(
            net_id,
            net,
            global_model,
            train_dl_local,
            val_dl_local,
            test_dl,
            n_epoch,
            args.lr,
            args.optimizer,
            args.temperature,
            args,
            round,
            device=device,
            op_net = op_net,
        )

    if global_model:
        global_model.to("cpu")
    return nets


if __name__ == "__main__":
    # wandb.init(project='trial', name='2_clients_guided', entity='joey61')
    args = get_args()
    # Create directory to save log and model
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    argument_path = f"{args.dataset}-{args.batch_size}-{args.n_parties}-{args.temperature}-{args.tt}-{args.ts}-{args.epochs}_arguments-%s.json" % datetime.datetime.now().strftime(
        "%Y-%m-%d-%H%M-%S"
    )

    # Save arguments
    with open(os.path.join(args.logdir, argument_path), "w") as f:
        json.dump(str(args), f)

    device = torch.device(args.device)

    # Set logger
    logger = set_logger(args)
    logger.info(device)

    # Set seed
    set_seed(args.init_seed)

    # Data partitioning with respect to the number of parties
    logger.info("Partitioning data")
    (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts) = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta
    )

    n_party_per_round = int(args.n_parties * args.sample_fraction)
    party_list = [i for i in range(args.n_parties)]
    party_list_rounds = []

    if n_party_per_round != args.n_parties:
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)

    n_classes = len(np.unique(y_train))

    # Get global dataloader (only used for evaluation)
    (train_dl_global, val_dl_global, test_dl, train_ds_global, _, test_ds_global) = get_dataloader(
        args.dataset, args.datadir, args.batch_size, args.batch_size * 2
    )

    print("len train_dl_global:", len(train_ds_global))
    train_dl = None
    data_size = len(test_ds_global)

    # Initializing net from each local party.
    logger.info("Initializing nets")
    nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.n_parties, args, device="cpu")

    # for i in range(len(nets)):
    #     state_dict = torch.load(args.modeldir + "/guided/local_model_{}".format(i) + ".pth")
    #     nets[i].load_state_dict(state_dict)
    #
    #     test_acc_1, test_acc_5 = test_linear_fedX(nets[i], val_dl_global, test_dl)
    #     logger.info(">> Private Model {} Test accuracy Top1 {}".format(i, test_acc_1))
    #     logger.info(">> Private Model {} Test accuracy Top5 {}".format(i, test_acc_5))

    # save_dir = './ckpt_1_self_taught_KD/0_49_'
    # test_acc_1, test_acc_5 = test_featrue_bank(save_dir, 512, 10)
    # logger.info(">> Private Model {} Test accuracy Top1 {}".format(0, test_acc_1))

    # acc = np.load('teacher_student_acc_data.npy', allow_pickle=True).item()
    acc = {}
    # acc_teacher = []
    # acc_student = []
    acc_KD = []
    acc_QR = []
    acc_QR_avg = []
    acc_teacher = []


    for epoch in range(50):
        save_dir = './ckpt_1_self_teaching/{}_{}_'.format(0, epoch)
        test_acc_1, test_acc_5 = test_featrue_bank(save_dir, 512, 10)
        logger.info(">> Private Model {} Test accuracy Top1 {}".format(0, test_acc_1))
        acc_teacher.append(test_acc_1)

        save_dir = './ckpt_1_self_taught_KD/{}_{}_'.format(0, epoch)
        test_acc_1, test_acc_5 = test_featrue_bank(save_dir, 512, 10)
        logger.info(">> Private Model {} Test accuracy Top1 {}".format(1, test_acc_1))
        acc_KD.append(test_acc_1)

        save_dir = './ckpt_1_self_taught_QR/{}_{}_'.format(0, epoch)
        test_acc_1, test_acc_5 = test_featrue_bank(save_dir, 512, 10)
        logger.info(">> Private Model {} Test accuracy Top1 {}".format(0, test_acc_1))
        acc_QR.append(test_acc_1)

        save_dir = './ckpt_1_self_taught_QR_avg/{}_{}_'.format(0, epoch)
        test_acc_1, test_acc_5 = test_featrue_bank(save_dir, 512, 10)
        logger.info(">> Private Model {} Test accuracy Top1 {}".format(0, test_acc_1))
        acc_QR_avg.append(test_acc_1)
        print(epoch)

    # acc = {
    #     'teacher': acc_teacher,
    #     'student': acc_student,
    # }
    acc['original'] = acc_teacher
    acc['KD'] = acc_KD
    acc['QR'] = acc_QR
    acc['QR_avg'] = acc_QR_avg
    np.save('acc_data.npy', acc)
    acc_teacher = acc['original']
    acc_KD = acc['KD']
    acc_QR = acc['QR']
    acc_QR_avg = acc['QR_avg']

    # plt.plot(acc_student, label = 'student')
    # plt.plot(acc_teacher, label='teacher')
    plt.plot(acc_KD, label = 'KD')
    plt.plot(acc_QR, label='QR')
    plt.plot(acc_QR_avg, label = 'QR_AVG')
    plt.plot(acc_teacher, label = 'Original')
    plt.legend()
    plt.savefig('compare.png')
    plt.show()


    # global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 1, args, device="cpu")

    # global_model = global_models[0]
    # n_comm_rounds = args.comm_round
    #
    # train_dl_local_dict = {}
    # val_dl_local_dict = {}
    # net_id = 0
    # permute_record = list(range(10))
    # np.random.shuffle(permute_record)
    # # def target_transform(label):
    # #     label = permute_record[label]
    # #     return label
    #
    # # Distribute dataset and dataloader to each local party
    # # We use two dataloaders for training FedX (train_dataloader, random_dataloader),
    # # and their batch sizes (args.batch_size // 2) are summed up to args.batch_size
    # for net in nets:
    #     dataidxs = net_dataidx_map[net_id]
    #     # if net_id ==0:
    #     train_dl_local, val_dl_local, _, _, _, _ = get_dataloader(
    #         args.dataset, args.datadir, args.batch_size // 2, args.batch_size * 2, dataidxs
    #     )
    #     # else:
    #     #     train_dl_local, val_dl_local, _, _, _, _ = get_dataloader(
    #     #         args.dataset, args.datadir, args.batch_size // 2, args.batch_size * 2, dataidxs, target_transform=target_transform
    #     #     )
    #     train_dl_local_dict[net_id] = train_dl_local
    #     val_dl_local_dict[net_id] = val_dl_local
    #     net_id += 1
    #
    # # Main training communication loop.
    # for round in range(n_comm_rounds):
    #     logger.info("in comm round:" + str(round))
    #     party_list_this_round = party_list_rounds[round]
    #
    #     # Download global model from (virtual) central server
    #     # global_w = global_model.state_dict()
    #     nets_this_round = {k: nets[k] for k in party_list_this_round}
    #     # for net in nets_this_round.values():
    #     #     net.load_state_dict(global_w)
    #     # Train local model with local data
    #     nets = local_train_net(
    #         nets_this_round,
    #         args,
    #         net_dataidx_map,
    #         train_dl_local_dict,
    #         val_dl_local_dict,
    #         train_dl=train_dl,
    #         test_dl=test_dl,
    #         global_model=global_model,
    #         device=device,
    #         round=round
    #     )
    #
    #     total_data_points = sum([len(net_dataidx_map[r]) for r in range(args.n_parties)])
    #     fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in range(args.n_parties)]
    #
    #
    #     # for net_id, net in enumerate(nets_this_round.values()):
    #     #     if net_id == 0:
    #     #         global_v = net.v * fed_avg_freqs[net_id]
    #     #     else:
    #     #         global_v += net.v * fed_avg_freqs[net_id]
    #     #
    #     # for net_id, net in enumerate(nets_this_round.values()):
    #     #     net.v = global_v
    #
    #     # Averaging the local models' parameters to get global model
    #     # for net_id, net in enumerate(nets_this_round.values()):
    #     #     net_para = net.state_dict()
    #     #     if net_id == 0:
    #     #         for key in net_para:
    #     #             global_w[key] = net_para[key] * fed_avg_freqs[net_id]
    #     #     else:
    #     #         for key in net_para:
    #     #             global_w[key] += net_para[key] * fed_avg_freqs[net_id]
    #
    #     # global_model.load_state_dict(copy.deepcopy(global_w))
    #     # global_model.cuda()
    #
    #     # Evaluating the global model
    #     # test_acc_1, test_acc_5 = test_linear_fedX(nets_this_round[0], val_dl_global, test_dl)
    #     # logger.info(">> Private Model 0 Test accuracy Top1: %f" % test_acc_1)
    #     # logger.info(">> Private Model 0 Test accuracy Top5: %f" % test_acc_5)
    #
    #     # test_acc_1, test_acc_5 = test_linear_fedX(nets_this_round[1], val_dl_global, test_dl)
    #     # logger.info(">> Private Model 1 Test accuracy Top1: %f" % test_acc_1)
    #     # logger.info(">> Private Model 1 Test accuracy Top5: %f" % test_acc_5)
    #     for net_id, net in nets.items():
    #         save_feature_bank(net, val_dl_global, './ckpt_2_individual_guided/'+ str(net_id)+'_'+str(round)+'test.pth')
    #
    # # record_data.to_csv('data.csv')
    # # np.save('matrix.npy', matrix_data)
    #
    # # Save the final round's local and global models
    # # torch.save(global_model.state_dict(), args.modeldir + "globalmodel" + args.log_file_name + ".pth")
    # for net_id,  net in nets.items():
    #     torch.save(net.state_dict(), args.modeldir + "/guided/local_model_{}".format(net_id) + ".pth")

