""" 
Main code for training and evaluating FedX.

"""

import argparse
import copy
import datetime
import json
import os
import random

import numpy as np
import torch
import torch.optim as optim
import wandb
import pandas as pd
import copy
from losses import js_loss, nt_xent
from model import init_nets
from utils import get_dataloader, mkdirs, partition_data, test_linear_fedX, set_logger, save_feature_bank, test_feature_distance
import ssl
import re
ssl._create_default_https_context = ssl._create_unverified_context

# record_data = None
# matrix_data = None

def get_gpu_memory():
    free_gpu_info = os.popen('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free').read()
    tep = re.findall('.*: (.*) MiB', free_gpu_info)
    gpu_dict = {}
    for one in range(len(tep)):
        gpu_dict[one] = int(tep[one])
    gpu_id = sorted(gpu_dict.items(), key=lambda item: item[1])[-1][0]
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(gpu_id))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet18", help="neural network used in training")
    parser.add_argument("--dataset", type=str, default="cifar100", help="dataset used for training")
    parser.add_argument("--net_config", type=lambda x: list(map(int, x.split(", "))))
    parser.add_argument("--partition", type=str, default="noniid", help="the data partitioning strategy")
    parser.add_argument("--batch-size", type=int, default=500, help="total sum of input batch size for training (default: 128)")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate (default: 0.1)")
    parser.add_argument("--epochs", type=int, default=1, help="number of local epochs")
    parser.add_argument("--n_parties", type=int, default=2, help="number of workers in a distributed cluster")
    parser.add_argument("--comm_round", type=int, default = 101, help="number of maximum communication roun")
    parser.add_argument("--init_seed", type=int, default=0, help="Random seed")
    parser.add_argument("--datadir", type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument("--reg", type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument("--logdir", type=str, required=False, default="./logs/", help="Log directory path")
    parser.add_argument("--modeldir", type=str, required=False, default="./models/", help="Model directory path")
    parser.add_argument(
        "--beta", type=float, default=10000, help="The parameter for the dirichlet distribution for data partitioning"
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

def recreate_feature(data, b):
    A = torch.matmul(b, data.t())
    u, s, v = torch.svd(A)
    u = torch.matmul(v, u.t())
    # u = u[:, :select]
    # b = b[:select]
    # f = torch.matmul(u, b)
    return u

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
    op_net = None,
    b_dict = None,
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
    # kl_loss = torch.nn.KLDivLoss()
    kl_loss = torch.nn.MSELoss()
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
    # train_iter = iter(train_dataloader)
    # iter_num = len(train_dataloader) * epochs

    if args.basis and round > 0:
        op_feature = torch.load(save_dir+ str(1-net_id)+'_'+str(round-1)+'_'+'.pth')

    for epoch in range(epochs):
        feature_all = []
        epoch_loss_collector = []
        for batch_idx, (x1, x2, target, _) in enumerate(train_dataloader):
            x1, x2, target = x1.cuda(), x2.cuda(), target.cuda()
            # for batch_idx, (x1, x2, target, _) in enumerate(train_dataloader):
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

            # pred1_original, pred1_pos, pred1_random = pred1.split([x1.size(0), x2.size(0), random_x.size(0)], dim=0)
            proj1_original, proj1_pos, proj1_random = proj1.split([x1.size(0), x2.size(0), random_x.size(0)], dim=0)
            # proj2_original, proj2_pos, proj2_random = proj2.split([x1.size(0), x2.size(0), random_x.size(0)], dim=0)
            feature_all.append(proj1_original.detach())
            loss_ours = 0
            # previous online-version
            if args.basis and round > 2:
                if len(proj1_original) < len(op_feature):
                    feature_tep = op_feature[:len(proj1_original)]
                    op_feature = op_feature[-(len(op_feature)-len(proj1_original)):]
                    if args.svd:
                        u, s, v = torch.svd(feature_tep)
                        sigma = torch.diag_embed(s)
                        b = torch.matmul(sigma, v.t())

                    else:
                        q, r = torch.linalg.qr(feature_tep)
                        b = r

                    if args.avg:
                        if args.svd:
                            u, s, v = torch.svd(proj1_original.detach())
                            sigma = torch.diag_embed(s)
                            b_me = torch.matmul(sigma, v.t())
                            w = u
                        else:
                            q, r = torch.linalg.qr(proj1_original.detach())
                            b_me = r
                            w = q
                        b_avg = (b + b_me) / 2
                        f_label = torch.matmul(w, b_avg)
                    else:
                        w = recreate_feature(proj1_original.detach(), b)
                        f_label = torch.matmul(w, b)

                    loss_ours += kl_loss(proj1_original, f_label)
            nt_local = nt_xent(proj1_original, proj1_pos, args.temperature)
            # nt_global = nt_xent(pred1_original, proj2_pos, args.temperature)
            # loss_nt = nt_local + nt_global
            loss_nt = nt_local

            # Relational losses (local, global)
            # js_global = js_loss(pred1_original, pred1_pos, proj2_random, args.temperature, args.tt)
            js_local = js_loss(proj1_original, proj1_pos, proj1_random, args.temperature, args.ts)
            # loss_js = js_global + js_local
            loss_js = js_local
            loss = loss_nt + loss_js + loss_ours
            loss.backward()
            torch.nn.utils.clip_grad_norm(net.parameters(), max_norm=1, norm_type=2)
            optimizer.step()
            epoch_loss_collector.append(loss.item())

        #     _, op_proj1, op_pred1 = op_net(x1)
        #     q, r = torch.linalg.qr(op_proj1)
        #     q_, r_ = torch.linalg.qr(proj1_original)
        #     r_avg = (r+r_)/2
        #     projection_tep = torch.matmul(q_, r_avg).detach()
        # q_tep = recreate_feature(proj1_original, r)
        # projection_tep = torch.matmul(q_tep, r).detach()
        #     loss_ours += kl_loss(proj1_original, projection_tep)

                # u_op, s_op, v_op = torch.svd(op_proj1)
                # sigma_op = torch.diag_embed(s_op)
                # b_op = torch.matmul(sigma_op, v_op.t())
                #
                # u, s, v = torch.svd(proj1_original.detach())
                #
                # if s.norm() < s_op.norm():
                #     u_tep = recreate_feature(proj1_original, b_op)
                #     projection_tep = torch.matmul(u_tep, b_op).detach()
                #     loss_ours += kl_loss(proj1_original, projection_tep)
                # loss_ours += kl_loss(proj1_original, op_proj1)

            # Contrastive losses (local, global)

            # f_tep = proj1_original.detach()
            # u, s, v = torch.svd(f_tep)
            # global record_data, matrix_data
            # record_data.loc[str(net_id)+'_'+str(round)+'_'+str(epoch)+'_'+str(batch_idx)] = s.cpu().tolist()
            # matrix_data[str(net_id)+'_'+str(round)+'_'+str(epoch)+'_'+str(batch_idx)] = v.cpu().numpy()


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
            # _, op_proj1, op_pred1 = op_net(x1)
            # op_loss = kl_loss(proj1_original, op_proj1)


    feature_all = torch.cat(feature_all, dim=0)
    feature_all = feature_all.detach()
    torch.save(feature_all, save_dir+ str(net_id)+'_'+str(round)+'_'+'.pth')
    # if round > 5:
    # base_dict = np.load('base_2.npy', allow_pickle=True).item()
    # for one in base_dict:
    #     if base_dict[one]['s'] > s.norm().item() and one+2 in base_dict:
    #         b_dict[1-net_id] = base_dict[one+2]['b']
    #         break
    #     else:
    #         b_dict[1-net_id] = None

    epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
    logger.info("Round: %d Loss: %f" % (round, epoch_loss))
    net.eval()
    logger.info(" ** Training complete **")

def feature_distillation(
    nets,
    args,
    public_data_loader,
    round=None,
    device="cpu",
):

    for batch_idx, (x1, x2, target, _) in enumerate(public_data_loader):
        ensemble_feature = []
        x1 = x1.cuda()
        for net_id, net in enumerate(nets.values()):
            net.eval()
            _, proj1, pred1 = net(x1)
            ensemble_feature.append(proj1.detach())
        ensemble_feature = sum(ensemble_feature)/len(ensemble_feature)

        for net_id, net in enumerate(nets.values()):
            kl_loss = torch.nn.MSELoss()
            if args.optimizer == "adam":
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr,
                                       weight_decay=args.reg)
            elif args.optimizer == "amsgrad":
                optimizer = optim.Adam(
                    filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg, amsgrad=True
                )
            elif args.optimizer == "sgd":
                optimizer = optim.SGD(
                    filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg
                )
            net.train()
            _, proj1, pred1 = net(x1)
            loss = kl_loss(proj1, ensemble_feature)
            loss.backward()
            torch.nn.utils.clip_grad_norm(net.parameters(), max_norm=1, norm_type=2)
            optimizer.step()

def p2p_train_nets(
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

    logger.info("Training {} networks".format(len(nets)))
    iter_list =[]
    for one in train_dl_local_dict:
        iter_list.append(len(train_dl_local_dict[one]))
    iter_num = int(sum(iter_list)/len(iter_list))


    optimizer_dict = {}
    for net_id, net in nets.items():
        net.cuda()
        if args.optimizer == "adam":
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg)
        elif args.optimizer == "amsgrad":
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg, amsgrad=True
            )
        elif args.optimizer == "sgd":
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg
            )
        optimizer_dict[net_id] = optimizer

    kl_loss = torch.nn.MSELoss()
    loss_dict = {}
    # feature_dict = {}
    projection_dict = {}
    b_dict = {}
    w_dict = {}
    loss_record = {i:[] for i in nets}
    for batch_id in range(iter_num):
        for net_id, net in nets.items():
            net.train()
            random_loader = copy.deepcopy(train_dl_local_dict[net_id])

            x1, x2, target, _ = iter(train_dl_local_dict[net_id]).next()
            x1, x2, target = x1.cuda(), x2.cuda(), target.cuda()
            optimizer_dict[net_id].zero_grad()
            target = target.long
            random_dataloader = iter(random_loader)
            random_x, _, _, _ = random_dataloader.next()
            random_x = random_x.cuda()
            all_x = torch.cat((x1, x2, random_x), dim=0).cuda()
            feature, proj, pred = net(all_x)
            proj_original, proj_pos, proj_random = proj.split([x1.size(0), x2.size(0), random_x.size(0)], dim=0)
            feature_original, _, _ = feature.split([x1.size(0), x2.size(0), random_x.size(0)], dim=0)
            # feature_dict[net_id] = feature_original
            projection_dict[net_id] = proj.detach()
            nt_local = nt_xent(proj_original, proj_pos, args.temperature)
            loss_nt = nt_local
            js_local = js_loss(proj_original, proj_pos, proj_random, args.temperature, args.ts)
            loss_js = js_local
            loss_dict[net_id] = loss_nt + loss_js
            if args.basis:
                if args.svd:
                    u, s, v = torch.svd(proj.detach())
                    sigma = torch.diag_embed(s)
                    b = torch.matmul(sigma, v.t())
                    w = u
                else:
                    w, b = torch.linalg.qr(proj.detach())
                w_dict[net_id] = w
                b_dict[net_id] = b

        if args.basis and round > 2:
            if args.avg:
                b_avg = []
                for net_id, b in b_dict.items():
                    b_avg.append(b)
                b_avg = sum(b_avg)/len(b_avg)
                for net_id, projection in projection_dict.items():
                    proj_label = torch.matmul(w_dict[net_id], b_avg)
                    loss_ours = kl_loss(projection, proj_label.detach())
                    loss_dict[net_id] += 0.5*loss_ours

            else:
                for net_id, projection in projection_dict.items():
                    for net_op, b_op in b_dict.items():
                        if net_id!=net_op:
                            proj_label = recreate_feature(projection, b_op)
                            loss_ours = kl_loss(projection, proj_label.detach())
                            loss_dict[net_id] += 0.5*loss_ours

        for net_id, optimizer in optimizer_dict.items():
            loss_dict[net_id].backward()
            loss_record[net_id].append(loss_dict[net_id].item())
            torch.nn.utils.clip_grad_norm(nets[net_id].parameters(), max_norm=1, norm_type=2)
            optimizer.step()
            nets[net_id].eval()

    for net_id, loss_list in loss_record.items():
        loss_avg = sum(loss_list)/len(loss_list)
        logger.info("Round: %d Client %d Loss: %f" % (round, net_id, loss_avg))
    logger.info(" ** Training complete **")
    return nets

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
        test_dl_local = test_dl[net_id]
        op_net = nets[1-net_id]
        # op_net = None
        train_net_fedx(
        net_id,
        net,
        global_model,
        train_dl_local,
        val_dl_local,
        test_dl_local,
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

    # get_gpu_memory()
    args = get_args()
    method_list = [(1,0,0,0,0),(0,1,0,0,0),(0,0,0,0,0),(0,0,1,0,0),(0,0,1,0,1),(0,0,1,1,0),(0,0,1,1,1)]
    for one in method_list[:]:
        args.aggregation, args.distillation, args.basis, args.svd, args.avg = one
        # Create directory to save log and model
        mkdirs(args.logdir)
        mkdirs(args.modeldir)
        argument_path = f"{args.dataset}-{args.batch_size}-{args.n_parties}-{args.temperature}-{args.tt}-{args.ts}-{args.epochs}_arguments-%s.json" % datetime.datetime.now().strftime(
            "%Y-%m-%d-%H%M-%S"
        )
        if args.aggregation:
            method = 'weights_aggregation'
        elif args.distillation:
            method = 'distillation_aggregation'
        elif args.basis:
            method = 'ours'
            if args.svd:
                method += '_svd'
            else:
                method += '_qr'
            if args.avg:
                method += '_avg'
            else:
                method += '_semantics'
        else:
            method = 'individual'

        args.name = '{}_clients_{}_alpha_{}'.format(args.n_parties, args.beta, method)

        save_dir = args.name + '/'
        model_dir = './models/' + save_dir

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            os.makedirs(model_dir)

        wandb.init(project='Basis_Aggregation_{}'.format(args.dataset), name=args.name, entity='peilab')
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
                # party_list_rounds.append(party_list[:1])

        n_classes = len(np.unique(y_train))

        # Get global dataloader (only used for evaluation)
        (train_dl_global, val_dl_global, test_dl, train_ds_global, _, test_ds_global) = get_dataloader(
            args.dataset, args.datadir, args.batch_size, args.batch_size * 2
        )

        # print("len train_dl_global:", len(train_ds_global))
        # train_dl = None
        # data_size = len(test_ds_global)

        # Initializing net from each local party.
        # logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.n_parties, args, device="cpu")

        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 1, args, device="cpu")
        global_model = global_models[0]
        n_comm_rounds = args.comm_round

        train_dl_local_dict = {}
        val_dl_local_dict = {}
        test_dl_local_dict = {}
        net_id = 0
        permute_record = list(range(10))
        np.random.shuffle(permute_record)
        # def target_transform(label):
        #     label = permute_record[label]
        #     return label
        # Distribute dataset and dataloader to each local party
        # We use two dataloaders for training FedX (train_dataloader, random_dataloader),
        # and their batch sizes (args.batch_size // 2) are summed up to args.batch_size
        for net in nets:
            dataidxs = net_dataidx_map[net_id]
            # if net_id ==0:
            train_dl_local, val_dl_local, _, _, _, _ = get_dataloader(
                args.dataset, args.datadir, args.batch_size // 2, args.batch_size * 2, dataidxs
            )
            # else:
            #     train_dl_local, val_dl_local, _, _, _, _ = get_dataloader(
            #         args.dataset, args.datadir, args.batch_size // 2, args.batch_size * 2, dataidxs, target_transform=target_transform
            #     )
            _, test_dl_local, _, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size //2, args.batch_size*2, net_dataidx_map['private_test'][net_id])
            train_dl_local_dict[net_id] = train_dl_local
            val_dl_local_dict[net_id] = val_dl_local
            test_dl_local_dict[net_id] = test_dl_local
            net_id += 1

        public_data_loader, _, _, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size//2, args.batch_size * 2, net_dataidx_map['public'])
        # Main training communication loop.
        # state_dict = torch.load('./models/ckpt_1_self_teaching/local_model_0.pth')
        # nets[1].load_state_dict(state_dict)
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]
            # Download global model from (virtual) central server
            nets_this_round = {k: nets[k] for k in party_list_this_round}

            if args.aggregation:
                global_w = global_model.state_dict()
                for net in nets_this_round.values():
                    net.load_state_dict(global_w)

            # Train local model with local data
            import time
            start_time = time.time()
            # if args.basis:
            nets = p2p_train_nets(
                nets_this_round,
                args,
                net_dataidx_map,
                train_dl_local_dict,
                val_dl_local_dict,
                train_dl=None,
                test_dl=test_dl_local_dict,
                global_model=global_model,
                device=device,
                round=round
            )
            # else:
            #     nets = local_train_net(
            #         nets_this_round,
            #         args,
            #         net_dataidx_map,
            #         train_dl_local_dict,
            #         val_dl_local_dict,
            #         train_dl=None,
            #         test_dl=test_dl_local_dict,
            #         global_model=global_model,
            #         device=device,
            #         round=round
            #     )
            if args.distillation:
                feature_distillation(
                    nets_this_round,
                    args,
                    public_data_loader,
                    device = device,
                    round = round
                )

            total_data_points = sum([len(net_dataidx_map[r]) for r in range(args.n_parties)])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in range(args.n_parties)]

            log_info = {}
            if args.aggregation:
                for net_id, net in enumerate(nets_this_round.values()):
                    net_para = net.state_dict()
                    if net_id == 0:
                        for key in net_para:
                            global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                    else:
                        for key in net_para:
                            global_w[key] += net_para[key] * fed_avg_freqs[net_id]

                global_model.load_state_dict(copy.deepcopy(global_w))

            if round % 5 == 0 or round > n_comm_rounds-5:
                acc_list = []
                for net_id, net in enumerate(nets_this_round.values()):
                    test_acc_1, test_acc_5 = test_linear_fedX(net, val_dl_local_dict[net_id], test_dl_local_dict[net_id])
                    logger.info(">> Private Model {} Test accuracy Top1: {}".format(net_id, test_acc_1))
                    logger.info(">> Private Model {} Test accuracy Top5: {}".format(net_id, test_acc_5))
                    log_info['acc_top1_client{}'.format(net_id)] = test_acc_1
                    log_info['acc_top5_client{}'.format(net_id)] = test_acc_5
                    acc_list.append(test_acc_1)
                log_info['avg_acc'] = sum(acc_list)/len(acc_list)

            feature_distance = test_feature_distance(nets, test_dl)
            log_info['feature_dis'] = feature_distance
            log_info['round'] = round
            wandb.log(log_info)

            # for net_id, net in nets_this_round.items():
            #     save_feature_bank(net, val_dl_global, test_dl, save_dir+ str(net_id)+'_'+str(round)+'_')

        # record_data.to_csv('data.csv')
        # np.save('matrix.npy', matrix_data)

        # Save the final round's local and global models
        if args.aggregation:
            torch.save(global_model.state_dict(), args.modeldir + save_dir + "global_model.pth")

        else:
            for net_id,  net in nets_this_round.items():
                torch.save(net.state_dict(), args.modeldir + save_dir+ "local_model_{}".format(net_id) + ".pth")
        wandb.finish()


