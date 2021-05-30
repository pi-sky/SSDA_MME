from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model.resnet import resnet34
from model.basenet import AlexNetBase, VGGBase, Predictor, Predictor_deep, FCnet
from utils.utils import weights_init
from utils.lr_schedule import inv_lr_scheduler
from utils.get_loader import get_dataloaders, get_domain_from_path
# from utils.return_dataset import return_dataset
from utils.loss import entropy, adentropy
# Training settings
parser = argparse.ArgumentParser(description='SSDA Classification')
parser.add_argument('--steps', type=int, default=50000, metavar='N',
                    help='maximum number of iterations '
                         'to train (default: 50000)')
parser.add_argument('--method', type=str, default='MME',
                    choices=['S+T', 'ENT', 'MME'],
                    help='MME is proposed method, ENT is entropy minimization,'
                         ' S+T is training only on labeled examples')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--multi', type=float, default=0.1, metavar='MLT',
                    help='learning rate multiplication')
parser.add_argument('--T', type=float, default=0.05, metavar='T',
                    help='temperature (default: 0.05)')
parser.add_argument('--lamda', type=float, default=0.1, metavar='LAM',
                    help='value of lamda')
parser.add_argument('--save_check', action='store_true', default=False,
                    help='save checkpoint or not')
parser.add_argument('--checkpath', type=str, default='./save_model_ssda',
                    help='dir to save checkpoint')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging '
                         'training status')
parser.add_argument('--save_interval', type=int, default=500, metavar='N',
                    help='how many batches to wait before saving a model')
parser.add_argument('--net', type=str, default='alexnet',
                    help='which network to use')
parser.add_argument('--source', type=str, default='Amazon',
                    help='source domain')
parser.add_argument('--target', type=str, default='Dslr',
                    help='target domain')
parser.add_argument('--dataset', type=str, default='office-31',
                    choices=['multi', 'office-31', 'office_home'],
                    help='the name of dataset')
parser.add_argument('--num', type=int, default=3,
                    help='number of labeled examples in the target')
parser.add_argument('--patience', type=int, default=5, metavar='S',
                    help='early stopping to wait for improvment '
                         'before terminating. (default: 5 (5000 iterations))')
parser.add_argument('--early', action='store_false', default=True,
                    help='early stopping on validation or not')
parser.add_argument('--source_data_path', type=str, default='data_office/amazon_amazon.csv',
                    help='data path for csv file from source')
parser.add_argument('--target_data_path', type=str, default='data_office/dslr_dslr.csv',
                    help='data path for csv file from target')
parser.add_argument('--frac', type=float, default=0.06,
                    help='Fraction of source data to use as labeled source samples')

args = parser.parse_args()
batch_size = 3
source_domain = get_domain_from_path(args.source_data_path)
target_domain = get_domain_from_path(args.target_data_path)

print('Dataset %s Source %s Target %s Labeled num perclass %s Network %s' %
      (args.dataset, source_domain, target_domain, args.num, args.net))
# source_loader, target_loader_unl, target_loader_val, \
#     target_loader_test = get_dataloaders(args.data_path, batch_size)

source_loader, source_loader_unl, source_loader_test = get_dataloaders(args.source_data_path,
                                                                       'source', batch_size, frac=args.frac)
target_loader_unl, target_loader_val, target_loader_test = get_dataloaders(args.target_data_path,
                                                                           'target', batch_size)

class_list = [x for x in range(31)]

use_gpu = torch.cuda.is_available()
record_dir = 'record/%s/%s/%s' % (args.dataset, source_domain, target_domain)
if not os.path.exists(record_dir):
    os.makedirs(record_dir)
record_file = os.path.join(record_dir,
                           '%s_source_%s_to_target_%s_num_class_%s' %
                           (args.method, source_domain,
                            target_domain, str(len(class_list))))

final_record_dir = 'record/%s' % (args.dataset)
if not os.path.exists(final_record_dir):
    os.makedirs(final_record_dir)
final_record_file = os.path.join(final_record_dir, 'Final_records_for_all_domains')

torch.cuda.manual_seed(args.seed)
if args.net == 'resnet34':
    G = resnet34()
    inc = 512
elif args.net == "alexnet":
    # G = AlexNetBase()
    # inc = 4096
    G = FCnet()
    inc = 2048
elif args.net == "vgg":
    G = VGGBase()
    inc = 4096
else:
    raise ValueError('Model cannot be recognized.')

params = []
for key, value in dict(G.named_parameters()).items():
    if value.requires_grad:
        if 'classifier' not in key:
            params += [{'params': [value], 'lr': args.multi,
                        'weight_decay': 0.0005}]
        else:
            params += [{'params': [value], 'lr': args.multi * 10,
                        'weight_decay': 0.0005}]

if "resnet" in args.net:
    F1 = Predictor_deep(num_class=len(class_list),
                        inc=inc)
else:
    F1 = Predictor(num_class=len(class_list), inc=inc//2,
                   temp=args.T)
weights_init(F1)
lr = args.lr
G.cuda()
F1.cuda()

im_data_s = torch.FloatTensor(1)
im_data_su = torch.FloatTensor(1)
im_data_t = torch.FloatTensor(1)
im_data_tu = torch.FloatTensor(1)
gt_labels_s = torch.LongTensor(1)
gt_labels_t = torch.LongTensor(1)
sample_labels_t = torch.LongTensor(1)
sample_labels_s = torch.LongTensor(1)

im_data_s = im_data_s.cuda()
im_data_su = im_data_su.cuda()
im_data_t = im_data_t.cuda()
im_data_tu = im_data_tu.cuda()
gt_labels_s = gt_labels_s.cuda()
gt_labels_t = gt_labels_t.cuda()
sample_labels_t = sample_labels_t.cuda()
sample_labels_s = sample_labels_s.cuda()

im_data_s.requires_grad_()
im_data_su.requires_grad_()
im_data_t.requires_grad_()
im_data_tu.requires_grad_()
# gt_labels_s.requires_grad_()
# gt_labels_t.requires_grad_()
# sample_labels_t = Variable(sample_labels_t)
# sample_labels_s = Variable(sample_labels_s)

if os.path.exists(args.checkpath) == False:
    os.mkdir(args.checkpath)


def train():
    G.train()
    F1.train()
    optimizer_g = optim.SGD(params, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)
    optimizer_f = optim.SGD(list(F1.parameters()), lr=1.0, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)
    loss_domain = nn.BCELoss().cuda()

    def zero_grad_all():
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
    param_lr_g = []
    for param_group in optimizer_g.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_f = []
    for param_group in optimizer_f.param_groups:
        param_lr_f.append(param_group["lr"])
    criterion = nn.CrossEntropyLoss().cuda()
    all_step = args.steps
    data_iter_s = iter(source_loader)
    data_iter_s_unl = iter(source_loader_unl)
    data_iter_t_unl = iter(target_loader_unl)
    len_train_source = len(source_loader)
    len_train_source_unl = len(source_loader_unl)
    len_train_target_semi = len(target_loader_unl)
    best_acc_source = 0
    best_acc = 0
    counter = 0
    for step in range(all_step):
        optimizer_g = inv_lr_scheduler(param_lr_g, optimizer_g, step,
                                       init_lr=args.lr)
        optimizer_f = inv_lr_scheduler(param_lr_f, optimizer_f, step,
                                       init_lr=args.lr)
        lr = optimizer_f.param_groups[0]['lr']
        if step % len_train_source_unl == 0:
            data_iter_s_unl = iter(source_loader_unl)
        if step % len_train_target_semi == 0:
            data_iter_t_unl = iter(target_loader_unl)
        if step % len_train_source == 0:
            data_iter_s = iter(source_loader)
        # data_t = next(data_iter_t)
        data_t_unl = next(data_iter_t_unl)
        data_s_unl = next(data_iter_s_unl)
        data_s = next(data_iter_s)
        im_data_s.data.resize_(data_s[0].size()).copy_(data_s[0])
        gt_labels_s.data.resize_(data_s[1].size()).copy_(data_s[1])
        im_data_su.data.resize_(data_s_unl[0].size()).copy_(data_s_unl[0])
        # gt_labels_t.data.resize_(data_t[1].size()).copy_(data_t[1])
        im_data_tu.data.resize_(data_t_unl[0].size()).copy_(data_t_unl[0])
        zero_grad_all()
        # data = torch.cat((im_data_s, im_data_t), 0)
        # print(im_data_s.size())
        # print(gt_labels_s.size())
        data = im_data_s
        # target = torch.cat((gt_labels_s, gt_labels_t), 0)
        target = gt_labels_s
        # print(target)

        # training on labeled source data
        output, domain_output = G(data, lamda = args.lamda)
        out1 = F1(output)

        domain_label = torch.zeros(batch_size)
        domain_label = domain_label.float().cuda()
        err_s_domain = loss_domain(domain_output, domain_label)

        #training on unlabeled source data
        output_su, domain_output = G(im_data_su, lamda = args.lamda)
        out2 = F1(output_su)
        domain_label = torch.zeros(batch_size)
        domain_label = domain_label.float().cuda()
        err_s_un_domain = loss_domain(domain_output, domain_label)

        #training on unlabeled target data
        tar_data = im_data_tu
        output_target_u, domain_output = G(tar_data, lamda = args.lamda)
        domain_label = torch.ones(batch_size)
        domain_label = domain_label.float().cuda()
        err_t_domain = loss_domain(domain_output, domain_label)

        # print(sum(out1))
        # print(target.detach())
        total_domain_loss = err_s_domain + err_t_domain + err_s_un_domain
        classification_loss = criterion(out1, target.detach().squeeze())
        loss = classification_loss + total_domain_loss
        loss.backward(retain_graph=True)
        optimizer_g.step()
        optimizer_f.step()
        zero_grad_all()
        if not args.method == 'S+T':
            output, dom_out = G(im_data_su, lamda = args.lamda)
            if args.method == 'ENT':
                loss_t = entropy(F1, output, args.lamda)
                loss_t.backward()
                optimizer_f.step()
                optimizer_g.step()
            elif args.method == 'MME':
                loss_t = adentropy(F1, output, args.lamda)
                loss_t.backward()
                optimizer_f.step()
                optimizer_g.step()
            else:
                raise ValueError('Method cannot be recognized.')
            log_train = 'Train Ep: {} lr{} \t ' \
                        'Loss Classification: {:.6f} Domain Classifier Loss {:.6f} Loss T {:.6f} ' \
                        'Method {}\n'.format(step, lr, classification_loss.data, total_domain_loss.data,
                                             -loss_t.data, args.method)
        else:
            log_train = 'S {} T {} Train Ep: {} lr{} \t ' \
                        'Loss Classification: {:.6f} Method {}\n'.\
                format(args.source, args.target,
                       step, lr, loss.data,
                       args.method)
        G.zero_grad()
        F1.zero_grad()
        zero_grad_all()
        if step % args.log_interval == 0:
            print(log_train)
        if step % args.save_interval == 0 and step > 0:
            print("eval on source unlabeled data")
            loss_source_test, acc_source_test, per_class_acc_s = test(source_loader_test)
            print("eval on target test and val respectively")
            loss_test, acc_test, per_class_acc_t = test(target_loader_test)
            loss_val, acc_val, _ = test(target_loader_val)
            G.train()
            F1.train()
            #this if block keep track of best acc on source
            if acc_source_test>=best_acc_source:
                best_acc_source = acc_source_test

            if acc_val >= best_acc:
                best_acc = acc_val
                best_acc_test = acc_test
                best_acc_source = acc_source_test
                counter = 0
            else:
                counter += 1

            print('best acc test source %f ,best acc test target %f ,current_ep acc val %f' % (best_acc_source, best_acc_test,
                                                                                       acc_val))
            print('record %s' % record_file)
            with open(record_file, 'a') as f:
                f.write('step %d best_acc_source_test %f best_acc_test_target %f final_ep_acc_val %f \n' % (step,
                                                         best_acc_source,
                                                         best_acc_test,
                                                         acc_val))
            if args.early:
                if counter > args.patience:
                    with open(final_record_file, 'a') as f:
                        f.write("Final accuracy for source: %s, target: %s is \n" % (source_domain, target_domain))
                        f.write("Test acc on unlabeled_source= {:.4f} and per_class_acc are {} \n".format(acc_source_test, per_class_acc_s))
                        f.write("Test acc on target= {:.4f} and per_class_acc are {} \n".format(acc_test, per_class_acc_t))
                        f.write('\n')
                    break

            G.train()
            F1.train()
            if args.save_check:
                print('saving model')
                torch.save(G.state_dict(),
                           os.path.join(args.checkpath,
                                        "G_iter_model_{}_{}_"
                                        "to_{}_step_{}.pth.tar".
                                        format(args.method, args.source,
                                               args.target, step)))
                torch.save(F1.state_dict(),
                           os.path.join(args.checkpath,
                                        "F1_iter_model_{}_{}_"
                                        "to_{}_step_{}.pth.tar".
                                        format(args.method, args.source,
                                               args.target, step)))


def test(loader):
    G.eval()
    F1.eval()
    test_loss = 0
    correct = 0
    size = 0
    num_class = len(class_list)
    output_all = np.zeros((0, num_class))
    criterion = nn.CrossEntropyLoss().cuda()
    confusion_matrix = torch.zeros(num_class, num_class)
    with torch.no_grad():
        for batch_idx, data_t in enumerate(loader):
            im_data_t.data.resize_(data_t[0].size()).copy_(data_t[0])
            gt_labels_t.data.resize_(data_t[1].size()).copy_(data_t[1])
            feat, dom_out = G(im_data_t, lamda = args.lamda)
            output1 = F1(feat)
            output_all = np.r_[output_all, output1.data.cpu().numpy()]
            size += im_data_t.size(0)
            pred1 = output1.data.max(1)[1]
            for t, p in zip(gt_labels_t.view(-1), pred1.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            correct += pred1.eq(gt_labels_t.data).cuda().sum()

            test_loss += criterion(output1, gt_labels_t.detach().squeeze()) / len(loader)
    per_class_acc = confusion_matrix.diag() / confusion_matrix.sum(1)
    print("Per class accuracy: {}".format(per_class_acc.detach()))
    print('\nTest set: Average loss: {:.4f}, '
          'Accuracy: {}/{} F1 ({:.0f}%)\n'.
          format(test_loss, correct, size,
                 100. * correct / size))
    return test_loss.data, 100. * float(correct) / size, per_class_acc


train()
print("Done")


