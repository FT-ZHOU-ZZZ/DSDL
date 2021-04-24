import argparse
from engine import *
from models import *
from coco import *
from util import *
from loss import MyLoss

parser = argparse.ArgumentParser(description='WILDCAT Training')
parser.add_argument('data', metavar='DIR', default='', type=str,
                    help='path to dataset (e.g. data/')
parser.add_argument('--image-size', '-i', default=448, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default=[40], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--device_ids', default=[0], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--lambd', default=0.01, type=float,
                    help='the parameter for semantic regression -> (DtD+lambda*I)DtF')
parser.add_argument('--beta', default=0.01, type=float,
                    help='the parameter for feature restructure -> beta * (||feature - score*semantic|| + lambda*||score||)')


def main_coco():
    global args, best_prec1, use_gpu
    args = parser.parse_args()
    print('#############################################################################################')
    print('score = (D.transpose()*D + lambda*I).inverse()*D.transpose()*F')
    print('loss = (cross_entropy(score, truth)+ beta*(||feature - score*semantic||+lambda*||score||))/cosine(S,S_hat) ')
    print('the value of lambda is %f' % args.lambd)
    print('the value of beta is %f' % args.beta)
    print('#############################################################################################')

    use_gpu = torch.cuda.is_available()

    train_dataset = COCO2014(args.data, phase='train',
                             inp_name='data/coco/coco_glove_word2vec.pkl')
    val_dataset = COCO2014(args.data, phase='val', inp_name='data/coco/coco_glove_word2vec.pkl')
    num_classes = 80

    # load model
    model = load_model(num_classes=num_classes, alpha=args.lambd)

    # define loss function (criterion)
    criterion = MyLoss(args.lambd, args.beta)

    # define optimizer
    optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    state = {'batch_size': args.batch_size, 'image_size': args.image_size, 'max_epochs': args.epochs,
             'evaluate': args.evaluate, 'resume': args.resume, 'num_classes': num_classes}
    state['difficult_examples'] = True
    state['save_model_path'] = 'checkpoint/coco/'
    state['workers'] = args.workers
    state['epoch_step'] = args.epoch_step
    state['lr'] = args.lr
    state['device_ids'] = args.device_ids
    if args.evaluate:
        state['evaluate'] = True
    engine = DSDLMultiLabelMAPEngine(state)
    engine.learning(model, criterion, train_dataset, val_dataset, optimizer)


if __name__ == '__main__':
    main_coco()
