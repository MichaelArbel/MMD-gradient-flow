import torch 

import argparse

from trainer import Trainer

torch.backends.cudnn.benchmark = True

def make_flags(args,config_file):
    if config_file:
        config = yaml.load(open(config_file))
        dic = vars(args)
        all(map(dic.pop, config))
        dic.update(config)
    return args



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

# Optimizer parameters 
parser.add_argument('--lr', default=.1, type=float, help='learning rate')
parser.add_argument('--batch_size', default = 100 ,type= int,  help='batch size')
parser.add_argument('--total_epochs', default=10000, type=int, help='total number of epochs')
parser.add_argument('--optimizer',  default = 'SGD' ,type= str,   help='Optimizer')
parser.add_argument('--use_scheduler',   action='store_true',  help=' By default uses the ReduceLROnPlateau scheduler ')

# Loss paramters
parser.add_argument('--loss', default = 'mmd_noise_injection',type= str,  help='loss to optimize: mmd_noise_injection, mmd_diffusion, sobolev')
parser.add_argument('--with_noise',   default = True ,type= bool, help='to use noise injection set to true')
parser.add_argument('--noise_level', default = 1. ,type= float,  help=' variance of the injected noise ')
parser.add_argument('--noise_decay_freq', default = 1000 ,type= int,  help='decays the variance of the injected every 1000 epochs by a factor "noise_decay"')
parser.add_argument('--noise_decay', default = 0.5 ,type= float,  help='factor for decreasing the variance of the injected noise')

# Hardware parameters 
parser.add_argument('--device', default = 0 ,type= int,  help='gpu device, set -1 for cpu')
parser.add_argument('--dtype', default = 'float32' ,type= str,  help='precision: single: float32 or double: float64')

# Reproducibility parameters 
parser.add_argument('--seed', default = 1 ,type= int,  help='seed for the random number generator on pytorch')
parser.add_argument('--log_dir', default = '',type= str,  help='log directory ')
parser.add_argument('--log_name', default = 'mmd',type= str,  help='log name')
parser.add_argument('--log_in_file', action='store_true',  help='to log output on a file')

# Network parameters
parser.add_argument('--bias',   action='store_true',  help='ste to include bias in the network parameters')
parser.add_argument('--teacher_net', default = 'OneHidden' ,type= str,   help='teacher network')
parser.add_argument('--student_net', default = 'NoisyOneHidden' ,type= str,   help='student network')
parser.add_argument('--d_int', default = 50 ,type= int,  help='dim input data')
parser.add_argument('--d_out', default = 1 ,type= int,  help='dim out feature')
parser.add_argument('--H', default = 3  ,type= int,  help='num of hidden layers in the teacher network')
parser.add_argument('--num_particles', default = 1000 ,type= int,  help='num_particles*H = number of hidden units in the student network ')

# Initialization parameters
parser.add_argument('--mean_student', default = 0.001  ,type= float,  help='mean initial value for the student weights')
parser.add_argument('--std_student', default = 1.  ,type= float,  help='std initial value for the student weights')
parser.add_argument('--mean_teacher', default = 0.  ,type= float,  help='mean initial value for the teacher weights')
parser.add_argument('--std_teacher', default = 1.  ,type= float,  help='std initial value for the teacher weights')

# Data parameters 
parser.add_argument('--input_data', default = 'Spherical' ,type= str,   help='input data distribution')
parser.add_argument('--N_train', default = 1000 ,type= int,  help='num samples for training')
parser.add_argument('--N_valid', default = 1000 ,type= int,  help='num samples for validation')

parser.add_argument('--config',  default = '' ,type= str,     help='config file for non default parameters')

args = parser.parse_args()
args = make_flags(args,args.config)


exp = Trainer(args)
exp.train()








