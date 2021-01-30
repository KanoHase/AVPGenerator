import argparse
import torchtext
from implementations.data_utils import prepare_binary, motif_restriction
import random
import gensim
from gensim.models import Word2Vec, KeyedVectors
from torchtext.vocab import Vectors
from transformer_model import *
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=500, help="number of epochs of training")
parser.add_argument("--vec_epoch", type=int, default=100, help="number of epochs of training vector")
parser.add_argument("--batch", type=int, default=64, help="number of batch size")
parser.add_argument("--lr", type=int, default=2e-5, help="learning rate")
parser.add_argument("--feature_size", type=int, default=10, help="learning rate")
parser.add_argument("--window", type=int, default=2, help="learning rate")
parser.add_argument("--figure_dir", type=str, default="./figures/", help="directory name to put figures")
parser.add_argument("--neg", type=bool, default=True, help="choose whether or not you want to include negative dataset")
parser.add_argument("--motif", type=bool, default=True, help="choose whether or not you want to include motif restriction")

opt = parser.parse_args()
data_dir = "./data/"
tsv_data_file = "amino_transformer_posneg.tsv"
vec_data_file = "amino_word2vec_vectors.vec"
neg = opt.neg
motif = opt.motif
feature_size = opt.feature_size
window = opt.window
num_epochs = opt.epoch
vec_epochs = opt.vec_epoch
batch_size = opt.batch
learning_rate = opt.lr

seq_arr, label_nparr, max_len = prepare_binary(neg = neg) # seq_arr: array that splits amino letters, max_len: 46
#print(seq_arr)

if motif == True:
    seq_arr, motif_list = motif_restriction(seq_arr)

def return_list(seq):
    '''
    returns a list that splits amino letters
    '''

    seq_list = list(seq)

    return seq_list

def make_tsv(seq_arr, label_nparr, tsv_data_file):
    '''
    make tsv file for transformer classification
    '''

    joined_seq_arr = ["".join(s) for s in seq_arr]

    with open(data_dir + tsv_data_file, "w") as f:
        for i, j_seq in enumerate(joined_seq_arr):
            
            tmp_list = [j_seq, "\t", str(label_nparr[i]), "\n"] 
            f.write("".join(tmp_list))

def make_vectors(seq_arr):
    '''
    returns vectors regarding seq_arr
    '''

    model = Word2Vec(seq_arr,size=feature_size,window=window) # able to identify the most similar amino to "D" by "model.wv.most_similar("D")", able to list amino letters up by "model.wv.index2entity"
    model.train(seq_arr,total_examples=len(seq_arr),epochs=vec_epochs)
    # model.save()
    model.wv.save_word2vec_format(data_dir + vec_data_file) # save vector file
    amino_vec = Vectors(data_dir + vec_data_file) # amino_vec: expresses 20 aminos by 10 dimension (somehow 0 only...)
    print("=========",len(amino_vec))
       
    return amino_vec

def build_text_vocab():
    '''
    builds vocab
    returns TEXT, LABEL and batch object regarding tsv file
    '''

    TEXT = torchtext.data.Field(sequential=True, tokenize=return_list, use_vocab=True,
                                lower=False, include_lengths=True, batch_first=True, fix_length=max_len, init_token="<cls>", eos_token="<eos>") # just translates all text (like splitting or padding)
    LABEL = torchtext.data.Field(sequential=False, use_vocab=False) 

    ds = torchtext.data.TabularDataset(
        path=data_dir+tsv_data_file, format='tsv',
        fields=[('Text', TEXT), ('Label', LABEL)]) # take data from tsv file, no shuffle, ds[0}:{'Text': ['D', 'L', ... 'D'], 'Label': '1'}

    train_ds, val_ds = ds.split(split_ratio=0.8, random_state=random.seed(1234)) # shuffled, train's len: 866, val's len: 216, ds[0}:{'Text': ['C', 'L', ... 'G'], 'Label': '0'}

    amino_vec = make_vectors(seq_arr) # load vectors

    TEXT.build_vocab(train_ds, vectors=amino_vec, min_freq=1) # make vectored vocab, TEXT.vocab.vectors.shape: (27, 10), TEXT.vocab.stoi: {'<unk>': 0, '<pad>': 1, 'L': 2, 'A': 3, ...})

    train_dl = torchtext.data.Iterator(train_ds, batch_size=batch_size, train=True) # divide into batches
    val_dl = torchtext.data.Iterator(val_ds, batch_size=batch_size, train=False, sort=False)

    '''
    batch = next(iter(train_dl)) 
    batch: dic(Text, Label)
    batch.Text[0]:[[9, 19, ... 1, 1, 1]
    batch.Text[0].shape: (64(batch_size), 46(max_len))
    batch.Label: 1 or 0, has len of batch_size
    batch.Text: (tensor([64, 46]), tensor([64]))
    '''

    return TEXT, LABEL, train_dl, val_dl

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        # Liner層の初期化
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print('-----start-------')
    net.to(device)

    torch.backends.cudnn.benchmark = True # ネットワークがある程度固定であれば、高速化させる

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train() # mode: train
            else:
                net.eval() # mode: evaluate

            epoch_loss = 0.0  # epoch's total loss
            epoch_corrects = 0  # epoch's total correct answers

            for batch in (dataloaders_dict[phase]):
                inputs = batch.Text[0].to(device)
                labels = batch.Label.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    input_pad = 1  # ID of '<pad>' is 1
                    input_mask = (inputs != input_pad) #if <pad> True else False?

                    outputs, _, _ = net(inputs, input_mask)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)  # predict label

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)  # calculate total loss
                    epoch_corrects += torch.sum(preds == labels.data) # calculate total correct answers

            # epochごとのlossと正解率
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

            print('Epoch {}/{} | {:^5} |  Loss: {:.4f} Acc: {:.4f}'.format(epoch+1, num_epochs,
                                                                           phase, epoch_loss, epoch_acc))

    return net


def transformer_classification():
    make_tsv(seq_arr, label_nparr, tsv_data_file)

    TEXT, LABEL, train_dl, val_dl = build_text_vocab()

    net = TransformerClassification(TEXT.vocab.vectors, feature_size, max_len, output_dim=2)
    net.train() 

    net.net3_1.apply(weights_init) # initialize TransformerBlock module
    net.net3_2.apply(weights_init)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    dataloaders_dict = {"train": train_dl, "val": val_dl}
    
    net_trained = train_model(net, dataloaders_dict, criterion, optimizer, num_epochs)


transformer_classification()