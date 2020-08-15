#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="0"


# In[ ]:


import torch
import torch.nn as nn
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule, XLMRobertaTokenizer, XLMRobertaModel, XLMRobertaConfig
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import re
from collections import defaultdict
from collections import Counter
from sklearn.model_selection import KFold


# In[ ]:


MAX_LEN = 500
TRAIN_BATCH_SIZE = 12
VALID_BATCH_SIZE = 8
EPOCHS = 100
BERT_MODEL = 'xlm-roberta-base'
TOKENIZER = transformers.XLMRobertaTokenizer.from_pretrained(BERT_MODEL)
#TRAIN_FILE = "./data/train_data_annotated_BIOES_v3.txt"
#VALID_FILE = "./data/e"
TRAIN_FILE = "./data/abcd"
VALID_FILE = "./data/e"
#TRAIN_FILE = "/home/ktlim/code/TTtagger/corpus/ud-treebanks-v2.3/UD_English-EWT/en_ewt-ud-train.conllu"
#VALID_FILE = "/home/ktlim/code/TTtagger/corpus/ud-treebanks-v2.3/UD_English-EWT/en_ewt-ud-test.conllu"


# In[ ]:


DEVICE=0
ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = range(10)


# In[ ]:


def normalize(word):
    return re.sub(r"\d", "0", word).lower()


def strong_normalize(word):
    w = ftfy.fix_text(word.lower())
    w = re.sub(r".+@.+", "*EMAIL*", w)
    w = re.sub(r"@\w+", "*AT*", w)
    w = re.sub(r"(https?://|www\.).*", "*url*", w)
    w = re.sub(r"([^\d])\1{2,}", r"\1\1", w)
    w = re.sub(r"([^\d][^\d])\1{2,}", r"\1\1", w)
    w = re.sub(r"``", '"', w)
    w = re.sub(r"''", '"', w)
    w = re.sub(r"\d", "0", w)
    return w


def buildVocab(graphs, cutoff=1):
    wordsCount = Counter()
    charsCount = Counter()
    uposCount = Counter()
    xposCount = Counter()
    relCount = Counter()
    featCount = Counter()
    langCount = Counter()

    for graph in graphs:
        wordsCount.update([node.norm for node in graph.nodes[1:]])
        for node in graph.nodes[1:]:
            charsCount.update(list(node.word))
            featCount.update(node.feats_set)
            #  charsCount.update(list(node.norm))
        uposCount.update([node.upos for node in graph.nodes[1:]])
        xposCount.update([node.xupos for node in graph.nodes[1:]])
        relCount.update([rel for rel in graph.rels[1:]])
        langCount.update([node.lang for node in graph.nodes[1:]])
        

    wordsCount = Counter({w: i for w, i in wordsCount.items() if i >= cutoff})
    print("Vocab containing {} words".format(len(wordsCount)))
    print("Charset containing {} chars".format(len(charsCount)))
    print("UPOS containing {} tags".format(len(uposCount)), uposCount)
    #print("XPOS containing {} tags".format(len(xposCount)), xposCount)
    print("Rels containing {} tags".format(len(relCount)), relCount)
    print("Feats containing {} tags".format(len(featCount)), featCount)
    print("lang containing {} tags".format(len(langCount)), langCount)

    ret = {
        "vocab": list(wordsCount.keys()),
        "wordfreq": wordsCount,
        "charset": list(charsCount.keys()),
        "charfreq": charsCount,
        "upos": list(uposCount.keys()),
        "xpos": list(xposCount.keys()),
        "rels": list(relCount.keys()),
        "feats": list(featCount.keys()),
        "lang": list(langCount.keys()),
    }

    return ret

def shuffled_stream(data):
    len_data = len(data)
    while True:
        for d in random.sample(data, len_data):
            yield d

def shuffled_balanced_stream(data):
    for ds in zip(*[shuffled_stream(s) for s in data]):
        ds = list(ds)
        random.shuffle(ds)
        for d in ds:
            yield d
            
            
def parse_dict(features):
    if features is None or features == "_":
        return {}

    ret = {}
    lst = features.split("|")
    for l in lst:
        k, v = l.split("=")
        ret[k] = v
    return ret


def parse_features(features):
    if features is None or features == "_":
        return set()

    return features.lower().split("|")


class Word:

    def __init__(self, word, upos, lemma=None, xpos=None, feats=None, misc=None, lang=None):
        self.word = word
        self.norm = normalize(word) #strong_normalize(word)
        self.lemma = lemma if lemma else "_"
        self.upos = upos
        self.xpos = xpos if xpos else "_"
        self.xupos = self.upos + "|" + self.xpos
        self.feats = feats if feats else "_"
        self.feats_set = parse_features(self.feats)
        self.misc = misc if misc else "_"
        self.lang = lang if lang else "_"

    def cleaned(self):
        return Word(self.word, "_")

    def clone(self):
        return Word(self.word, self.upos, self.lemma, self.xpos, self.feats, self.misc)

    def __repr__(self):
        return "{}_{}".format(self.word, self.upos)


class DependencyGraph(object):

    def __init__(self, words, tokens=None):
        #  Token is a tuple (start, end, form)
        if tokens is None:
            tokens = []
        self.nodes = np.array([Word("*root*", "*root*")] + list(words))
        self.tokens = tokens
        self.heads = np.array([-1] * len(self.nodes))
        self.rels = np.array(["_"] * len(self.nodes), dtype=object)

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.nodes = self.nodes
        result.tokens = self.tokens
        result.heads = self.heads.copy()
        result.rels = self.rels.copy()
        return result

    def cleaned(self, node_level=True):
        if node_level:
            return DependencyGraph([node.cleaned() for node in self.nodes[1:]], self.tokens)
        else:
            return DependencyGraph([node.clone() for node in self.nodes[1:]], self.tokens)

    def attach(self, head, tail, rel):
        self.heads[tail] = head
        self.rels[tail] = rel

    def __repr__(self):
        return "\n".join(["{} ->({})  {} ({})".format(str(self.nodes[i]), self.rels[i], self.heads[i], self.nodes[self.heads[i]]) for i in range(len(self.nodes))])


def read_conll(filename, lang_code=None):
    
    print("read_conll with", lang_code)
    def get_word(columns):
        return Word(columns[FORM], columns[UPOS], lemma=columns[LEMMA], xpos=columns[XPOS], feats=columns[FEATS], misc=columns[MISC], lang=lang_code)

    def get_graph(graphs, words, tokens, edges):
        graph = DependencyGraph(words, tokens)
        for (h, d, r) in edges:
            graph.attach(h, d, r)
        graphs.append(graph)

    file = open(filename, "r", encoding="UTF-8")

    graphs = []
    words = []
    tokens = []
    edges = []

    num_sent = 0
    sentence_start = False
    while True:
        line = file.readline()
        if not line:
            if len(words) > 0:
                get_graph(graphs, words, tokens, edges)
                words, tokens, edges = [], [], []
            break
        line = line.rstrip("\r\n")

        # Handle sentence start boundaries
        if not sentence_start:
            # Skip comments
            if line.startswith("#"):
                continue
            # Start a new sentence
            sentence_start = True
        if not line:
            sentence_start = False
            if len(words) > 0:
                if (len(words) < 250):
                    get_graph(graphs, words, tokens, edges)
                words, tokens, edges = [], [], []
                num_sent += 1
            continue

        # Read next token/word
        columns = line.split("\t")

        # Skip empty nodes
        if "." in columns[ID]:
            continue

        # Handle multi-word tokens to save word(s)
        if "-" in columns[ID]:
            start, end = map(int, columns[ID].split("-"))
            tokens.append((start, end + 1, columns[FORM]))

            for _ in range(start, end + 1):
                word_line = file.readline().rstrip("\r\n")
                word_columns = word_line.split("\t")
                words.append(get_word(word_columns))
                if word_columns[HEAD].isdigit():
                    head = int(word_columns[HEAD])
                else:
                    head = -1
                edges.append((head, int(word_columns[ID]), word_columns[DEPREL].split(":")[0]))
        # Basic tokens/words
        else:
            words.append(get_word(columns))
            if columns[HEAD].isdigit():
                head = int(columns[HEAD])
            else:
                head = -1
            edges.append((head, int(columns[ID]), columns[DEPREL].split(":")[0]))

    file.close()

    return graphs


# In[ ]:


# 2. Data Loader
class CoNLLDataset:
    def __init__(self, graphs, tokenizer, max_len, fullvocab=None):
        self.conll_graphs = graphs
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        self._fullvocab = fullvocab if fullvocab else buildVocab(self.conll_graphs, cutoff=1)
            
        self._upos = {p: i for i, p in enumerate(self._fullvocab["upos"])}
        self._iupos = self._fullvocab["upos"]
        self._xpos = {p: i for i, p in enumerate(self._fullvocab["xpos"])}
        self._ixpos = self._fullvocab["xpos"]
        self._vocab = {w: i+3 for i, w in enumerate(self._fullvocab["vocab"])}
        self._wordfreq = self._fullvocab["wordfreq"]
        self._charset = {c: i+3 for i, c in enumerate(self._fullvocab["charset"])}
        self._charfreq = self._fullvocab["charfreq"]
        self._rels = {r: i for i, r in enumerate(self._fullvocab["rels"])}
        self._irels = self._fullvocab["rels"]
        self._feats = {f: i for i, f in enumerate(self._fullvocab["feats"])}
        self._langs = {r: i+2 for i, r in enumerate(self._fullvocab["lang"])}
        self._ilangs = self._fullvocab["lang"]
        
        #self._posRels = {r: i for i, r in enumerate(self._fullvocab["posRel"])}
        #self._iposRels = self._fullvocab["posRel"]
        
        self._vocab['*pad*'] = 0
        self._charset['*pad*'] = 0
        self._langs['*pad*'] = 0
        
        self._vocab['*root*'] = 1
        self._charset['*whitespace*'] = 1
        
        self._vocab['*unknown*'] = 2
        self._charset['*unknown*'] = 2
        
        
    
    def __len__(self):
        return len(self.conll_graphs)
        
        
    def __getitem__(self, item):
        
        graph = self.conll_graphs[item]
        word_list = [node.word for node in graph.nodes]
        upos_list = [node.upos for node in graph.nodes]
        feat_list = [node.feats for node in graph.nodes]
        
        encoded = self.tokenizer.encode_plus(' '.join(word_list[1:]),
                                             None,
                                             add_special_tokens=True,
                                             max_length = self.max_len,
                                             truncation=True,
                                             pad_to_max_length = True)
        
        ids, mask = encoded['input_ids'], encoded['attention_mask']
        
        bpe_head_mask = [0]; upos_ids = [-1]; feat_ids = [-1] # --> CLS token
        
        for word, upos, feat in zip(word_list[1:], upos_list[1:], feat_list[1:]):
            bpe_len = len(self.tokenizer.tokenize(word))
            head_mask = [1] + [0]*(bpe_len-1)
            bpe_head_mask.extend(head_mask)
            upos_mask = [self._upos.get(upos)] + [-1]*(bpe_len-1)
            upos_ids.extend(upos_mask)
            feat_mask = [self._feats.get(feat.lower(), 2)] + [-1]*(bpe_len-1)
            feat_ids.extend(feat_mask)
            
            #print("head_mask", head_mask)
        
        bpe_head_mask.append(0); upos_ids.append(-1); feat_ids.append(-1) # --> END token
        bpe_head_mask.extend([0] * (self.max_len - len(bpe_head_mask))) ## --> padding by max_len
        upos_ids.extend([-1] * (self.max_len - len(upos_ids))) ## --> padding by max_len
        feat_ids.extend([-1] * (self.max_len - len(feat_ids))) ## --> padding by max_len
        
        return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'bpe_head_mask': torch.tensor(bpe_head_mask, dtype=torch.long),
                'upos_ids': torch.tensor(upos_ids, dtype=torch.long),
                'feat_ids': torch.tensor(feat_ids, dtype=torch.long)
               }
    
    

  


# In[ ]:


def f1_score(total_pred, total_targ, noNER_idx):
    
    p = 0 # (retrived SB and real SB) / retrived SB  # The percentage of (the number of correct predictions) / (the number of predction that system predicts as B-SENT)
    r = 0
    f1= 0
    
    np_total_pred = np.array(total_pred)
    np_total_tag = np.array(total_targ)
    
    #Get noPad
    incidence_nopad = np.where(np_total_tag != -1) ## eliminate paddings
    np_total_pred_nopad = np_total_pred[incidence_nopad]
    np_total_tag_nopad = np_total_tag[incidence_nopad]
    
    
    #precision
    incidence_nopad_sb = np.where(np_total_pred_nopad != noNER_idx)
    np_total_pred_nopad_sb = np_total_pred_nopad[incidence_nopad_sb]
    np_total_tag_nopad_sb = np_total_tag_nopad[incidence_nopad_sb]
    
    count_active_tokens_p = len(np_total_pred_nopad_sb)
    count_correct_p = np.count_nonzero((np_total_pred_nopad_sb==np_total_tag_nopad_sb) == True)
    
    '''
    np_total_pred_incid = np_total_pred[incidence_p]
    print("np_total_pred_incid", np_total_pred_incid)
    ids_sb_pred_p = np.where(np_total_pred_incid==1)
    np_total_pred_p = np_total_pred_incid[ids_sb_pred_p]
    np_total_tag_p = np_total_tag[ids_sb_pred_p]
    
    print("ids_sb_pred_p", ids_sb_pred_p)
    print("np_total_pred_p", np_total_pred_p)
    print("np_total_tag_p", np_total_tag_p)
    
    count_active_tokens_p = len(np_total_pred_p)
    count_correct_p = np.count_nonzero((np_total_pred_p==np_total_tag_p) == True)
    '''
    
    print("count_correct_p", count_correct_p)
    print("count_active_tokens_p", count_active_tokens_p)
    
    p = count_correct_p/count_active_tokens_p
    print("precision:", p)

    
    #recall
    ids_sb_pred_r = np.where(np_total_tag_nopad != noNER_idx)
    np_total_pred_r = np_total_pred_nopad[ids_sb_pred_r]
    np_total_tag_r = np_total_tag_nopad[ids_sb_pred_r]
    
    #print("ids_sb_pred_r", ids_sb_pred_r)
    #print("np_total_pred_r", np_total_pred_r)
    #print("np_total_tag_r", np_total_tag_r)
    
    count_active_tokens_r = len(np_total_pred_r)
    count_correct_r = np.count_nonzero((np_total_pred_r==np_total_tag_r) == True)
    
    print("count_active_tokens_r", count_active_tokens_r)
    print("count_correct_r", count_correct_r)
    
    r = count_correct_r/count_active_tokens_r
    print("recall:", r)
    
    
    #F1
    f1 = 2*(p*r) / (p+r)
    print("F1:", f1)
    
    #count_active_tokens_recall = np.count_nonzero(np.array(total_targ) > -1)
    #print("count_active_tokens_recall", count_active_tokens_recall)
    #count_active_tokens_precision = np.count_nonzero(np.array(total_targ) > -1)
    
    #count_correct = np.count_nonzero((np.array(total_pred)==np.array(total_targ)) == True)
    #print("count_correct",count_correct)
    #print("ACCURACY:", count_correct/count_active_tokens)
    


# In[ ]:


class XLMRobertaEncoder(nn.Module):
    def __init__(self, num_upos, num_feat):
        super(XLMRobertaEncoder, self).__init__()
        self.xlm_roberta = transformers.XLMRobertaModel.from_pretrained('xlm-roberta-base')
        self.dropout = nn.Dropout(0.33)
        self.linear = nn.Linear(768, num_upos)
        
        self.f_dropout = nn.Dropout(0.33)
        self.f_linear = nn.Linear(768, num_feat)
            
    def forward(self, ids, mask):
        o1, o2 = self.xlm_roberta(ids, mask)
        
        #apool = torch.mean(o1, 1)
        #mpool, _ = torch.max(o1, 1)
        #cat = torch.cat((apool, mpool), 1)
        #bo = self.dropout(cat)
        p_logits = self.linear(o1)        
        f_logits = self.f_linear(o1)   
        
        return p_logits, f_logits
        


# In[ ]:


#train_graphs = read_conll(TRAIN_FILE, 'ko')
#cv = KFold(n_splits=5, random_state=1, shuffle=False)
#for t,v in cv.split(train_graphs):
#    train_graph=train_graphs[t]         # Train Set
#    valid_graph=train_graphs[v]         # Validation Set


# In[ ]:


train_graphs = read_conll(TRAIN_FILE, 'ko')
train_dataset = CoNLLDataset(graphs=train_graphs, tokenizer=TOKENIZER, max_len=MAX_LEN)
train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=4, batch_size=TRAIN_BATCH_SIZE, shuffle=False)
valid_graphs = read_conll(VALID_FILE, 'ko')
valid_dataset = CoNLLDataset(graphs=valid_graphs, tokenizer=TOKENIZER, max_len=MAX_LEN, fullvocab=train_dataset._fullvocab)
valid_loader = torch.utils.data.DataLoader(valid_dataset, num_workers=4, batch_size=VALID_BATCH_SIZE, shuffle=False)


# In[ ]:


num_upos = len(train_dataset._upos)
num_feat = len(train_dataset._feats)
model = XLMRobertaEncoder(num_upos, num_feat)
model = nn.DataParallel(model)
model = model.cuda()


# In[ ]:


loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
lr = 0.000005
optimizer = AdamW(model.parameters(), lr=lr)


# In[ ]:


def train_loop_fn(train_loader, model, optimizer, DEVICE, scheduler=None):
    model.train()
    
    p_total_pred = []
    p_total_targ = []
    p_total_loss = []
    
    f_total_pred = []
    f_total_targ = []
    f_total_loss = []
    
    for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()
        
        p_logits, f_logits = model(batch['ids'].cuda(), batch['mask'].cuda())
        
        #UPOS
        b,s,l = p_logits.size()
        #print(p_logits.view(b*s,l), p_logits.view(b*s,l).size())
        #print(batch['upos_ids'].cuda().view(b*s), batch['upos_ids'].cuda().view(b*s).size())
        p_loss = loss_fn(p_logits.view(b*s,l), batch['upos_ids'].cuda().view(b*s))
        p_total_loss.append(p_loss.item())
        p_total_pred.extend(torch.argmax(p_logits.view(b*s,l), 1).cpu().tolist())
        p_total_targ.extend(batch['upos_ids'].cuda().view(b*s).cpu().tolist())
        
        #FEAT
        b,s,l = f_logits.size()
        f_loss = loss_fn(f_logits.view(b*s,l), batch['feat_ids'].cuda().view(b*s))
        f_total_loss.append(f_loss.item())
        f_total_pred.extend(torch.argmax(f_logits.view(b*s,l), 1).cpu().tolist())
        f_total_targ.extend(batch['feat_ids'].cuda().view(b*s).cpu().tolist())
        
        loss = p_loss+f_loss
        loss.backward()
        optimizer.step()
        
    count_active_tokens = np.count_nonzero(np.array(p_total_targ) > -1)
    count_correct = np.count_nonzero((np.array(p_total_pred)==np.array(p_total_targ)) == True)
    print("TRAINING POS ACCURACY:", count_correct/count_active_tokens)
    
    count_active_tokens = np.count_nonzero(np.array(f_total_targ) > -1)
    count_correct = np.count_nonzero((np.array(f_total_pred)==np.array(f_total_targ)) == True)
    f1_score(f_total_pred, f_total_targ, train_dataset._feats.get('o', 2))
    print("TRAINING FEAT ACCURACY:", count_correct/count_active_tokens)


# In[ ]:


def valid_loop_fn(dev_loader, model, DEVICE):
    model.eval()
    
    p_total_pred = []
    p_total_targ = []
    p_total_loss = []
    
    f_total_pred = []
    f_total_targ = []
    f_total_loss = []
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dev_loader), total=len(dev_loader)):

            p_logits, f_logits = model(batch['ids'].cuda(), batch['mask'].cuda())

            #UPOS
            b,s,l = p_logits.size()
            p_loss = loss_fn(p_logits.view(b*s,l), batch['upos_ids'].cuda().view(b*s))
            p_total_loss.append(p_loss.item())
            p_total_pred.extend(torch.argmax(p_logits.view(b*s,l), 1).cpu().tolist())
            p_total_targ.extend(batch['upos_ids'].cuda().view(b*s).cpu().tolist())

            #FEAT
            b,s,l = f_logits.size()
            f_loss = loss_fn(f_logits.view(b*s,l), batch['feat_ids'].cuda().view(b*s))
            f_total_loss.append(f_loss.item())
            f_total_pred.extend(torch.argmax(f_logits.view(b*s,l), 1).cpu().tolist())
            f_total_targ.extend(batch['feat_ids'].cuda().view(b*s).cpu().tolist())

            loss = p_loss+f_loss
        
    count_active_tokens = np.count_nonzero(np.array(p_total_targ) > -1)
    count_correct = np.count_nonzero((np.array(p_total_pred)==np.array(p_total_targ)) == True)
    print("VALIDATION POS ACCURACY:", count_correct/count_active_tokens)
    
    count_active_tokens = np.count_nonzero(np.array(f_total_targ) > -1)
    count_correct = np.count_nonzero((np.array(f_total_pred)==np.array(f_total_targ)) == True)
    f1_score(f_total_pred, f_total_targ, train_dataset._feats.get('o', 2))
    print("VALIDATION FEAT ACCURACY:", count_correct/count_active_tokens)


# In[ ]:


print(train_dataset._feats.get('o'))


# In[ ]:


for idx in range(EPOCHS):
    train_loop_fn(train_loader, model, optimizer, DEVICE)
    valid_loop_fn(valid_loader, model, DEVICE)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




