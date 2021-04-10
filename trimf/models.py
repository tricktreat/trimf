import torch
from torch.nn import functional as F
from torch import nn as nn
from typing import List
from torch_geometric.data import Data, Batch
from torch_geometric.nn import RGCNConv, GATConv
from torch_geometric import utils
from transformers import BertConfig
from transformers import BertModel
from transformers import BertPreTrainedModel
import random
from itertools import combinations, permutations
from functools import reduce
from trimf import sampling
from trimf import util
import torchsnooper

def get_token(h: torch.tensor, x: torch.tensor, token: int):
    emb_size = h.shape[-1]

    token_h = h.view(-1, emb_size)
    flat = x.contiguous().view(-1)

    token_h = token_h[flat == token, :]

    return token_h


class TriMF(BertPreTrainedModel):

    def __init__(self, config: BertConfig, cls_token: int, relation_types: int, entity_types: int,
                 size_embedding: int, prop_drop: float, freeze_transformer: bool, device: torch.device, syn_graph= True, sema_graph=True, fusion_rgcn = True, tw_grad_flow_token=True,tw_grad_flow_subword=True, tw_rel_atten_token=True, tw_ent_atten_token=True, tw_rel_atten_subword=False, tw_ent_atten_subword=False, trigger_attn = True,trigger_grad_flow=False, full_graph_retain_rate=0.8,dt_graph_retain_rate=0.8,  max_pairs: int = 100, split_epoch=18):
        super(TriMF, self).__init__(config)

        if fusion_rgcn:
            assert syn_graph and sema_graph
        
        # if tw_atten_token:
        #     assert tw_rel_atten_token and tw_ent_atten_token

        # if tw_atten_subword:
        #     assert tw_rel_atten_subword and tw_ent_atten_subword

        self._cls_token = cls_token
        self._relation_types = relation_types
        self._entity_types = entity_types
        self._max_pairs = max_pairs
        self._device = device

        self._syn_graph = syn_graph
        self._sema_graph = sema_graph
        self._fusion_rgcn = fusion_rgcn
        self._tw_grad_flow_token = tw_grad_flow_token
        self._tw_grad_flow_subword = tw_grad_flow_subword
        # self._tw_atten_token=tw_atten_token
        self._tw_rel_atten_token = tw_rel_atten_token
        self._tw_ent_atten_token = tw_ent_atten_token
        # self._tw_atten_subword=tw_atten_subword
        self._tw_rel_atten_subword = tw_rel_atten_subword
        self._tw_ent_atten_subword = tw_ent_atten_subword
        self._trigger_attn = trigger_attn
        self._trigger_grad_flow = trigger_grad_flow

        self._full_graph_retain_rate = full_graph_retain_rate
        self._dt_graph_retain_rate = dt_graph_retain_rate

        self.bert = BertModel(config)
        
        # layers
        self.rel_linear = nn.Linear(config.hidden_size * 3 + size_embedding * 2, config.hidden_size, False)
        self.rel_classifier = nn.Linear(config.hidden_size, relation_types, False)
        self.entity_linear = nn.Linear(config.hidden_size * 2 + size_embedding, config.hidden_size, False)
        self.entity_classifier = nn.Linear(config.hidden_size, entity_types, False)
        self.size_embeddings = nn.Embedding(100, size_embedding)
        self.dropout = nn.Dropout(prop_drop)

        if self._tw_rel_atten_token:
            self.token_rel_attn_W = nn.Linear(config.hidden_size,config.hidden_size,False)
        if self._tw_ent_atten_token:
            self.token_ent_attn_W = nn.Linear(config.hidden_size,config.hidden_size,False)

        if self._tw_rel_atten_subword:
            self.subword_rel_attn_W = nn.Linear(config.hidden_size,config.hidden_size,False)
        if self._tw_ent_atten_subword:
            self.subword_ent_attn_W = nn.Linear(config.hidden_size,config.hidden_size,False)

        # if self._trigger_attn:
            # self.head_W=nn.Linear(config.hidden_size,config.hidden_size,False)
            # self.tail_W=nn.Linear(config.hidden_size,config.hidden_size,False)
            # self.rel_W=nn.Linear(config.hidden_size,config.hidden_size,False)

        if self._fusion_rgcn:
            self.node_W = nn.Linear(config.hidden_size,config.hidden_size,False)

        self.relu = nn.ReLU()

        self._split_epoch = split_epoch
        self.save_config = [tw_rel_atten_token,tw_ent_atten_token,tw_rel_atten_subword,tw_ent_atten_subword,trigger_attn]

        if self._syn_graph:
            self.conv1 = RGCNConv(in_channels = config.hidden_size, out_channels = config.hidden_size, num_relations = 46, num_bases= None, root_weight= True, bias = True)
            # self.conv2 = RGCNConv(in_channels = config.hidden_size, out_channels = config.hidden_size, num_relations = 46, num_bases= None, root_weight= True, bias = True)
        if self._sema_graph:
            self.gat = GATConv(in_channels = config.hidden_size, out_channels = config.hidden_size,bias = True)
            # self.conv3 = RGCNConv(in_channels = config.hidden_size, out_channels = config.hidden_size, num_relations = 1, num_bases= None, root_weight= True, bias = True)

        self.init_weights()

        if freeze_transformer:
            print("Freeze transformer weights")

            # freeze all transformer weights
            for param in self.bert.parameters():
                param.requires_grad = False

    def _common_forward(self, encodings: torch.tensor, context_masks: torch.tensor,token_m:torch.tensor,token_masks: torch.tensor, dt_data_list: List[Data], entity_masks: torch.tensor, entity_masks_token: torch.tensor, entity_sizes: torch.tensor):
        # sub-word
        # context_masks = context_masks.float()
        h = self.bert(input_ids = encodings, attention_mask = context_masks.float())[0]
        # import pdb; pdb.set_trace()
        if not self._tw_grad_flow_token:
            token_rel_w = self.rel_classifier.weight.detach()
            # token_rel_b=self.rel_classifier.bias.detach()
            token_ent_w = self.entity_classifier.weight[1:].detach()
            # token_ent_b=self.entity_classifier.bias[1:].detach()
        else:
            token_rel_w = self.rel_classifier.weight
            # token_rel_b=self.rel_classifier.bias
            token_ent_w = self.entity_classifier.weight[1:]
            # token_ent_b=self.entity_classifier.bias[1:]

        if not self._tw_grad_flow_subword:
            subword_rel_w = self.rel_classifier.weight.detach()
            # subword_rel_b=self.rel_classifier.bias.detach()
            subword_ent_w = self.entity_classifier.weight[1:].detach()
            # subword_ent_b=self.entity_classifier.bias[1:].detach()
        else:
            subword_rel_w = self.rel_classifier.weight
            # subword_rel_b=self.rel_classifier.bias
            subword_ent_w = self.entity_classifier.weight[1:]
            # subword_ent_b=self.entity_classifier.bias[1:]

        tw_attns_subword = [h]
        if self._tw_rel_atten_subword:
            # B #T 768  ----> B #T #Rel-1
            subword_rel_attention = F.linear(h,self.subword_rel_attn_W(subword_rel_w))
            # subword_rel_attention=self.relu(subword_rel_attention)
            # subword_rel_attention = F.softmax(subword_rel_attention, dim=1)  # todo mask
            # subword_rel_attention = util.masked_softmax(subword_rel_attention,context_masks.unsqueeze(-1).repeat(1,1,subword_rel_attention.size(-1)), dim=1)


            # B #T #Rel-1   ----> B #T 1
            subword_rel_attention = torch.sum(subword_rel_attention, dim=2, keepdim=True)
            h_rel_attend = torch.mul(subword_rel_attention, h)
            tw_attns_subword.append(h_rel_attend)

        if self._tw_ent_atten_subword:
            # B #T 768  ----> B #T #Entity-1
            subword_entity_attention = F.linear(h,self.subword_ent_attn_W(subword_ent_w))
            # subword_entity_attention = F.softmax(subword_entity_attention, dim=1)
            # subword_entity_attention=self.relu(subword_entity_attention)
            # subword_entity_attention = util.masked_softmax(subword_entity_attention, context_masks.unsqueeze(-1).repeat(1,1,subword_entity_attention.size(-1)), dim=1)


            # B #T #Entity-1   ----> B #T 1
            subword_entity_attention = torch.sum(subword_entity_attention, dim=2, keepdim=True)
            h_entity_attend = torch.mul(subword_entity_attention, h)
            tw_attns_subword.append(h_entity_attend)

        if len(tw_attns_subword):
            h=reduce(lambda x,y: x+y,tw_attns_subword)/len(tw_attns_subword)

        m = (token_masks.unsqueeze(-1) == 0).float() * (-1e30)
        token_spans_pool = m + h.unsqueeze(1).repeat(1, token_masks.shape[1], 1, 1)
        token_spans_pool = token_spans_pool.max(dim=2)[0]

        token_rel_attention = None
        tw_attns_token = [token_spans_pool]
        if self._tw_rel_atten_token:
            # B #T 768  ----> B #T #Rel-1
            token_rel_attention = F.linear(token_spans_pool,self.token_rel_attn_W(token_rel_w))
            # token_rel_attention=self.relu(token_rel_attention)
            # token_rel_attention = F.softmax(token_rel_attention, dim=1)  # todo mask
            token_rel_attention = util.masked_softmax(token_rel_attention,token_m.unsqueeze(-1).repeat(1,1,token_rel_attention.size(-1)), dim=1)


            # B #T #Rel-1   ----> B #T 1
            token_rel_attention = torch.sum(token_rel_attention, dim=2, keepdim=True)
            token_spans_pool_rel_attend = torch.mul(token_rel_attention, token_spans_pool)
            tw_attns_token.append(token_spans_pool_rel_attend)

        token_entity_attention=None
        if self._tw_ent_atten_token:
            # B #T 768  ----> B #T #Entity-1
            token_entity_attention = F.linear(token_spans_pool,self.token_ent_attn_W(token_ent_w))
            # token_entity_attention=self.relu(token_entity_attention)
            # token_entity_attention = F.softmax(token_entity_attention, dim=1)
            token_entity_attention = util.masked_softmax(token_entity_attention, token_m.unsqueeze(-1).repeat(1,1,token_entity_attention.size(-1)), dim=1)


            # B #T #Entity-1   ----> B #T 1
            token_entity_attention = torch.sum(token_entity_attention, dim=2, keepdim=True)
            token_spans_pool_entity_attend = torch.mul(token_entity_attention, token_spans_pool)
            tw_attns_token.append(token_spans_pool_entity_attend)

        if len(tw_attns_token):
            token_spans_pool=reduce(lambda x,y: x+y,tw_attns_token)/len(tw_attns_token)

        if not self.training:
            full_graph_retain_rate=1
            dt_graph_retain_rate=1
        else:
            full_graph_retain_rate=self._full_graph_retain_rate
            dt_graph_retain_rate=self._dt_graph_retain_rate
        
        dt_data_list_add_feature = []
        if self._sema_graph or self._syn_graph:
            # node features added
            for dt,token_span_pool in zip(dt_data_list,token_spans_pool):
                dt.x = token_span_pool[:dt.num_nodes]
                dt_data_list_add_feature.append(dt)
        hs=[]
        if self._syn_graph:
            # token_spans_pool dt
            # batched_dt = Batch.from_data_list(dt_data_list_add_feature)
            # batched_dt.to(self._device)
            # batched_dt.x = self.conv1(batched_dt.x, batched_dt.edge_index, batched_dt.edge_type)
            # updated_dt_data_list = batched_dt.to_data_list()
            # h_token_dt = util.padded_stack([dt_data["x"] for dt_data in updated_dt_data_list])
            # hs.append(h_token_dt)

            # token_spans_pool dt drop
            # dt_graph_drp_rate=0.8
            dt_data_list_drop=[]
            for dt in dt_data_list_add_feature: # or updated_dt_data_list
                num=dt.num_nodes
                if num<2:
                    dt_data_list_drop.append(dt)
                    continue
                edge_index,edge_type=zip(*random.sample(list(zip(dt.edge_index.t().tolist(),dt.edge_type.tolist())), max(1, round(dt.edge_type.size(0)* dt_graph_retain_rate))))
                edge_index = list(edge_index)
                edge_type = list(edge_type)
                for i in range(num):
                    if (i , i) not in edge_index:
                        edge_index.append((i, i))
                        edge_type.append(max(edge_type)+1)
                dt.edge_index=torch.tensor(edge_index).t()
                dt.edge_type=torch.tensor(edge_type)
                dt_data_list_drop.append(dt)
            batched_dt_drop = Batch.from_data_list(dt_data_list_drop)
            batched_dt_drop.to(self._device)
            batched_dt_drop.x = self.conv1(batched_dt_drop.x, batched_dt_drop.edge_index, batched_dt_drop.edge_type)
            updated_dt_data_list = batched_dt_drop.to_data_list()
            h_token_dt_drop = util.padded_stack([dt_data["x"] for dt_data in updated_dt_data_list])
            hs.append(h_token_dt_drop)

        if self._sema_graph:
            # token_spans_pool dt drop
            # full_graph_drp_rate=0.8
            full_data_list_drop=[]
            for dt in dt_data_list_add_feature: # or updated_dt_data_list
                num=dt.num_nodes
                # if(num!=1):
                edge=list(permutations(range(num),2))
                for i in range(num):
                    edge.append((i, i))
                dt.edge_index = torch.tensor(list(zip(*edge)))
                dt.edge_type = torch.tensor([0]*len(edge))
                if num<5:
                    full_data_list_drop.append(dt)
                    continue
                edge_index,edge_type=zip(*random.sample(list(zip(dt.edge_index.t().tolist(),dt.edge_type.tolist())),round(dt.edge_type.size(0)* full_graph_retain_rate)))
                dt.edge_index,dt.edge_type=torch.tensor(edge_index).t(),torch.tensor(edge_type)
                full_data_list_drop.append(dt)
            batched_full = Batch.from_data_list(full_data_list_drop)
            batched_full.to(self._device)
            # batched_full.x = self.conv3(batched_full.x, batched_full.edge_index, batched_full.edge_type)
            batched_full.x = self.gat(batched_full.x, batched_full.edge_index)

            updated_full_data_list = batched_full.to_data_list()
            h_token_full = util.padded_stack([dt_data["x"] for dt_data in updated_full_data_list])
            hs.append(h_token_full)

        if self._fusion_rgcn: # hs must be full
            # B 768 1
            entity_ctx_tmp = get_token(h, encodings, self._cls_token).unsqueeze(-1)
            # B #T 768
            attns=[]
            for h_ in hs:
                # B #T 1
                attns.append(torch.matmul(self.node_W(h_),entity_ctx_tmp).squeeze(-1))
            # B #T 3 1
            attn=torch.stack(attns,dim=-1)
            # attn=self.relu(attn)
            attn=F.softmax(attn,dim=-1).unsqueeze(-1)
            # B #T 3 768
            hs=torch.stack(hs,dim=2)
            h_token=torch.mul(attn,hs).sum(dim=-2)
            token_spans_pool=(h_token+token_spans_pool)/2
        elif len(hs):
            h_token=(reduce(lambda x,y: x+y,hs)+token_spans_pool)/(len(hs)+1)
            # h_token+=token_spans_pool
        else:
            h_token=token_spans_pool

        # classify entities
        size_embeddings = self.size_embeddings(entity_sizes)  # embed entity candidate sizes
        entity_clf, entity_spans_pool = self._classify_entities(encodings, h, h_token, entity_masks, entity_masks_token, size_embeddings)

        return h, h_token, entity_spans_pool, size_embeddings, entity_clf, token_rel_attention, token_entity_attention

    def _forward_train(self, encodings: torch.tensor, context_masks: torch.tensor,token_m:torch.tensor, token_masks: torch.tensor, dt_data_list: List[Data], entity_masks: torch.tensor, entity_masks_token: torch.tensor,
                       entity_sizes: torch.tensor, relations: torch.tensor, rel_masks: torch.tensor,epoch=0):

        if epoch<self._split_epoch:
            self._tw_rel_atten_token=False
            self._tw_ent_atten_token=False
            self._tw_rel_atten_subword=False
            self._tw_ent_atten_subword=False
            self._trigger_attn=False
            # for param in self.bert.parameters():
                # param.requires_grad = False
        else:
            self._tw_rel_atten_token=self.save_config[0]
            self._tw_ent_atten_token=self.save_config[1]
            self._tw_rel_atten_subword=self.save_config[2]
            self._tw_ent_atten_subword=self.save_config[3]
            self._trigger_attn=self.save_config[4]
            # for param in self.bert.parameters():
                # param.requires_grad = False


        h, h_token, entity_spans_pool, size_embeddings, entity_clf,_,_=self._common_forward(encodings, context_masks,token_m,token_masks, dt_data_list, entity_masks, entity_masks_token, entity_sizes)

        batch_size = encodings.shape[0]

        # classify entities
        size_embeddings = self.size_embeddings(entity_sizes)  # embed entity candidate sizes
        entity_clf, entity_spans_pool = self._classify_entities(encodings, h, h_token, entity_masks, entity_masks_token, size_embeddings)

        # classify relations
        h_large = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)
        rel_clf = torch.zeros([batch_size, relations.shape[1], self._relation_types]).to(
            self.rel_classifier.weight.device)

        for i in range(0, relations.shape[1], self._max_pairs):
            # classify relation candidates
            chunk_rel_logits, _ = self._classify_relations(entity_spans_pool, size_embeddings,
                                                        relations, rel_masks, h_large,h_token, entity_masks_token, i,token_m)
            rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_logits
            
        return entity_clf, rel_clf

    def _forward_eval(self, encodings: torch.tensor, context_masks: torch.tensor,token_m:torch.tensor,token_masks: torch.tensor, dt_data_list: List[Data], entity_masks: torch.tensor, entity_masks_token: torch.tensor, entity_sizes: torch.tensor, entity_spans: torch.tensor, entity_sample_masks: torch.tensor):

        h, h_token, entity_spans_pool, size_embeddings, entity_clf, token_rel_attention, token_entity_attention=self._common_forward(encodings, context_masks,token_m,token_masks, dt_data_list, entity_masks, entity_masks_token, entity_sizes)

        batch_size = encodings.shape[0]
        ctx_size = context_masks.shape[-1]
        # classify entities
        size_embeddings = self.size_embeddings(entity_sizes)  # embed entity candidate sizes
        entity_clf, entity_spans_pool = self._classify_entities(encodings, h, h_token, entity_masks, entity_masks_token, size_embeddings)

        # ignore entity candidates that do not constitute an actual entity for relations (based on classifier)
        relations, rel_masks, rel_sample_masks = self._filter_spans(entity_clf, entity_spans,
                                                                    entity_sample_masks, ctx_size)

        rel_sample_masks = rel_sample_masks.float().unsqueeze(-1)
        h_large = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)
        rel_clf = torch.zeros([batch_size, relations.shape[1], self._relation_types]).to(
            self.rel_classifier.weight.device)

        trigger_attn = None
        for i in range(0, relations.shape[1], self._max_pairs):
            # classify relation candidates
            chunk_rel_logits,trigger_attn = self._classify_relations(entity_spans_pool, size_embeddings,
                                                        relations, rel_masks, h_large,h_token, entity_masks_token, i,token_m)
            # apply sigmoid
            chunk_rel_clf = torch.sigmoid(chunk_rel_logits)
            rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_clf

        rel_clf = rel_clf * rel_sample_masks  # mask

        # apply softmax
        entity_clf = torch.softmax(entity_clf, dim=2)
        if token_entity_attention is not None:
            token_entity_attention = token_entity_attention.squeeze(-1).tolist()
        if token_rel_attention is not None:
            token_rel_attention = token_rel_attention.squeeze(-1).tolist()
        if trigger_attn is not None:
            trigger_attn = trigger_attn[:,0,:].tolist()
        return entity_clf, rel_clf, relations, token_entity_attention,token_rel_attention, trigger_attn

    def _classify_entities(self, encodings, h, h_token, entity_masks, entity_masks_token, size_embeddings):
        # import pdb; pdb.set_trace()
        # max pool entity candidate spans
        m = (entity_masks.unsqueeze(-1) == 0).float() * (-1e30)
        entity_spans_pool = m + h.unsqueeze(1).repeat(1, entity_masks.shape[1], 1, 1)
        entity_spans_pool = entity_spans_pool.max(dim=2)[0]

        m = (entity_masks_token.unsqueeze(-1) == 0).float() * (-1e30)
        entity_spans_pool_token = m + h_token.unsqueeze(1).repeat(1, entity_masks_token.shape[1], 1, 1)
        entity_spans_pool = entity_spans_pool_token.max(dim=2)[0] + entity_spans_pool

        # get cls token as candidate context representation
        entity_ctx = get_token(h, encodings, self._cls_token)

        # create candidate representations including context, max pooled span and size embedding
        entity_repr = torch.cat([entity_ctx.unsqueeze(1).repeat(1, entity_spans_pool.shape[1], 1),
                                 entity_spans_pool, size_embeddings], dim=2)
        entity_repr = self.dropout(entity_repr)

        # classify entity candidates
        entity_repr = self.entity_linear(entity_repr)
        # entity_repr = self.dropout(entity_repr)
        entity_clf = self.entity_classifier(entity_repr)
        # print(entity_clf.size())

        return entity_clf, entity_spans_pool

    # @torchsnooper.snoop()
    def _classify_relations(self, entity_spans, size_embeddings, relations, rel_masks, h, h_token, entity_masks_token,chunk_start,token_m):
        # import pdb; pdb.set_trace()
        # torch.set_printoptions(profile="full")
        batch_size = relations.shape[0]
        if not self._trigger_grad_flow:
            trigger_rel_w = self.rel_classifier.weight.detach()
            # trigger_rel_b=self.rel_classifier.bias.detach()
            trigger_ent_w = self.entity_classifier.weight[1:].detach()
            # trigger_ent_b=self.entity_classifier.bias[1:].detach()
        else:
            trigger_rel_w = self.rel_classifier.weight
            # trigger_rel_b=self.rel_classifier.bias
            trigger_ent_w = self.entity_classifier.weight[1:]
            # trigger_ent_b=self.entity_classifier.bias[1:]


        # create chunks if necessary
        if relations.shape[1] > self._max_pairs:
            relations = relations[:, chunk_start:chunk_start + self._max_pairs]
            rel_masks = rel_masks[:, chunk_start:chunk_start + self._max_pairs]
            h = h[:, :relations.shape[1], :]
        # import pdb; pdb.set_trace()
        # B #Rels 2 768
        entity_pairs = util.batch_index(entity_spans, relations)
        entity_pair_embed = entity_pairs.view(batch_size, entity_pairs.shape[1], -1)


        # relation context (context between entity candidate pair)
        # mask non entity candidate tokens
        m = ((rel_masks == 0).float() * (-1e30)).unsqueeze(-1)
        rel_ctx = m + h
        # max pooling
        rel_ctx = rel_ctx.max(dim=2)[0]
        # set the context vector of neighboring or adjacent entity candidates to zero
        rel_ctx[rel_masks.to(torch.uint8).any(-1) == 0] = 0

        
        attn=None
        if self._trigger_attn:

            # type: 1
            # # B #Rels 768
            # attn_head=torch.matmul(self.head_W(entity_pairs[:,:,0,:]),h_token.transpose(-1,-2))
            # attn_tail=torch.matmul(self.tail_W(entity_pairs[:,:,1,:]),h_token.transpose(-1,-2))
            # # ent_pair_sub=(entity_pairs[:,:,0,:]+entity_pairs[:,:,1,:])/2
            # # B #Rels #token
            # # attn=torch.matmul(ent_pair_sub,h_token.transpose(-1,-2))
            # attn=torch.sum(torch.stack([attn_head,attn_tail],dim=-1),dim=-1)

            # # B #Rels #token
            # # mask=(h_token.sum(-1)!=0).unsqueeze(1).repeat(1,relations.size(1),1)
            # mask=token_m.unsqueeze(1).repeat(1,relations.size(1),1)

            # # B #Rels 2 #token
            # rel_token_mask= util.batch_index(entity_masks_token, relations)
            # # B #Rels #token
            # rel_token_mask=rel_token_mask[:,:,0,:]+rel_token_mask[:,:,1,:]
            # # B #Rels #token
            # mask=mask^rel_token_mask
            # # B #Rels #token
            # attn=self.dropout(attn)
            
            # attn=util.masked_softmax(attn,mask,dim=-1)

            # # rel_ctx2=torch.sum(h_token.unsqueeze(1)*mask.unsqueeze(-1),dim=-2)

            # # print(attn)
            # # B #Rels 768
            # rel_ctx2=torch.matmul(attn,h_token)
            # rel_ctx2[rel_masks.to(torch.uint8).any(-1) == 0] = 0
            # rel_ctx+=rel_ctx2


            # type: 2
            # B #Rel #longest_context_size  ---> B #Rel
            rel_num_mask=(rel_masks.sum(-1)!=0)
            # B #Rels #Rel_Cls
            # attn_head=torch.matmul(self.head_W(entity_pairs[:,:,0,:]),trigger_rel_w.transpose(-1,0))
            # attn_head=self.relu(attn_head)
            attn_head=torch.matmul(entity_pairs[:,:,0,:],trigger_rel_w.transpose(-1,0))

            # attn_head=attn_head.masked_fill(~rel_num_mask.unsqueeze(-1).repeat(1,1,trigger_rel_w.size(0)),0)
            attn_head=F.softmax(attn_head,dim = -1)
            attn_head=attn_head.masked_fill(~rel_num_mask.unsqueeze(-1).repeat(1,1,trigger_rel_w.size(0)),0)

            # attn_tail=torch.matmul(self.head_W(entity_pairs[:,:,1,:]),trigger_rel_w.transpose(-1,0))
            # attn_tail=self.relu(attn_tail)
            attn_tail=torch.matmul(entity_pairs[:,:,1,:],trigger_rel_w.transpose(-1,0))

            # attn_tail=attn_tail.masked_fill(~rel_num_mask.unsqueeze(-1).repeat(1,1,trigger_rel_w.size(0)),0)
            attn_tail=F.softmax(attn_tail,dim = -1)
            attn_tail=attn_tail.masked_fill(~rel_num_mask.unsqueeze(-1).repeat(1,1,trigger_rel_w.size(0)),0)
            # B #Rels 768
            
            rel_re=torch.matmul(torch.max(torch.stack([attn_head,attn_tail],dim=-1),dim=-1)[0],trigger_rel_w)
            # rel_re=F.linear(torch.mean(torch.stack([attn_head,attn_tail],dim=-1),dim=-1),trigger_rel_w)

            # attn_rel_ctx=torch.matmul(self.head_W(rel_ctx.clone()),trigger_rel_w.transpose(-1,0))
            # attn_rel_ctx=attn_rel_ctx.masked_fill(~rel_num_mask.unsqueeze(-1).repeat(1,1,trigger_rel_w.size(0)),0)

            # B #Rels 768    B #token 768   ----> B #Rels #token
            attn=torch.matmul(rel_re, h_token.transpose(-1,-2))
            # attn=self.relu(attn)
            # print("relations")
            # print(relations)
            # print("rel_masks")
            # print(rel_masks)
            # print("trigger_rel_w")
            # print(trigger_rel_w)
            # print("h_token")
            # print(h_token)

            # print("rel_re")
            # print(rel_re)
            # # return

            mask=token_m.unsqueeze(1).repeat(1,relations.size(1),1)
            # # B #Rels 2 #token
            # rel_token_mask= util.batch_index(entity_masks_token, relations)
            # # B #Rels #token
            # rel_token_mask=rel_token_mask[:,:,0,:]+rel_token_mask[:,:,1,:]
            # # B #Rels #token
            # mask=mask^rel_token_mask

            attn=util.masked_softmax(attn,mask,dim=-1)
            rel_ctx2=torch.matmul(attn,h_token)
            rel_ctx=(rel_ctx2+rel_ctx)
            # print("attn")
            # print(attn)
            # return

        size_pair_embeddings = util.batch_index(size_embeddings, relations)
        size_pair_embeddings = size_pair_embeddings.view(batch_size, size_pair_embeddings.shape[1], -1)

        rel_repr = torch.cat([rel_ctx, entity_pair_embed, size_pair_embeddings], dim=2)
        rel_repr = self.dropout(rel_repr)

        rel_repr = self.rel_linear(rel_repr)
        # rel_repr = self.dropout(rel_repr)
        chunk_rel_logits = self.rel_classifier(rel_repr)
        return chunk_rel_logits, attn

    def _filter_spans(self, entity_clf, entity_spans, entity_sample_masks, ctx_size):
        batch_size = entity_clf.shape[0]
        entity_logits_max = entity_clf.argmax(dim=-1) * entity_sample_masks.long()  # get entity type (including none)
        batch_relations = []
        batch_rel_masks = []
        batch_rel_sample_masks = []

        for i in range(batch_size):
            rels = []
            rel_masks = []
            sample_masks = []

            non_zero_indices = (entity_logits_max[i] != 0).nonzero().view(-1)
            non_zero_spans = entity_spans[i][non_zero_indices].tolist()
            non_zero_indices = non_zero_indices.tolist()

            # create relations and masks
            for i1, s1 in zip(non_zero_indices, non_zero_spans):
                for i2, s2 in zip(non_zero_indices, non_zero_spans):
                    if i1 != i2:
                        rels.append((i1, i2))
                        rel_masks.append(sampling.create_rel_mask(s1, s2, ctx_size))
                        sample_masks.append(1)

            if not rels:
                # case: no more than two spans classified as entities
                batch_relations.append(torch.tensor([[0, 0]], dtype=torch.long))
                batch_rel_masks.append(torch.tensor([[0] * ctx_size], dtype=torch.bool))
                batch_rel_sample_masks.append(torch.tensor([0], dtype=torch.bool))
            else:
                # case: more than two spans classified as entities
                batch_relations.append(torch.tensor(rels, dtype=torch.long))
                batch_rel_masks.append(torch.stack(rel_masks))
                batch_rel_sample_masks.append(torch.tensor(sample_masks, dtype=torch.bool))

        # stack
        device = self.rel_classifier.weight.device
        batch_relations = util.padded_stack(batch_relations).to(device)
        batch_rel_masks = util.padded_stack(batch_rel_masks).to(device)
        batch_rel_sample_masks = util.padded_stack(batch_rel_sample_masks).to(device)

        return batch_relations, batch_rel_masks, batch_rel_sample_masks

    def forward(self, *args, evaluate=False, **kwargs):
        if not evaluate:
            return self._forward_train(*args, **kwargs)
        else:
            return self._forward_eval(*args, **kwargs)


# Model access

_MODELS = {
    'trimf': TriMF,
}


def get_model(name):
    return _MODELS[name]
