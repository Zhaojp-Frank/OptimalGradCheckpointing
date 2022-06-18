#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../')
import torch
import torch.nn as nn
import networkx as nx
from net.layer import BasicCatDim1, BasicIdentity


# # Define Model and parse_graph function
# You can also define your own parse_graph function when the automatic parse_graph fails or you want to customize the granularity of your computation graph.

# In[2]:


class CustomNet(nn.Module):

    def __init__(self):
        super(CustomNet, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(64, 32, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(64, 32, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(64)



    def forward(self, x):
        x = self.relu(x)
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        out = torch.cat([x1, x2], dim=1)
        out = self.bn(out)
        return out

    def parse_graph(self, x):
        # parse_graph forwards the network and create nodes and edges for the computation graph
        # Each node is a tensor and has field cost=tensor.numel(), each edge is an operation and has field cost=0, module=op
        # You may want to combine nn.ReLU(inplace=True) with previous operation into 1 op so that the computation graph 
        # reflects the actual memory usage
        
        G = nx.MultiDiGraph()
        source = 0
        vertex_id = 0
        G.add_node(vertex_id, cost=x.numel())
        
        # inplace=False for relu so not combining relu and conv2d into 1 op
        op = self.relu
        x = op(x)
        G.add_node(vertex_id + 1, cost=x.numel())
        G.add_edge(vertex_id, vertex_id + 1, cost=0, module=op)
        vertex_id += 1
        input_id = vertex_id


        op1 = self.conv_1
        x1 = op1(x)
        G.add_node(vertex_id + 1, cost=x1.numel())
        G.add_edge(input_id, vertex_id + 1, cost=0, module=op1)
        vertex_id += 1
        x1_id = vertex_id


        op2 = self.conv_2
        x2 = op2(x)
        G.add_node(vertex_id + 1, cost=x2.numel())
        G.add_edge(input_id, vertex_id + 1, cost=0, module=op2)
        vertex_id += 1
        x2_id = vertex_id

        # for an op with multiple input tensors, we define the op as a transition op, to transition multiple input tensor into 1 output tensor
        # for example x = torch.cat([x1, x2]), we need to first add node x to the computation graph
        # and put torch.cat operation as a transition operation into node x, and define the tranisition order as [x1_id, x2_id] for node x. 
        # Then we need to add the edge x1->x, x2->x into the computation graph. For these two edges, we just need to put place holder 
        # operation BasicIdentity to it. BasicIdentity does nothing and directly returns the tensor. 
        
        op = BasicCatDim1()
        identity = BasicIdentity()
        x = op([x1, x2])
        G.add_node(vertex_id + 1, cost=x.numel(), transition=op)
        G.nodes[vertex_id + 1]['transition_input_order'] = []
        for id in [x1_id, x2_id]:
            edge_id = G.add_edge(id, vertex_id + 1, cost=0, module=identity)
            G.nodes[vertex_id + 1]['transition_input_order'].append((id, edge_id))
        vertex_id += 1

        op = self.bn
        x = op(x)
        G.add_node(vertex_id + 1, cost=x.numel())
        G.add_edge(vertex_id, vertex_id + 1, cost=0, module=op)
        vertex_id += 1
        
        target = vertex_id
        
        # finally we returns the computation graph, source and target node key of the computation graph
        return G, source, target


# # Forward and backward check of the parsed graph
# If our forward function contains random operation, such as dropout, the forward and backward check will fail as the rng status has changed. We may want to disable the random operation when doing forward backward check.

# In[3]:


from graph import Segment, set_segment_training

def forward_check(net, parsed_segment, device, input_size=(1,3,224,224)):
    inp = torch.rand(*input_size).to(device)
    net.train()
    set_segment_training(parsed_segment, train=True)

    with torch.no_grad():
        ori_output = net(inp)
        parsed_graph_output = parsed_segment.forward(inp)

    max_graph_err = torch.max(torch.abs(parsed_graph_output - ori_output))
    if max_graph_err < 1e-05:
        print('Parsed graph forward check passed')
    else:
        print('Parsed graph forward check failed: Max Difference {}'.format(max_graph_err))

    torch.cuda.empty_cache()


def backward_check(net, parsed_segment, device, input_size=(1,3,224,224)):
    inp = torch.rand(*input_size).to(device)
    inp.requires_grad = True
    net.train()
    set_segment_training(parsed_segment, train=True)

    ori_output = net(inp)
    output_target = torch.rand(*ori_output.shape).to(device)

    loss = torch.sum(output_target - ori_output)
    loss.backward()
    ori_grad = [p.grad.clone() for p in net.parameters()]
    net.zero_grad()
    del ori_output, loss
    torch.cuda.empty_cache()

    parsed_graph_output = parsed_segment.forward(inp)
    loss = torch.sum(output_target - parsed_graph_output)
    loss.backward()
    graph_grad = [p.grad.clone() for p in net.parameters()]

    net.zero_grad()

    max_graph_err = 0
    for g1, g2 in zip(ori_grad, graph_grad):
        if torch.norm(g1) > 1e-02:
            rel_err = torch.max(torch.abs(g2 - g1)) / torch.norm(g1)
        else:
            rel_err = torch.max(torch.abs(g2 - g1))
        if rel_err > max_graph_err:
            max_graph_err = rel_err

    if max_graph_err < 1e-03:
        print('Parsed graph backward check passed')
    else:
        print('Parsed graph backward check failed: Max Difference {}'.format(max_graph_err))

    torch.cuda.empty_cache()
    
input_size = (4, 64, 32, 32)
device = torch.device('cuda:0')
net = CustomNet().to(device)
net.train()
inp = torch.rand(*input_size).to(device)
G, source, target = net.parse_graph(inp)

parsed_segment = Segment(G, source, target, do_checkpoint=True)
print((G.nodes[4]['transition_input_order']))

#for kv in parsed_segment.info_dict.items():
#   print(kv)

forward_check(net, parsed_segment, device, input_size=input_size)
backward_check(net, parsed_segment, device, input_size=input_size)


# In[ ]:




