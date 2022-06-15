import torch
from solver import ArbitrarySolver
from graph import Segment, set_segment_training, parse_computation_graph, optimize_computation_graph, get_source_target
from utils import set_reproductibility, disable_dropout
import time
import numpy as np
from tqdm import tqdm
from net.model_factory import model_factory, input_sizes
from utils import load_pickle, save_pickle
import argparse
import os

torch.backends.cudnn.enabled = True


def forward_check(net, parsed_segment, run_segment, device, input_size=(1,3,224,224)):
    inp = torch.rand(*input_size).to(device)
    ## net type 'net.resnet.ResNet
    net.train()
    # net.eval()
    set_segment_training(parsed_segment, train=True)
    set_segment_training(run_segment, train=True)

    torch.cuda.empty_cache()
    with torch.no_grad():
        ori_output = net(inp)
        parsed_graph_output = parsed_segment.forward(inp)
        run_graph_output = run_segment.forward(inp)
    max_graph_err = torch.max(torch.abs(parsed_graph_output - ori_output))
    if max_graph_err < 1e-05:
        print('Parsed graph forward check passed')
    else:
        ## zhaojp why difference?? 
        print('Parsed graph forward check failed: Max Difference {} !!!!'.format(max_graph_err))

    max_run_graph_err = torch.max(torch.abs(run_graph_output - ori_output))
    if max_run_graph_err < 1e-05:
        print('Run graph forward check passed')
    else:
        print('Run graph forward check failed: Max Difference {}'.format(max_run_graph_err))

    torch.cuda.empty_cache()


def backward_check(net, parsed_segment, run_segment, device, input_size=(1,3,224,224)):
    inp = torch.rand(*input_size).to(device)
    inp.requires_grad = True
    net.train()
    set_segment_training(parsed_segment, train=True)
    set_segment_training(run_segment, train=True)
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

    run_graph_output = run_segment.forward(inp)
    loss = torch.sum(output_target - run_graph_output)
    loss.backward()
    run_graph_grad = [p.grad.clone() for p in net.parameters()]

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

    max_run_graph_err = 0
    for g1, g2 in zip(ori_grad, run_graph_grad):
        if torch.norm(g1) > 1e-02:
            rel_err = torch.max(torch.abs(g2 - g1)) / torch.norm(g1)
        else:
            rel_err = torch.max(torch.abs(g2 - g1))
        if rel_err > max_run_graph_err:
            max_run_graph_err = rel_err

    if max_run_graph_err < 1e-03:
        print('Run graph backward check passed')
    else:
        print('Run graph backward check failed: Max Difference {}'.format(max_run_graph_err))

    torch.cuda.empty_cache()

def forward_backward(module, device, input_size=(1,3,224,224), repeat=100, min_repeat=5):
    # do backward 1 time to get gradients counted
    '''
    input2 = torch.rand(*input_size, device=device)
    input2.requires_grad = True
    output2 = module(input2)
    loss = torch.sum(output2)
    loss.backward()
    del input2, output2, loss
    '''
    # del input2, output2

    #torch.cuda.synchronize()
    #torch.cuda.empty_cache()
    #time.sleep(2)
    #os.system('nvidia-smi')

    torch.cuda.reset_max_memory_allocated(device)
    regular_start_memory = torch.cuda.max_memory_allocated(device)
    regular_times = []

    for i in tqdm(range(repeat)):
        start = time.time()
        input2 = torch.rand(*input_size, device=device)
        input2.requires_grad = True
        output2 = module(input2)
        loss = torch.sum(output2)
        loss.backward()
        end = time.time()
        regular_times.append(end - start)
        del input2, output2, loss
        # del input2, output2
        #print("--------fwd_bwd_benchmark loop done")
        #torch.cuda.synchronize()
        #print(torch.cuda.memory_summary())

    torch.cuda.synchronize()
    regular_peak_memory = torch.cuda.max_memory_allocated(device)
    torch.cuda.empty_cache()
    regular_end_memory = torch.cuda.memory_allocated(device)
    print(torch.cuda.memory_summary())

    regular_avg_time = np.mean(np.array(regular_times)[min_repeat:])

    torch.cuda.empty_cache()

    return regular_start_memory, regular_end_memory, regular_peak_memory, regular_avg_time


def forward_backward_benchmark(net, run_segment, device, input_size=(1,3,224,224), repeat=100, min_repeat=5):
    assert repeat > min_repeat
    print("--------fwd_bwd_benchmark before net.train()")
    torch.cuda.synchronize()
    print(torch.cuda.memory_summary())
    net.train()
    print("--------fwd_bwd_benchmark after net.train()")
    torch.cuda.synchronize()
    print(torch.cuda.memory_summary())

    checkpoint_start_memory, checkpoint_end_memory, checkpoint_peak_memory, checkpoint_avg_time = forward_backward(run_segment, device, input_size, repeat, min_repeat)
    checkpoint_pytorch_overhead = max(checkpoint_start_memory, checkpoint_end_memory)
    checkpoint_intermediate_tensors = checkpoint_peak_memory - checkpoint_pytorch_overhead

    print('Average Iteration Time: Checkpointing {:.4f} s '.format(checkpoint_avg_time))
    print('Average Peak Memory: Checkpointing {:.4f} MB '.format( checkpoint_peak_memory / (1024**2)))
    print('Average Intermediate Tensors: Checkpointing {:.4f} MB start: {:.4f} MB end: {:.4f}'.format(
        checkpoint_intermediate_tensors / (1024 ** 2),
        checkpoint_start_memory / (1024 ** 2),
        checkpoint_end_memory / (1024 ** 2)))

    torch.cuda.synchronize()
    #torch.cuda.empty_cache()
    time.sleep(3)

    regular_start_memory, regular_end_memory, regular_peak_memory, regular_avg_time = forward_backward(net, device, input_size, repeat, min_repeat)
    regular_pytorch_overhead = max(regular_start_memory, regular_end_memory)
    regular_intermediate_tensors = regular_peak_memory - regular_pytorch_overhead
    print('Average Iteration Time: baseline {:.4f} s '.format(regular_avg_time))
    print('Average Peak Memory: baseline {:.4f} MB '.format( regular_peak_memory / (1024**2)))
    print('Average Intermediate Tensors: baseline {:.4f} MB start: {:.4f} MB end: {:.4f}'.format(
        regular_intermediate_tensors / (1024 ** 2),
        regular_start_memory / (1024 ** 2),
        regular_end_memory / (1024 ** 2)))

    '''
    print('Average Iteration Time: Checkpointing {:.4f} s, Regular {:.4f} s, overhead {:.2f}%'.format(
        checkpoint_avg_time, regular_avg_time, (checkpoint_avg_time - regular_avg_time) * 100 / regular_avg_time))
    print('Average Peak Memory: Checkpointing {:.4f} MB, Regular {:.4f} MB, Memory Cut off {:.2f}%'.format(
        checkpoint_peak_memory / (1024**2), regular_peak_memory / (1024**2), (regular_peak_memory - checkpoint_peak_memory) * 100 / regular_peak_memory))
    print('Average Intermediate Tensors: Checkpointing {:.4f} MB, Regular {:.4f} MB, Memory Cut off {:.2f}%'.format(
        checkpoint_intermediate_tensors / (1024 ** 2), regular_intermediate_tensors / (1024 ** 2), (regular_intermediate_tensors - checkpoint_intermediate_tensors) * 100 / regular_intermediate_tensors))
        '''



def main(arch, device):
    set_reproductibility(2020)
    input_size = input_sizes[arch]
    print('Processing {}, Input size {}'.format(arch, input_size) + '-' * 20)
    net = model_factory[arch]().to(device)
    disable_dropout(arch, net)
    ## zhaojp 'net.resnet.ResNet
    print(type(net))
    print("--------main before net.train()")
    torch.cuda.synchronize()
    print(torch.cuda.memory_summary())
    net.train()
    print("--------main after net.train()")
    torch.cuda.synchronize()
    print(torch.cuda.memory_summary())
    # with torch.no_grad():
    #     inp = torch.rand(*input_size).to(device)
    #     G, source, target = net.parse_graph(inp)
    print('Parsing Computation Graph')
    inputs = [torch.rand(*input_size).to(device)]
    with torch.no_grad():
        inp = torch.rand(*input_size).to(device)
        print("--------before manual parse_graph")
        torch.cuda.synchronize()
        print(torch.cuda.memory_summary())
        G, source, target = net.parse_graph(inp)
        print("--------after manual parse_graph")
        torch.cuda.synchronize()
        print(torch.cuda.memory_summary())
    '''
    try:
        G, source, target = parse_computation_graph(net, inputs)
    except:
        ## zhaojp note it didn't release memory !!!
        print('Parsing Computation Graph with torch.jit failed, revert to manual parse_graph function')
        with torch.no_grad():
            inp = torch.rand(*input_size).to(device)
            print("--------before manual parse_graph")
            torch.cuda.synchronize()
            print(torch.cuda.memory_summary())
            G, source, target = net.parse_graph(inp)
            print("--------after manual parse_graph")
            torch.cuda.synchronize()
            print(torch.cuda.memory_summary())
            '''

    solver = ArbitrarySolver()

    start = time.time()
    run_graph, best_cost = solver.solve(G, source, target)

    torch.cuda.synchronize()
    print("--------before run_segment")
    print(torch.cuda.memory_summary())
    run_segment = Segment(run_graph, source, target, do_checkpoint=True)
    torch.cuda.synchronize()
    print("--------after run_segment")
    print(torch.cuda.memory_summary())

    parsed_segment = Segment(G, source, target, do_checkpoint=False)
    print("--------after parsed_segment")
    print(torch.cuda.memory_summary())

    end = time.time()
    print('Solving optimal gradient checkpointing takes {:.4f} s'.format(end - start))
    # zhaojp check use batch=1
    #forward_check(net, parsed_segment, run_segment, device, input_size=input_size)
    #backward_check(net, parsed_segment, run_segment, device, input_size=input_size)

    torch.cuda.synchronize()
    print("--------before forward_check")
    print(torch.cuda.memory_summary())
    forward_check(net, parsed_segment, run_segment, device)
    torch.cuda.synchronize()
    print("--------after forward_check")
    print(torch.cuda.memory_summary())

    backward_check(net, parsed_segment, run_segment, device)
    torch.cuda.synchronize()
    print("--------after bwd_check")
    print(torch.cuda.memory_summary())

    forward_backward_benchmark(net, run_segment, device, input_size=input_size, repeat=100, min_repeat=1)

    time.sleep(1)
    print("--------to finish")
    print(torch.cuda.memory_summary())

def parse_args():
    parser = argparse.ArgumentParser(description='Run Optimal Gradient Checkpoiting')
    parser.add_argument('--arch',
                        help='network architecture name',
                        required=True,
                        type=str)
    parser.add_argument('--device',
                        help='gpu device',
                        default='cuda:0',
                        type=str)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    device = torch.device(args.device)
    arch = args.arch
    if arch not in model_factory:
        print('Available Archs are {}'.format(model_factory.keys()))
        raise KeyError
    main(arch, device)
