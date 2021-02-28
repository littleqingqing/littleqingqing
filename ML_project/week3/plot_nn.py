import numpy as np
from viznet import connecta2a, node_sequence, NodeBrush, EdgeBrush, DynamicShow


def draw_feed_forward(ax, num_node_list):
    '''
    draw a feed forward neural network.

    Args:
        num_node_list (list<int>): array
    '''
    num_hidden_layer = len(num_node_list) - 2  # hidden layers
    token_list = ['\sigma^z'] + \
        ['y^{(%s)}' % (i + 1) for i in range(num_hidden_layer)] + ['\psi']
    kind_list = ['nn.input'] + ['nn.hidden'] * num_hidden_layer + ['nn.output']
    radius_list = [0.3] + [0.2] * num_hidden_layer + [0.3]   # ridius
    y_list = - 1.5 * np.arange(len(num_node_list))  # axis
    
    seq_list = []
    for n, kind, radius, y in zip(num_node_list, kind_list, radius_list, y_list):
        b = NodeBrush(kind, ax)
        seq_list.append(node_sequence(b, n, center=(0, y)))

    eb = EdgeBrush('-->', ax)
    for st, et in zip(seq_list[:-1], seq_list[1:]):
        connecta2a(st, et, eb)
    #for i, layer_nodes in enumerate(seq_list):
        #[node.text('$z_%i^{(%i)}$'%(j, i), 'center', fontsize=16) for j, node in enumerate(layer_nodes)]
    return seq_list


def real_bp():
    with DynamicShow((6, 6), '_feed_forward.png') as d:  # hidden axis
        seq_list = draw_feed_forward(d.ax, num_node_list=[4, 1])
        for i, layer_nodes in enumerate(seq_list):
            [node.text('$z_{%i}^{(%i)}$'%(j, i), 'center', fontsize=16) for j, node in enumerate(layer_nodes)]


if __name__ == '__main__':
    real_bp()
