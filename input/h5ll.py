import fire
import h5py

def print_attributes(node, shift):
    for key, val in node.attrs.items():
        print(f"{shift} {key}: {val} [attribute]")

def print_node_info(name, node):
    space = '    '
    shift = name.count('/') * space
    if isinstance(node, h5py.Dataset):
        print(shift + node.name, ' [dataset]')
        print(f"{shift + space} shape: {node.shape} [shape]")
    else:
        print(shift + node.name, ' [group]')
    print_attributes(node, shift + space)

def h5ls(name='mytestfile.h5'):
    with h5py.File(name, 'r') as obj:
        print_attributes(obj, shift='')
        obj.visititems(print_node_info)
    return '***************** end of file *****************'

if __name__ == '__main__':
    fire.Fire(h5ls)
