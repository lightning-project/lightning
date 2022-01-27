import argparse
import json
import tempfile
import subprocess
import numpy as np
from collections import defaultdict

def hex2rgb(s):
    return tuple(int(s[i:i+2], 16) for i in (1, 3, 5))

def rgb2hex(rgb):
    return '#%02x%02x%02x' % (int(rgb[0]), int(rgb[1]), int(rgb[2]))


class DotWriter:
    NODE_COLORS = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a',
            '#66a61e', '#e6ab02', '#a6761d', '#666666',] * 10

    NODE_COLORS = [rgb2hex(np.array(hex2rgb(s)) * 0.25 + 128 + 64) for s in NODE_COLORS]



    def __init__(self, args):
        self.args = args
        self.clusters = []
        self.cluster_names = []
        self.mapping = defaultdict(list)
        self.reachable = defaultdict(list)
        self.transfers = defaultdict(list)
        self.node_epoch = defaultdict(str)

        self.last_op = None

    def process_line(self, data):
        #self.mapping.clear()

        if not data:
            return

        cluster = ''
        epoch = len(self.clusters)
        name = 'cluster_epoch_{}'.format(epoch)

        if not self.args.no_outline:
            cluster += 'subgraph {} {{\n'.format(name)
            cluster += 'bgcolor="#f1f1f1";\n'
            cluster += 'newrank="true";'

        data.sort(key=lambda p: p['node_id'])
        last_node = None

        for op in data:
            id = op['id']
            node = op['node_id']

            if op['id'] < self.args.start or op['id'] > self.args.end:
                continue

            #if node != 0:
            #    continue

            body = self.process_node(op, epoch)

            if not body:
                continue

            if last_node != node:
                if not self.args.no_outline:
                    if last_node is not None:
                        cluster += '}\n'

                    cluster += 'subgraph cluster_node_{}_{} {{\n'.format(node, len(self.clusters))
                    cluster += 'bgcolor="#ffffff";\n'
                    #cluster += 'style=invis;\n'
                last_node = node

            cluster += body + '\n'

        if last_node is None:
            return

        if not self.args.no_outline:
            cluster += '}\n'

        if not self.args.no_outline:
            #cluster += f'dummy_{name} [label="", shape=plaintext, width=0, height=0, margin=0];'
            cluster += '}\n'

            if self.cluster_names:
                a = self.cluster_names[-1]
                b = name
                #cluster += f'dummy_{a} -> dummy_{b} [ltail={a} lhead={b}];\n'


        self.cluster_names.append(name)
        self.clusters.append(cluster)


        for key in list(self.mapping):
            #self.mapping[key] = []
            pass

    def format_id(self, id):
        return f'${id}'

    def generate_label(self, op):
        op_id = op['id']
        props = op['task']
        kind = op['task']['kind']

        if kind == 'create_chunk':
            id = self.format_id(props["chunk_id"])
            label = f'create {id}'

        elif kind == 'destroy_chunk':
            id = self.format_id(props["chunk_id"])
            label = f'delete {id}'

        elif kind == 'copy':
            src_id = self.format_id(op['chunks'][0]['id'])
            dst_id = self.format_id(op['chunks'][-1]['id'])
            label = f'copy {src_id}→{dst_id}'

        elif kind == 'send':
            src_id = self.format_id(op['chunks'][0]['id'])
            label = f'send {src_id}'

        elif kind == 'recv':
            dst_id = self.format_id(op['chunks'][0]['id'])
            label = f'recv {dst_id}'

        elif kind == "probe":
            label = f'probe'

        elif kind == 'execute':
            name = props['name']
            src_id = self.format_id(op['chunks'][0]['id'])
            dst_id = self.format_id(op['chunks'][-1]['id'])

            if 'Fill' in name:
                label = f'fill {dst_id}'
            elif 'Read' in name:
                label = f'read {src_id}'
            elif 'Write' in name:
                label = f'write {dst_id}'
            elif 'Reduce' in name:
                label = f'reduce {src_id}→{dst_id}'
            elif 'Combine' in name:
                label = f'combine {src_id}→{dst_id}'
            elif 'Launch' in name:
                label = 'launch'
            elif 'Fold' in name:
                label = 'combine'
            else:
                label = name
        elif kind == 'sync':
            label = 'sync'
        else:
            print(f'what?: {kind}')
            label = kind

        if self.args.short_labels:
            return label
        else:
            return f'({op_id}) {label}'


    def process_node(self, op, epoch):
        node = op['node_id']
        id = op['id']
        kind = op['task']['kind']
        key = f'op_{id}'
        hide_node = False

        if kind == 'execute' and 'fill' in op['task']['name']:
            hide_node = True

        if not self.args.include_events and kind == 'sync':
            hide_node = True

        if self.args.hide_chunks and kind in ['create_chunk', 'destroy_chunk']:
            hide_node = True

        if not self.args.include_joins and (kind == 'empty' or kind == 'probe'):
            hide_node = True

        if hide_node:
            result = []
            for dep in op['dependencies']:
                result.extend(self.mapping[dep])

            self.mapping[id] = result
            return ''

        attr = ''

        if kind != 'empty':
            #shape = 'hexagon' if kind == 'execute' else 'box'
            shape = 'box'
            label = self.generate_label(op)

            s = f'{key} [shape={shape}, label="{label}", style="filled", color="{self.NODE_COLORS[node]}", bgcolor="red" {attr}];\n'
        else:
            s = f'{key} [shape=triangle, label="", width=0.15, height=0.2 {attr}];\n'

        flows = list(op['chunks'])
        deps = list(op['dependencies'])

        for index, flow in enumerate(flows):
            if not self.args.include_flows:
                deps += flow['dependencies']
                continue

            name = '{}_flow_{}'.format(key, index)
            s += '{} [label="", shape=circle, width=0.2];\n'.format(name)
            s += '{} -> {};\n'.format(name, key)

            for dep in flow['dependencies']:
                for node in self.mapping[dep]:
                    s += '{} -> {};\n'.format(node, name)


        nodes = list(set(node for dep in deps for node in self.mapping[dep]))

        if self.args.transitive_reduce:
            reachable = [p for node in nodes for p in self.reachable[node]]
            nodes = [n for n in nodes if n not in reachable]
            self.reachable[key] = set(nodes) | set(reachable)

        for node in nodes:
            if epoch == self.node_epoch[node] or True:
                constraint = 'true'
            else:
                constraint = 'false'

            s += f'{node} -> {key} [constraint={constraint}]\n'

        self.mapping[id] = [key]
        self.node_epoch[key] = epoch

        if 'send' == kind or 'recv' == kind:
            tag = op['task']['tag']

            if tag in self.transfers:
                src = self.mapping[id]
                dst = self.mapping[self.transfers[tag]]

                if 'recv' in kind:
                    src, dst = dst, src

                for a in src:
                    for b in dst:
                        s += f'{a} -> {b} [style=dotted, constraint=false];\n'

            else:
                self.transfers[tag] = id

        if self.last_op != None:
            pass #s += '{} -> op_{} [style=invis]\n'.format(self.last_op, id)
        self.last_op = key

        if self.args.include_flows:
            return 'subgraph cluster_{} {{ style=invis; {} }}'.format(id, s)
        else:
            return s


    def to_string(self):
        output = 'digraph { \n'
        output += 'rankdir=LR;\n'
        output += 'compound=true;\n'
        output += '\n'.join(self.clusters)
        output += '\n}\n'
        return output

    def write_to_file(self, filename):
        with open(filename, 'w') as handle:
            handle.write(self.to_string())

    def write_to_image(self, filename, format):
        with tempfile.NamedTemporaryFile() as handle:
            self.write_to_file(handle.name)
            handle.flush()

            subprocess.run(['dot', '-T', format, '-o', filename, handle.name])






def main():
    parser = argparse.ArgumentParser(description='visualize a execution plan.')
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--no-outline', default=False, action='store_true')
    parser.add_argument('--hide-chunks', default=False, action='store_true')
    parser.add_argument('--include-flows', default=False, action='store_true')
    parser.add_argument('--include-joins', default=False, action='store_true')
    parser.add_argument('--include-events', default=False, action='store_true')
    parser.add_argument('--short-labels', default=False, action='store_true')
    parser.add_argument('--transitive-reduce', '--simplify', default=False, action='store_true')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=1000, type=int)
    args = parser.parse_args()

    writer = DotWriter(args)

    with open(args.input) as handle:
        for (index, line) in enumerate(handle):
            writer.process_line(json.loads(line))

    if args.output.endswith('.dot'):
        writer.write_to_file(args.output)
    elif args.output.endswith('.png'):
        writer.write_to_image(args.output, 'png')
    elif args.output.endswith('.pdf'):
        writer.write_to_image(args.output, 'pdf')
    else:
        print(f'error: {args.output} is not a valid output file.')
        return

    print(f'saved as {args.output}')


if __name__ == '__main__':
    main()
