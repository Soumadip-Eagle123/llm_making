import math, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from operation import Value
from visualieop import draw_dot

ACTS = {'relu': lambda v: v.relu(), 'tanh': lambda v: v.tanh(), 'linear': lambda v: v}


def parse_config(r):
    def num(k, lo, hi, cast=int):
        try:
            v = cast(r[k])
        except (KeyError, ValueError, TypeError):
            raise ValueError(f'{k} must be a number')
        if not lo <= v <= hi:
            raise ValueError(f'{k} must be between {lo} and {hi}')
        return v
    nin, nout = num('nin', 1, 6), num('nout', 1, 3)
    lr, seed = num('lr', 1e-6, 10, float), num('seed', 0, 10**6)
    try:
        hidden = [int(t) for t in str(r.get('hidden', '')).replace(' ', '').split(',') if t]
    except ValueError:
        raise ValueError('hidden layers: comma-separated integers, e.g. 4,4')
    if len(hidden) > 4 or any(not 1 <= h <= 8 for h in hidden):
        raise ValueError('use at most 4 hidden layers of 1-8 neurons each')
    act = r.get('act')
    if act not in ('relu', 'tanh'):
        raise ValueError('activation must be relu or tanh')
    if r.get('loss') not in ('mse', 'sse'):
        raise ValueError('loss must be mse or sse')
    xs, ys = [], []
    for ln, line in enumerate(str(r.get('data', '')).strip().splitlines(), 1):
        if not line.strip():
            continue
        try:
            vals = [float(t) for t in line.replace(',', ' ').split()]
        except ValueError:
            raise ValueError(f'data line {ln}: non-numeric value')
        if len(vals) != nin + nout:
            raise ValueError(f'data line {ln}: expected {nin + nout} numbers '
                             f'({nin} inputs + {nout} targets), got {len(vals)}')
        xs.append(vals[:nin]); ys.append(vals[nin:])
    if not 1 <= len(xs) <= 8:
        raise ValueError('provide 1-8 training examples')
    return dict(nin=nin, nout=nout, hidden=hidden, act=act, loss=r['loss'], lr=lr, seed=seed,
                out_act='linear' if r.get('out_act') == 'linear' else act, xs=xs, ys=ys)


class Neuron:
    def __init__(s, nin, act, tag):
        s.tag, s.act, s.zs = tag, act, []
        s.w = [Value(random.uniform(-1, 1), label=f'{tag}.w{i}') for i in range(nin)]
        s.b = Value(random.uniform(-1, 1), label=f'{tag}.b')

    def __call__(s, x):
        z = sum((w * xi for w, xi in zip(s.w, x)), s.b)   
        z.label = f'{s.tag}.z'
        s.zs.append(z.data)
        out = ACTS[s.act](z)
        if s.act != 'linear':
            out.label = f'{s.tag}.a'
        return out

    def parameters(s):
        return s.w + [s.b]


class Model:
    def __init__(s, c):
        random.seed(c['seed'])
        sz = [c['nin']] + c['hidden'] + [c['nout']]
        s.layers = []
        for l in range(len(sz) - 1):
            act = c['out_act'] if l == len(sz) - 2 else c['act']
            s.layers.append([Neuron(sz[l], act, f'L{l}N{j}') for j in range(sz[l + 1])])

    def __call__(s, x):
        for layer in s.layers:
            x = [n(x) for n in layer]
        return x

    def parameters(s):
        return [p for L in s.layers for n in L for p in n.parameters()]


def collect(root):
    seen, stack = {}, [root]
    while stack:
        v = stack.pop()
        if id(v) not in seen:
            seen[id(v)] = v
            stack.extend(v._prev)
    return sorted(seen.values(), key=lambda v: v._id)   


class Session:
    def __init__(s, c):
        s.c, s.m, s.dot_src = c, Model(c), None
        s.params = s.m.parameters()
        s.initial = {p.label: p.data for p in s.params}

    def _build(s):
        c = s.c
        for L in s.m.layers:
            for n in L:
                n.zs = []
        terms, ex = [], []
        for e, (xv, yv) in enumerate(zip(c['xs'], c['ys'])):
            out = s.m([Value(v, label=f'x{i}[{e}]') for i, v in enumerate(xv)])
            sq = []
            for j, (o, y) in enumerate(zip(out, yv)):
                err = o - Value(y, label=f'y{j}[{e}]'); err.label = f'err{j}[{e}]'
                q = err ** 2; q.label = f'sq{j}[{e}]'
                sq.append(q)
            ex.append((out, yv, sq)); terms += sq
        total = terms[0]
        for q in terms[1:]:
            total = total + q
        loss = total * (1.0 / len(terms)) if c['loss'] == 'mse' else total
        loss.label = 'loss'
        return loss, ex

    def iteration(s):
        """forward + loss + gradient reset + backward. Does NOT touch parameters."""
        loss, ex = s._build()
        if not math.isfinite(loss.data):
            raise ValueError('Loss diverged (not finite). Lower the learning rate and rebuild.')
        nodes = collect(loss)
        if len(nodes) > 900:
            raise ValueError(f'graph has {len(nodes)} nodes (limit 900); shrink the network or dataset')
        idx = {id(v): i for i, v in enumerate(nodes)}
        for i, v in enumerate(nodes):
            v.label = v.label or f'{v._op or "c"}{i}'
        par = lambda v: sorted(v._prev, key=lambda p: p._id)

        events = []
        for i, v in enumerate(nodes):
            if v._op:
                args = ', '.join(f'{p.label}={p.data:.4f}' for p in par(v))
                events.append({'phase': 'forward', 'node': i,
                               'text': f'{v.label} = {v._op}({args}) = {v.data:.4f}'})
        events.append({'phase': 'loss', 'node': len(nodes) - 1,
                       'text': f'{s.c["loss"].upper()} loss = {loss.data:.4f}'})

        for v in nodes:                      
            v.grad = 0.0
        loss.grad = 1.0
        events += [{'phase': 'zero'}, {'phase': 'seed', 'node': idx[id(loss)]}]
        for i in range(len(nodes) - 1, -1, -1):  
            v = nodes[i]
            if not v._prev:
                continue
            before = {id(p): p.grad for p in v._prev}
            v._backward()
            contrib = []
            for p in par(v):
                d = p.grad - before[id(p)]
                contrib.append({'to': idx[id(p)], 'delta': d, 'local': d / v.grad if v.grad else None})
            events.append({'phase': 'backward', 'node': i, 'grad_in': v.grad, 'contrib': contrib})

        pids = {id(p) for p in s.params}
        def kind(v):
            if id(v) in pids: return 'param'
            if v._op: return 'op'
            return {'x': 'input', 'y': 'target'}.get(v.label[0], 'const')
        gj = [{'id': i, 'label': v.label, 'op': v._op, 'data': v.data, 'grad': v.grad,
               'kind': kind(v), 'parents': [idx[id(p)] for p in par(v)]} for i, v in enumerate(nodes)]
        dead = {n.tag: sum(z <= 0 for z in n.zs) / len(n.zs)
                for L in s.m.layers for n in L if n.act == 'relu'}
        exj = [{'pred': [o.data for o in out], 'target': yv,
                'sq': sum(q.data for q in sq), 'node': idx[id(sq[0])]} for out, yv, sq in ex]

        dot = draw_dot(loss)
        dot.attr(label=f'loss={loss.data:.4f}  act={s.c["act"]}  lr={s.c["lr"]}  nodes={len(nodes)}',
                 labelloc='t', bgcolor='white', dpi='150')
        s.dot_src = dot.source              
        return {'graph': {'nodes': gj}, 'events': events, 'loss': loss.data, 'dead': dead,
                'examples': exj, 'params': {p.label: p.data for p in s.params}}

    def update(s):
        lr, ev = s.c['lr'], []
        for p in s.params:
            old = p.data
            p.data += -lr * p.grad
            ev.append({'phase': 'update', 'label': p.label, 'old': old, 'grad': p.grad,
                       'lr': lr, 'new': p.data, 'delta': p.data - old})
        return ev