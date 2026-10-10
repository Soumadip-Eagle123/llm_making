import os, shutil, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from operation import Value
from engine import Session, parse_config

CFG = dict(nin=2, nout=1, hidden='3', act='relu', out_act='linear', loss='mse', lr=0.05, seed=1,
           data='1,2,1\n2,-1,-1\n0.5,1,1')


class T(unittest.TestCase):
    def test_add_mul(self):
        a, b = Value(2.), Value(3.); c = a * b + a; c.backward()
        self.assertEqual((a.grad, b.grad), (4., 2.))

    def test_repeated_operand(self):
        x = Value(3.); y = x * x; y.backward(); self.assertEqual(x.grad, 6.)

    def test_multi_path(self):
        x = Value(3.); t = x * 2; y = t + t * x; y.backward(); self.assertEqual(x.grad, 14.)

    def test_relu(self):
        for x, d in ((2., 1.), (-2., 0.), (0., 0.)):
            v = Value(x); o = v.relu(); o.backward()
            self.assertEqual((o.data, v.grad), (max(0., x), d))

    def test_accumulate_and_reset(self):
        x = Value(2.); y = x * x; y.backward(); y.backward()
        self.assertEqual(x.grad, 8.)
        x.grad = 0.0; y.backward(); self.assertEqual(x.grad, 4.)

    def test_known_loss(self):
        s = Session(parse_config(dict(CFG, nin=1, hidden='', data='1,3\n2,3')))
        w, b = s.params; w.data, b.data = 2., 0.        # preds 2, 4 vs target 3
        self.assertAlmostEqual(s.iteration()['loss'], 1.0)

    def test_finite_difference(self):
        for act in ('relu', 'tanh'):
            s = Session(parse_config(dict(CFG, act=act)))
            s.iteration()
            for p in s.params:
                g, h, d0 = p.grad, 1e-6, p.data
                p.data = d0 + h; up = s._build()[0].data
                p.data = d0 - h; dn = s._build()[0].data
                p.data = d0
                self.assertAlmostEqual(g, (up - dn) / (2 * h), places=4)

    def test_update_and_training(self):
        s = Session(parse_config(CFG))
        first = s.iteration()['loss']
        old = [(p.data, p.grad) for p in s.params]
        s.update()
        for p, (d, g) in zip(s.params, old):
            self.assertAlmostEqual(p.data, d - 0.05 * g)
        for _ in range(30):
            last = s.iteration()['loss']; s.update()
        self.assertLess(last, first)

    @unittest.skipUnless(shutil.which('dot'), 'graphviz not installed')
    def test_jpeg(self):
        import graphviz
        s = Session(parse_config(CFG)); s.iteration()
        self.assertEqual(graphviz.Source(s.dot_src, format='jpg').pipe()[:2], b'\xff\xd8')

    def test_validation(self):
        with self.assertRaises(ValueError):
            parse_config(dict(CFG, data='1,2'))


if __name__ == '__main__':
    unittest.main()