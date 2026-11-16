import math
import unittest
import numpy as np
import torch
from jace_rec import gating as C
from jace_rec.decoding import mixed_log_values
from jace_rec.metrics import all_metrics
from jace_rec.tree import PathMass, conditionals

class MechanismTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.paths=[[0,2,4,1],[0,2,5,1],[0,3,1],[0,3,7,8,1]]
        self.tree=PathMass(self.paths,'cpu')
        self.experts=np.array([[.7,.8],[.4,.6],[.9,.2],[.2,.7]])
        self.x,self.seen=C.features(self.experts,[1,3,1])


    def test_group_probability_and_every_prefix_reconstruction(self):
        head=C.Head('joint_user_alpha')
        scores,p=C.catalog_scores(head,self.x,self.experts,self.seen);q=conditionals(self.tree.index,scores)
        reconstructed=np.array([math.exp(sum(q[tuple(path[:t])][path[t]] for t in range(1,len(path)))) for path in self.paths])
        np.testing.assert_allclose(reconstructed,np.exp(20*scores),atol=2e-12)
        self.assertAlmostEqual(reconstructed.sum(),1.);self.assertAlmostEqual(reconstructed[self.seen].sum(),p['root_rho'])


    def test_mixture_preserves_mass_and_alpha_zero(self):
        native=np.log([.12,.08]);q=np.log([.7,.3])
        np.testing.assert_array_equal(mixed_log_values(native,np.log(.2),q,0),native)
        for alpha in (.01,.3,.9):
            p=np.exp(mixed_log_values(native,np.log(.2),q,alpha));self.assertAlmostEqual(p.sum(),.2)


    def test_joint_loss_matches_direct_probability_calculation_and_gradients(self):
        head=C.Head('joint_user_alpha')
        batch=dict(features=torch.tensor(np.stack([self.x,self.x])),experts=torch.tensor(np.stack([self.experts]*2)),
            seen=torch.tensor(np.stack([self.seen]*2)),targets=torch.tensor([0,3]),valid=self.tree.valid[[0,3]],
            native=torch.full((2,4),-3.,dtype=torch.float64),legal=torch.full((2,4),-.4,dtype=torch.float64))
        values=C.loss(head,batch,self.tree.ancestors);scores,p=C.catalog_scores(head,self.x,self.experts,self.seen)
        q=conditionals(self.tree.index,scores);expected=[]
        for item in (0,3):
            path=self.paths[item]
            mixed=[.7*math.exp(-3)+.3*math.exp(-.4+q[tuple(path[:t])][path[t]]) for t in range(1,len(path))]
            expected.append(-np.log(mixed).mean())
        np.testing.assert_allclose(values.detach().numpy(),expected,atol=1e-12)
        values.mean().backward()
        self.assertTrue((head.beta_rho.bias.grad.abs()>1e-10).all());self.assertGreater(abs(float(head.alpha.bias.grad)),1e-10)


    def test_parameter_count_and_shared_initial_distribution(self):
        reference=None
        for arm,count in [('joint_user_alpha',18)]:
            head=C.Head(arm);self.assertEqual(sum(p.numel() for p in head.parameters()),count)
            score,p=C.catalog_scores(head,self.x,self.experts,self.seen)
            np.testing.assert_allclose(list(p.values()),[.5,.01,.3])
            if reference is not None:np.testing.assert_array_equal(reference,score)
            reference=score


    def test_single_target_metrics(self):
        m=all_metrics([1,3,8,1<<30]);self.assertEqual(m['R5'],.5);self.assertEqual(m['R10'],.75)
        self.assertAlmostEqual(m['N5'],(1+.5)/4)
